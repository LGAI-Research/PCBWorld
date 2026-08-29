"""KiCad-cli based PCB renderer.

Rendering pipeline:
    engine.save(temp.kicad_pcb) -> kicad-cli export -> raster -> np.ndarray

Supports three render modes:
    - "pdf": kicad-cli pcb export pdf -> pypdfium2 -> PNG (default)
    - "render3d": kicad-cli pcb render --side top -> PNG directly
    - "gui_capture": Open KiCad GUI and capture window via macOS screencapture
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from pcb_world.engine.kicad_engine import KiCadEngine


class PCBRenderer:
    """KiCad-cli based real PCB renderer.

    Each step:
    1. KiCadEngine.save() writes current board to a temp .kicad_pcb
    2. kicad-cli pcb export pdf generates a PDF
    3. pypdfium2 rasterises the first page to PNG
    4. Returns numpy array

    Args:
        kicad_cli: Path to kicad-cli executable.
        layers: PCB layers to render.
        dpi: Rasterisation resolution (PDF conversion).
        figsize: Target image size (width, height) in pixels.
        mode: Default render mode — "pdf", "render3d" or "gui_capture".
    """

    KICAD_CLI = os.environ.get("KICAD_CLI", shutil.which("kicad-cli") or "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli")
    DEFAULT_LAYERS = ["F.Cu", "B.Cu", "F.SilkS", "Edge.Cuts"]
    VALID_MODES = ("render3d", "pdf", "gui_capture")

    def __init__(
        self,
        kicad_cli: str | None = None,
        layers: list[str] | None = None,
        dpi: int = 150,
        figsize: tuple[int, int] = (800, 800),
        mode: str = "pdf",
    ) -> None:
        self.kicad_cli = kicad_cli or self.KICAD_CLI
        self.layers = layers or self.DEFAULT_LAYERS
        self.dpi = dpi
        self.figsize = figsize
        self.mode = mode
        self._temp_dir: str | None = None

    def _get_temp_dir(self) -> str:
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="pcb_render_")
        return self._temp_dir

    # ------------------------------------------------------------------
    # Multi-mode rendering from .kicad_pcb file (no C++ engine needed)
    # ------------------------------------------------------------------

    def render_from_file(
        self,
        pcb_path: str,
        mode: str | None = None,
        step_info: dict | None = None,
    ) -> np.ndarray:
        """Render a .kicad_pcb file using the specified mode.

        Args:
            pcb_path: Path to .kicad_pcb file.
            mode: "pdf", "render3d" or "gui_capture". Uses self.mode if
                None.
            step_info: Optional overlay metadata dict.

        Returns:
            RGB numpy array (H, W, 3) uint8.

        Raises:
            RuntimeError: If kicad-cli is not found or rendering fails.
        """
        mode = mode or self.mode
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}, must be one of {self.VALID_MODES}")

        if mode == "render3d":
            image = self._render_3d(pcb_path)
        elif mode == "pdf":
            image = self._render_pdf(pcb_path)
        else:
            image = self._render_gui_capture(pcb_path)

        if step_info:
            image = self._add_overlay(image, step_info)
        return image

    def render_from_file_timed(
        self,
        pcb_path: str,
        mode: str | None = None,
    ) -> tuple[np.ndarray, float]:
        """Render and return (image, elapsed_seconds)."""
        t0 = time.time()
        img = self.render_from_file(pcb_path, mode=mode)
        return img, time.time() - t0

    def export_svg(
        self,
        pcb_path: str,
        output_svg: str,
        layers: list[str] | None = None,
    ) -> str:
        """Export board to clean SVG file (no drawing sheet, board area only).

        Args:
            pcb_path: Path to .kicad_pcb file.
            output_svg: Destination SVG path.
            layers: Override layers to export. Defaults to self.layers.

        Returns:
            The output_svg path written to.
        """
        export_layers = layers or self.layers
        cmd = [
            self.kicad_cli, "pcb", "export", "svg",
            "--layers", ",".join(export_layers),
            "--page-size-mode", "2",
            "--exclude-drawing-sheet",
            pcb_path,
            "-o", output_svg,
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        return output_svg

    def _render_3d(self, pcb_path: str) -> np.ndarray:
        """Render via kicad-cli pcb render (3D raytracer)."""
        temp_dir = self._get_temp_dir()
        temp_png = os.path.join(temp_dir, "render3d.png")
        w, h = self.figsize

        cmd = [
            self.kicad_cli, "pcb", "render",
            "--side", "top",
            "--width", str(w),
            "--height", str(h),
            "--quality", "high",
            pcb_path,
            "-o", temp_png,
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)

        img = Image.open(temp_png).convert("RGB")
        img = img.resize((w, h), Image.LANCZOS)
        return np.array(img, dtype=np.uint8)

    def _render_pdf(self, pcb_path: str) -> np.ndarray:
        """Render via kicad-cli pcb export pdf -> pypdfium2 -> PNG."""
        # Lazy import: SVG-export-only users don't need the rasteriser; a
        # missing pypdfium2 (a pinned dependency) fails loudly here.
        import pypdfium2 as pdfium

        temp_dir = self._get_temp_dir()
        temp_pdf = os.path.join(temp_dir, "export.pdf")
        w, h = self.figsize

        cmd = [
            self.kicad_cli, "pcb", "export", "pdf",
            "--layers", ",".join(self.layers),
            pcb_path,
            "-o", temp_pdf,
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)

        # PDFium renders at a scale factor; PDF user space is 72 units/inch.
        doc = pdfium.PdfDocument(temp_pdf)
        try:
            img = doc[0].render(scale=self.dpi / 72.0).to_pil()
        finally:
            doc.close()
        img = img.convert("RGB").resize((w, h), Image.LANCZOS)
        return np.array(img, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Engine-based rendering (board taken from a live KiCadEngine)
    # ------------------------------------------------------------------

    def render(self, engine: KiCadEngine, step_info: dict | None = None) -> np.ndarray:
        """Render current board state via the self.mode kicad-cli pipeline.

        Args:
            engine: KiCadEngine instance (uses save() to write board).
            step_info: Optional overlay metadata dict.

        Returns:
            RGB numpy array (H, W, 3) uint8.
        """
        temp_dir = self._get_temp_dir()
        temp_pcb = os.path.join(temp_dir, "current_board.kicad_pcb")
        engine.save(temp_pcb)
        return self.render_from_file(temp_pcb, step_info=step_info)

    def render_to_file(
        self,
        engine: KiCadEngine,
        output_path: str,
        step_info: dict | None = None,
    ) -> str:
        """Render board directly to a file (PNG or SVG).

        Args:
            engine: KiCadEngine instance.
            output_path: Destination path (.png or .svg).
            step_info: Optional overlay metadata.

        Returns:
            The output_path written to.
        """
        if output_path.endswith(".svg"):
            temp_dir = self._get_temp_dir()
            temp_pcb = os.path.join(temp_dir, "current_board.kicad_pcb")
            engine.save(temp_pcb)
            cmd = [
                self.kicad_cli, "pcb", "export", "svg",
                "--layers", ",".join(self.layers),
                "--page-size-mode", "2",
                temp_pcb,
                "-o", output_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        else:
            frame = self.render(engine, step_info=step_info)
            Image.fromarray(frame).save(output_path)
        return output_path

    PCBNEW_APP = "/Applications/KiCad/KiCad.app/Contents/Applications/pcbnew.app"
    PCBNEW_BIN = "/Applications/KiCad/KiCad.app/Contents/Applications/pcbnew.app/Contents/MacOS/pcbnew"

    def _render_gui_capture(self, pcb_path: str) -> np.ndarray:
        """Open KiCad PCB editor (pcbnew) and capture the window via macOS screencapture.

        macOS only. Requires:
        - Screen recording permission in System Settings > Privacy & Security > Screen Recording
        - pyobjc-framework-Quartz (for CGWindowListCopyWindowInfo)

        Uses Quartz API instead of AppleScript because pcbnew (wxWidgets) does not
        expose windows through the AppleScript accessibility interface.
        """
        import sys
        if sys.platform != "darwin":
            raise RuntimeError("gui_capture mode is only supported on macOS")

        try:
            import Quartz
        except ImportError:
            raise RuntimeError(
                "gui_capture requires pyobjc-framework-Quartz. "
                "Install with: pip install pyobjc-framework-Quartz"
            )

        temp_dir = self._get_temp_dir()
        temp_png = os.path.join(temp_dir, "gui_capture.png")
        w, h = self.figsize

        try:
            # 1. Launch pcbnew binary directly with board file
            #    (open --args doesn't reliably pass the file to pcbnew)
            abs_pcb = os.path.abspath(pcb_path)
            subprocess.Popen(
                [self.PCBNEW_BIN, abs_pcb],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # 2. Wait for GUI to load
            time.sleep(8)

            # 3. Bring pcbnew to front and Zoom to Fit (Home key)
            subprocess.run(
                ["osascript", "-e",
                 'tell application "System Events" to set frontmost of process "pcbnew" to true'],
                capture_output=True, timeout=5,
            )
            time.sleep(0.5)
            subprocess.run(
                ["osascript", "-e",
                 'tell application "System Events" to key code 115'],  # 115 = Home key
                capture_output=True, timeout=5,
            )
            time.sleep(1)

            # 4. Find pcbnew window ID via Quartz CGWindowListCopyWindowInfo
            window_id = None
            for attempt in range(5):
                windows = Quartz.CGWindowListCopyWindowInfo(
                    Quartz.kCGWindowListOptionOnScreenOnly,
                    Quartz.kCGNullWindowID,
                )
                for win in windows:
                    owner = win.get("kCGWindowOwnerName", "")
                    if "pcb" in owner.lower() or "PCB" in owner:
                        bounds = win.get("kCGWindowBounds", {})
                        bw = bounds.get("Width", 0)
                        bh = bounds.get("Height", 0)
                        # Skip tiny auxiliary windows (toolbars, tooltips)
                        if bw > 200 and bh > 200:
                            window_id = str(win["kCGWindowNumber"])
                            break
                if window_id is not None:
                    break
                time.sleep(2)

            if window_id is None:
                raise RuntimeError(
                    "Could not find pcbnew window. "
                    "Ensure KiCad is installed and screen recording permission is granted."
                )

            # 5. Capture window
            subprocess.run(
                ["screencapture", "-l", window_id, "-o", temp_png],
                check=True, capture_output=True, timeout=10,
            )

            # 6. Read and resize
            img = Image.open(temp_png).convert("RGB")
            img = img.resize((w, h), Image.LANCZOS)
            return np.array(img, dtype=np.uint8)

        finally:
            # 7. Quit pcbnew
            try:
                subprocess.run(
                    ["osascript", "-e",
                     'tell application "System Events" to tell process "pcbnew" to set frontmost to true'],
                    capture_output=True, timeout=3,
                )
                subprocess.run(
                    ["osascript", "-e",
                     'tell application "System Events" to keystroke "q" using command down'],
                    capture_output=True, timeout=3,
                )
            except Exception:
                # Fallback: kill process
                try:
                    subprocess.run(["pkill", "-f", "pcbnew"], capture_output=True, timeout=5)
                except Exception:
                    pass

    def _add_overlay(self, image: np.ndarray, step_info: dict) -> np.ndarray:
        """Add text overlay with step info onto the rendered image."""
        try:
            from PIL import ImageDraw, ImageFont
            img = Image.fromarray(image)
            draw = ImageDraw.Draw(img)

            # Build text lines
            lines = [f"{k}: {v}" for k, v in step_info.items()]
            text = "  |  ".join(lines)

            # Use default font
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
            except (OSError, IOError):
                font = ImageFont.load_default()

            # Draw background rectangle + text at top-left
            bbox = draw.textbbox((5, 5), text, font=font)
            draw.rectangle(
                [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
                fill=(0, 0, 0, 180),
            )
            draw.text((5, 5), text, fill=(255, 255, 255), font=font)
            return np.array(img, dtype=np.uint8)
        except Exception:
            return image

    def close(self) -> None:
        """Clean up temporary directory."""
        if self._temp_dir is not None and os.path.isdir(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
