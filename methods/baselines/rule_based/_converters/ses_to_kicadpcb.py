#!/usr/bin/env python3
"""SES (freerouting Specctra session) -> kicad_pcb converter (text-based, no pcbnew).

Reads a Specctra session file (.ses) emitted by freerouting and the original
unrouted .kicad_pcb, and emits a routed .kicad_pcb by injecting (segment ...) /
(via ...) tokens into the source PCB just before its closing ')'.

KiCad 9 PCB files are S-expression text; this is a lossless append. The SES
parser reads each (net <name> ...) under (network_out) and converts:
  - (wire (path <layer> <width> x1 y1 ... xn yn)) -> n-1 (segment ...) tokens
  - (via "<padstack>" x y)                         -> 1 (via ...) token

Coordinate convention: the SES files generated for these boards declare
`(resolution um 10)`, i.e. one SES unit = 0.1 um. Therefore mm = unit / 10000.
SES Y axis is inverted relative to KiCad; we negate Y when emitting.

Via diameter/drill come from the padstack name `Via[0-1]_<dia>:<drill>_um`
(values in um). If the name doesn't match, we fall back to the library_out
circle radius (per Specctra: `(circle <layer> <diameter> ...)`).
"""
import argparse
import json
import re
import sys
import uuid
from pathlib import Path


# ---------- minimal Specctra/SES tokenizer + tree builder ----------

_TOKEN_RE = re.compile(r'''
    \s+                       |  # whitespace
    \(                        |  # open
    \)                        |  # close
    "(?:[^"\\]|\\.)*"         |  # quoted string
    [^\s()"]+                    # bareword
''', re.VERBOSE)


def _tokenize(text: str):
    pos = 0
    n = len(text)
    out = []
    for m in _TOKEN_RE.finditer(text):
        if m.start() != pos:
            raise SyntaxError(f'Unexpected character at {pos}: {text[pos:pos+20]!r}')
        pos = m.end()
        tok = m.group(0)
        if tok.isspace():
            continue
        out.append(tok)
    if pos != n:
        raise SyntaxError(f'Trailing junk at {pos}: {text[pos:pos+20]!r}')
    return out


def _parse(tokens, i=0):
    """Parse tokens[i:] as an S-expression tree, returning (node, next_index).
    Lists are python lists; atoms are python strings (quoted strings include quotes).
    """
    if i >= len(tokens):
        raise SyntaxError('Unexpected EOF')
    t = tokens[i]
    if t == '(':
        i += 1
        node = []
        while i < len(tokens) and tokens[i] != ')':
            child, i = _parse(tokens, i)
            node.append(child)
        if i >= len(tokens):
            raise SyntaxError('Unclosed list')
        return node, i + 1
    elif t == ')':
        raise SyntaxError('Unexpected )')
    else:
        return t, i + 1


def parse_ses(text: str):
    tokens = _tokenize(text)
    tree, _ = _parse(tokens, 0)
    return tree


def head(node):
    return node[0] if isinstance(node, list) and node else None


def find_child(node, name):
    if not isinstance(node, list):
        return None
    for c in node[1:]:
        if isinstance(c, list) and head(c) == name:
            return c
    return None


def find_children(node, name):
    if not isinstance(node, list):
        return []
    return [c for c in node[1:] if isinstance(c, list) and head(c) == name]


def unquote(s: str) -> str:
    if isinstance(s, str) and len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return bytes(s[1:-1], 'utf-8').decode('unicode_escape')
    return s


# ---------- conversion ----------

def parse_net_map(pcb_text: str):
    nets = {}
    for m in re.finditer(r'\(net\s+(\d+)\s+"([^"]*)"\)', pcb_text):
        nid = int(m.group(1))
        name = m.group(2)
        if name and name not in nets:
            nets[name] = nid
    return nets


def parse_resolution(node):
    """(resolution <unit> <value>) -> ses_unit_in_mm."""
    if node is None:
        return 1.0 / 10000.0  # default: um 10
    unit = node[1]
    value = float(node[2])
    if unit == 'um':
        return (1.0 / value) / 1000.0       # 1 unit = (1/value) um -> mm
    if unit == 'mm':
        return 1.0 / value                  # 1 unit = (1/value) mm
    if unit == 'cm':
        return 10.0 / value                 # 1 unit = (1/value) cm = 10/value mm
    if unit == 'inch':
        return 25.4 / value                 # 1 unit = (1/value) inch
    if unit == 'mil':
        return 0.0254 / value
    raise ValueError(f'Unknown resolution unit: {unit}')


def parse_padstack_dim(name: str):
    """`Via[0-1]_<dia>:<drill>_um` -> (dia_mm, drill_mm) or (None, None)."""
    m = re.search(r'_(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)_um', name)
    if not m:
        return None, None
    return float(m.group(1)) / 1000.0, float(m.group(2)) / 1000.0


def parse_via_dim_from_library(library_out, padstack_name: str, ses_unit_mm: float):
    """Fallback: read padstack circle in library_out -> diameter (mm)."""
    if library_out is None:
        return None
    for pad in find_children(library_out, 'padstack'):
        nm = unquote(pad[1]) if len(pad) > 1 and isinstance(pad[1], str) else ''
        if nm != padstack_name:
            continue
        for sh in find_children(pad, 'shape'):
            for circ in find_children(sh, 'circle'):
                # (circle <layer> <diameter> [x y])
                if len(circ) >= 3:
                    return float(circ[2]) * ses_unit_mm
    return None


def fmt_num(v):
    s = f'{float(v):.6f}'
    s = s.rstrip('0').rstrip('.')
    return s if s else '0'


def fmt_seg(x1, y1, x2, y2, width, layer, net_id):
    return (
        '  (segment\n'
        f'    (start {fmt_num(x1)} {fmt_num(y1)})\n'
        f'    (end {fmt_num(x2)} {fmt_num(y2)})\n'
        f'    (width {fmt_num(width)})\n'
        f'    (layer "{layer}")\n'
        f'    (net {net_id})\n'
        f'    (uuid "{uuid.uuid4()}")\n'
        '  )'
    )


def fmt_via(x, y, size, drill, layer_a, layer_b, net_id):
    return (
        '  (via\n'
        f'    (at {fmt_num(x)} {fmt_num(y)})\n'
        f'    (size {fmt_num(size)})\n'
        f'    (drill {fmt_num(drill)})\n'
        f'    (layers "{layer_a}" "{layer_b}")\n'
        f'    (net {net_id})\n'
        f'    (uuid "{uuid.uuid4()}")\n'
        '  )'
    )


def convert(ses_path: Path, base_pcb: Path, out_pcb: Path):
    ses_text = ses_path.read_text(encoding='utf-8')
    base_text = base_pcb.read_text(encoding='utf-8')
    net_map = parse_net_map(base_text)

    tree = parse_ses(ses_text)
    if head(tree) != 'session':
        raise RuntimeError(f'Not a Specctra session file: {ses_path}')

    routes = find_child(tree, 'routes')
    if routes is None:
        raise RuntimeError(f'No (routes ...) section: {ses_path}')

    # Resolution applies to coordinates inside routes (and library_out shapes).
    ses_unit_mm = parse_resolution(find_child(routes, 'resolution'))
    library_out = find_child(routes, 'library_out')
    network_out = find_child(routes, 'network_out')
    if network_out is None:
        raise RuntimeError(f'No (network_out ...) section: {ses_path}')

    # (via ...) entries inside (net ...) cite a padstack name; cache lookups.
    via_dim_cache = {}

    def via_dim(name: str):
        if name in via_dim_cache:
            return via_dim_cache[name]
        dia, drill = parse_padstack_dim(name)
        if dia is None:
            dia = parse_via_dim_from_library(library_out, name, ses_unit_mm)
            drill = None  # library shape doesn't carry drill
        if dia is None:
            dia = 0.6
        if drill is None:
            drill = max(dia * 0.5, 0.3)
        via_dim_cache[name] = (dia, drill)
        return dia, drill

    out_blocks = []
    track_count = 0
    via_count = 0
    unmapped_nets = set()

    for net_node in find_children(network_out, 'net'):
        net_name = unquote(net_node[1]) if len(net_node) > 1 else ''
        net_id = net_map.get(net_name)
        if net_id is None:
            unmapped_nets.add(net_name)
            net_id = 0  # NoNet — keeps geometry visible

        for wire in find_children(net_node, 'wire'):
            path = find_child(wire, 'path')
            if path is None:
                continue
            # path = ['path', layer, width, x1, y1, x2, y2, ...]
            if len(path) < 5:
                continue
            layer = path[1]
            width_units = float(path[2])
            width_mm = width_units * ses_unit_mm
            coords = [float(c) for c in path[3:]]
            if len(coords) % 2 != 0 or len(coords) < 4:
                continue
            pts = list(zip(coords[0::2], coords[1::2]))
            for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
                x1 = xa * ses_unit_mm
                y1 = -ya * ses_unit_mm
                x2 = xb * ses_unit_mm
                y2 = -yb * ses_unit_mm
                if abs(x1 - x2) < 1e-9 and abs(y1 - y2) < 1e-9:
                    continue
                out_blocks.append(fmt_seg(x1, y1, x2, y2, width_mm, layer, net_id))
                track_count += 1

        for via in find_children(net_node, 'via'):
            # (via "<padstack>" x y [other...])
            if len(via) < 4:
                continue
            name = unquote(via[1])
            x_units = float(via[2])
            y_units = float(via[3])
            dia, drill = via_dim(name)
            x = x_units * ses_unit_mm
            y = -y_units * ses_unit_mm
            out_blocks.append(fmt_via(x, y, dia, drill, 'F.Cu', 'B.Cu', net_id))
            via_count += 1

    rstripped = base_text.rstrip()
    if not rstripped.endswith(')'):
        raise RuntimeError(f'Unexpected PCB tail in {base_pcb}: does not end with ")"')
    body = rstripped[:-1].rstrip()
    injected = body + '\n\n' + '\n'.join(out_blocks) + '\n)\n'
    out_pcb.parent.mkdir(parents=True, exist_ok=True)
    out_pcb.write_text(injected, encoding='utf-8')

    return {
        'tracks_written': track_count,
        'vias_written': via_count,
        'nets_mapped': len(net_map),
        'unmapped_nets': sorted(unmapped_nets),
        'ses_unit_mm': ses_unit_mm,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('ses')
    ap.add_argument('base_pcb')
    ap.add_argument('output')
    args = ap.parse_args()
    stats = convert(Path(args.ses), Path(args.base_pcb), Path(args.output))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
