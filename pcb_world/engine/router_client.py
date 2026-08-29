"""Environment-side transport to the GPL engine server (engine/engine_server/).

``KiCadEngine`` swaps its in-process pybind ``RLRouter`` for a
:class:`RouterProxy` when engine IPC is enabled (the default). The proxy
forwards every attribute call over a unix socket to a dedicated server
child process — the only process that loads the GPL ``kicad_rl_router``
shared library — and reconstructs results into the plain wire mirrors of
``wire.py``. Method-call semantics are otherwise identical to the
pybind object, so ``KiCadEngine`` logic is transport-agnostic.

Performance model (spike-validated on the v0.27.1 ipc series):
- small-command roundtrip ≈ 7µs; the cost that matters is CALL COUNT and
  snapshot payloads, so the proxy adds
  * a client-side query cache: pure getters (``get_*``/``is_*``/``was_*``/
    ``has_*``) are cached per (name, args, kwargs) and invalidated by any
    non-getter call — repeated reads within a step cost one RPC total;
  * ``batch_prewarm``: one roundtrip that executes a list of getters
    server-side and seeds the cache, so a fixed getter sequence
    (board snapshot, session state) costs one RPC.
- no per-call timeout: shove has legitimate multi-minute outliers.

Failure policy: a dead server (C++ crash) raises
:class:`EngineServerCrashed` with the last op and the server stderr tail —
never a silent fallback to in-process mode.

Servers are pooled per client process: closing an engine parks its server
(router destroyed, process kept) and the next engine construction reuses
it, so per-board reloads don't pay the ~0.8s spawn+import cost. Parked and
live servers alike exit on client-process death (socket EOF) — no orphans.
"""

from __future__ import annotations

import atexit
import builtins
import os
import pickle
import socket
import struct
import subprocess
import sys
import tempfile
import time
import weakref

from pcb_world.engine.containers import KRL_FIELDS, from_wire, to_wire

_LEN = struct.Struct(">Q")
# Protocol 2 (port v0.31): call/batch frames carry a kwargs dict —
# ("call", (name, args, kwargs)) — for keyword-arg binding calls
# (cleanup_tracks). Must match the server's PROTOCOL_VERSION.
PROTOCOL_VERSION = 2

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def engine_home() -> str:
    """Root of the engine checkout: ``PCBWORLD_ENGINE_HOME``, else ``engine/``.

    The engine is a separate GPLv3 program with its own repository, pinned
    here as the ``engine/`` submodule. Everything that reads its tree — this
    client, ``tools/setup/setup_all.sh``, the build-provenance tests — goes
    through this one contract, so an engine checked out elsewhere works
    without editing anything.
    """
    return os.environ.get(
        "PCBWORLD_ENGINE_HOME", os.path.join(_REPO_ROOT, "engine"))


_SERVER_SCRIPT = os.path.join(
    engine_home(), "engine_server", "rl_engine_server.py")

_ENGINE_MISSING_HINT = (
    "engine server not found at {path}\n"
    "The routing engine is a separate program (GPLv3) and is not part of this\n"
    "repository. Fetch and build it:\n"
    "    git submodule update --init --recursive\n"
    "    BUILD_DIR=\"$PWD/build_rl\" bash engine/build_rl_router.sh\n"
    "or set PCBWORLD_ENGINE_HOME to an existing checkout."
)

_GETTER_PREFIXES = ("get_", "is_", "was_", "has_")

# Getters whose results cannot change over a router's lifetime: no binding
# API mutates pads / the copper stackup / the netlist. Cached once per
# construct, immune to mutator invalidation (the "pads cache" first-order
# mitigation from the spike, handoff §6.4).
_PERSISTENT_GETTERS = frozenset({
    "get_pads", "get_copper_layer_count", "get_board_net_count",
})

# Router-config setters that alter no queryable board/session state — calling
# them must not flush the query cache (set_routing_mode alone fires ~0.5×/step).
_NON_INVALIDATING = frozenset({
    "set_routing_mode", "set_corner_mode", "set_track_width",
    "set_via_diameter", "set_via_drill", "reset_via_mode",
    "set_shove_iter_limit",
})

# Parked servers (router closed, process alive) for reuse. They hold an
# open socket to this process, so they self-terminate when we exit.
_IDLE_SERVERS: list["_ServerConn"] = []
_MAX_IDLE = 2

# Every conn ever created (weak — killed conns drop out on GC). The atexit
# hook tears down whatever is still alive at normal interpreter exit (parked
# AND in-use servers), removing their /tmp/krl_ipc_* dirs. Processes that
# skip atexit (multiprocessing workers exit via os._exit; kill -9) are
# covered by the server's own EOF cleanup of its dir instead.
_LIVE_CONNS: "weakref.WeakSet[_ServerConn]" = weakref.WeakSet()


def _kill_all_conns() -> None:
    for conn in list(_LIVE_CONNS):
        conn.kill()


atexit.register(_kill_all_conns)


def ipc_enabled() -> bool:
    """Engine IPC mode knob (default ON). ``KICAD_ENGINE_IPC=0`` forces the
    legacy in-process pybind path — a debugging/benchmark escape hatch, not
    a fallback: failures in IPC mode never silently switch modes."""
    return os.environ.get("KICAD_ENGINE_IPC", "1").strip().lower() not in (
        "0", "false", "no")


class EngineServerCrashed(RuntimeError):
    """The engine server process died (usually a fatal C++ signal)."""


class _ServerConn:
    """One spawned server process + its socket."""

    def __init__(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="krl_ipc_")
        sock_path = os.path.join(self.tmpdir, "s.sock")
        self.stderr_path = os.path.join(self.tmpdir, "server_stderr.log")
        self._stderr_f = open(self.stderr_path, "ab")
        if not os.path.isfile(_SERVER_SCRIPT):
            raise EngineServerCrashed(
                _ENGINE_MISSING_HINT.format(path=_SERVER_SCRIPT))
        # The engine resolves both of these against ITS OWN repository root,
        # which is the submodule checkout — not this tree, where the build and
        # the crash logs live. Point it at ours unless the caller chose values.
        env = os.environ.copy()
        env.setdefault("CADAGENT_KICAD_RL_BUILD_DIR",
                       os.path.join(_REPO_ROOT, "build_rl"))
        env.setdefault("KICAD_CRASH_LOG_DIR",
                       os.path.join(_REPO_ROOT, "var", "crashlogs"))
        self.proc = subprocess.Popen(
            [sys.executable, _SERVER_SCRIPT, sock_path],
            stdout=subprocess.DEVNULL, stderr=self._stderr_f, env=env,
        )
        _LIVE_CONNS.add(self)    # after Popen: kill() needs self.proc
        deadline = time.monotonic() + 120.0
        while not os.path.exists(sock_path):
            if self.proc.poll() is not None:
                tail = self._stderr_tail()   # read before kill() unlinks it
                self.kill()
                raise EngineServerCrashed(
                    "engine server died during startup\n" + tail)
            if time.monotonic() > deadline:
                self.kill()
                raise EngineServerCrashed("engine server startup timed out")
            time.sleep(0.005)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(sock_path)
        hs = self._recv("<handshake>")
        if hs.get("protocol") != PROTOCOL_VERSION:
            self.kill()
            raise RuntimeError(
                f"engine server protocol {hs.get('protocol')} != client "
                f"{PROTOCOL_VERSION}")
        if hs.get("schema") != KRL_FIELDS:
            self.kill()
            raise RuntimeError(
                "engine IPC constant-handshake failed: server wire schema "
                "differs from pcb_world.engine.wire.KRL_FIELDS — "
                "rebuild/sync the binding and registry")
        self.constants: dict[str, int] = hs["constants"]
        self.pid: int = hs["pid"]

    # --- wire ---

    def request(self, op: str, payload):
        data = pickle.dumps((op, payload), protocol=pickle.HIGHEST_PROTOCOL)
        try:
            self.sock.sendall(_LEN.pack(len(data)) + data)
        except OSError:
            # OSError (not just ConnectionError): after a crash killed this
            # conn, later calls hit the CLOSED socket (EBADF) — those must
            # surface as EngineServerCrashed too, or a broad guard that
            # swallowed the first raise would turn the crash into a naked
            # OSError at the next call site.
            raise self._crashed(op) from None
        reply = self._recv(op)
        if reply.get("ok"):
            return reply["value"]
        exc_cls = getattr(builtins, reply.get("etype", ""), RuntimeError)
        if not (isinstance(exc_cls, type) and issubclass(exc_cls, BaseException)):
            exc_cls = RuntimeError
        raise exc_cls(
            f"{reply.get('msg')}\n--- engine server traceback ---\n"
            f"{reply.get('tb', '')}")

    def _recv(self, op: str):
        hdr = self._recv_exact(_LEN.size, op)
        (n,) = _LEN.unpack(hdr)
        return pickle.loads(self._recv_exact(n, op))

    def _recv_exact(self, n: int, op: str) -> bytes:
        chunks = []
        while n:
            try:
                b = self.sock.recv(min(n, 1 << 20))
            except OSError:      # incl. EBADF on an already-killed conn
                b = b""
            if not b:
                raise self._crashed(op)
            chunks.append(b)
            n -= len(b)
        return b"".join(chunks)

    def _crashed(self, op: str) -> EngineServerCrashed:
        rc = self.proc.poll()
        self.kill()
        return EngineServerCrashed(
            f"engine server (pid {self.pid}, exit {rc}) died during "
            f"{op!r} — usually a fatal C++ signal; see var/crashlogs/ for "
            "the native backtrace.\n" + self._stderr_tail())

    def _stderr_tail(self, limit: int = 4096) -> str:
        try:
            with open(self.stderr_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - limit))
                return "--- server stderr tail ---\n" + f.read().decode(
                    "utf-8", "replace")
        except OSError:
            return "(server stderr unavailable)"

    def alive(self) -> bool:
        if self.proc.poll() is not None:
            return False
        try:
            return self.request("ping", None) == "pong"
        except Exception:  # noqa: BLE001
            return False

    def kill(self) -> None:
        try:
            self.sock.close()
        except Exception:  # noqa: BLE001
            pass
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._stderr_f.close()
        for name in ("s.sock", "server_stderr.log"):
            try:
                os.unlink(os.path.join(self.tmpdir, name))
            except OSError:
                pass
        try:
            os.rmdir(self.tmpdir)
        except OSError:
            pass


class RouterProxy:
    """Drop-in stand-in for the pybind ``RLRouter`` over the IPC boundary."""

    def __init__(self, conn: _ServerConn) -> None:
        # Bypass __setattr__-free plain attrs; proxy is a normal object.
        self._conn = conn
        self._cache: dict[tuple, object] = {}
        self._persistent: dict[tuple, object] = {}
        self.constants = conn.constants

    # --- call plumbing ---

    @staticmethod
    def _is_getter(name: str) -> bool:
        return name.startswith(_GETTER_PREFIXES)

    def _decode(self, wire):
        return from_wire(wire)

    def _cached_return(self, value):
        # Hand out shallow copies of cached lists so a consumer-side
        # mutation (sort/append) can never poison the cache.
        return list(value) if isinstance(value, list) else value

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        conn = self._conn
        cache = self._cache
        is_getter = self._is_getter(name)

        persistent = self._persistent if name in _PERSISTENT_GETTERS else None

        def call(*args, **kwargs):
            if is_getter:
                # Getters take hashable (primitive) args/kwargs only.
                key = (name, args, tuple(sorted(kwargs.items())))
                store = persistent if persistent is not None else cache
                if key in store:
                    return self._cached_return(store[key])
            elif name not in _NON_INVALIDATING:
                cache.clear()
            value = self._decode(conn.request("call", (
                name, tuple(to_wire(a) for a in args),
                {k: to_wire(v) for k, v in kwargs.items()})))
            if is_getter:
                (persistent if persistent is not None else cache)[key] = value
                return self._cached_return(value)
            return value

        return call

    def batch_prewarm(self, calls) -> None:
        """One roundtrip executing pure getters server-side, seeding the cache.

        ``calls`` = [(name, args_tuple), ...]; every name must be a getter
        (asserted — mutators in a prewarm would corrupt cache coherence).
        Already-cached entries are skipped; no RPC if everything is warm.
        """
        todo = []
        for name, args in calls:
            assert self._is_getter(name), f"batch_prewarm on non-getter {name}"
            store = (self._persistent if name in _PERSISTENT_GETTERS
                     else self._cache)
            if (name, tuple(args), ()) not in store:
                todo.append((name, tuple(args)))
        if not todo:
            return
        values = self._conn.request(
            "batch",
            [(n, tuple(to_wire(a) for a in args), {}) for n, args in todo])
        for (name, args), wire in zip(todo, values):
            store = (self._persistent if name in _PERSISTENT_GETTERS
                     else self._cache)
            store[(name, args, ())] = self._decode(wire)

    def module_call(self, name: str, *args):
        return self._decode(self._conn.request("module_call", (name, args)))

    # --- lifecycle ---

    def release(self) -> None:
        """Destroy the remote router and park the server for reuse."""
        conn, self._conn = self._conn, None
        self._cache.clear()
        if conn is None:
            return
        try:
            conn.request("close_router", None)
        except Exception:  # noqa: BLE001 — crashed server: drop it
            conn.kill()
            return
        if len(_IDLE_SERVERS) < _MAX_IDLE:
            _IDLE_SERVERS.append(conn)
        else:
            conn.kill()


def acquire_router(
    board_path: str, project_path: str, seed: int,
    shove_iter_limit: int, followbranch_iter_limit: int,
) -> tuple[RouterProxy, str]:
    """Spawn-or-reuse a server, construct the remote router.

    Returns ``(proxy, board_path)``. The server opens the source file
    directly under the strict load contract (no upgrade/normalize layer —
    develop v0.28+); the caller runs the post-load contract checks through
    the proxy.
    """
    conn: _ServerConn | None = None
    while _IDLE_SERVERS:
        cand = _IDLE_SERVERS.pop()
        if cand.alive():
            conn = cand
            break
        cand.kill()
    if conn is None:
        conn = _ServerConn()
    try:
        info = conn.request("construct", {
            "board_path": str(board_path),
            "project_path": project_path or "",
            "seed": int(seed),
            "shove_iter_limit": int(shove_iter_limit),
            "followbranch_iter_limit": int(followbranch_iter_limit),
        })
    except EngineServerCrashed:
        raise
    except BaseException:
        # Construction failed but the server survived (e.g. bad board
        # path): park it for reuse before propagating.
        if len(_IDLE_SERVERS) < _MAX_IDLE:
            _IDLE_SERVERS.append(conn)
        else:
            conn.kill()
        raise
    return RouterProxy(conn), info["board_path"]
