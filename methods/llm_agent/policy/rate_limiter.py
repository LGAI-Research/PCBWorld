"""Sliding-window rate limiter for API providers (Anthropic, OpenAI, ...).

Thread-safe; expects callers to invoke :meth:`acquire` before each request
and :meth:`record` after the response with the actual usage from the
provider's response (so the placeholder is replaced by truth).

The ``safety`` factor shrinks each cap so we never quite touch the published
limit, leaving headroom for token-count misestimation and provider clock skew.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional


class SlidingWindowLimiter:
    """Per-minute sliding-window cap on requests + input/output tokens.

    Pass ``None`` for any cap to disable that dimension.
    """

    def __init__(
        self,
        rpm: Optional[int] = None,
        itpm: Optional[int] = None,
        otpm: Optional[int] = None,
        window_s: float = 60.0,
        safety: float = 0.95,
    ) -> None:
        big = float(10 ** 12)
        self.rpm = (rpm if rpm is not None else big) * safety
        self.itpm = (itpm if itpm is not None else big) * safety
        self.otpm = (otpm if otpm is not None else big) * safety
        self.window_s = window_s
        self._lock = threading.Lock()
        self._req: deque = deque()       # ts
        self._in: deque = deque()        # (ts, tokens)
        self._out: deque = deque()       # (ts, tokens)

    # -------- internal --------

    def _gc(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._req and self._req[0] < cutoff:
            self._req.popleft()
        while self._in and self._in[0][0] < cutoff:
            self._in.popleft()
        while self._out and self._out[0][0] < cutoff:
            self._out.popleft()

    # -------- public --------

    def acquire(self, est_input: int, est_output: int) -> None:
        """Block until enough capacity for one request of the given size.

        On success, inserts a placeholder so concurrent acquirers don't
        double-spend the budget. Caller MUST follow with :meth:`record`
        once the actual usage is known.
        """
        while True:
            with self._lock:
                now = time.time()
                self._gc(now)
                cur_req = len(self._req)
                cur_in = sum(t for _, t in self._in)
                cur_out = sum(t for _, t in self._out)
                wait = 0.0
                if cur_req + 1 > self.rpm and self._req:
                    wait = max(wait, self._req[0] + self.window_s - now)
                if cur_in + est_input > self.itpm and self._in:
                    wait = max(wait, self._in[0][0] + self.window_s - now)
                if cur_out + est_output > self.otpm and self._out:
                    wait = max(wait, self._out[0][0] + self.window_s - now)
                if wait <= 0:
                    self._req.append(now)
                    self._in.append((now, est_input))
                    self._out.append((now, est_output))
                    return
            time.sleep(min(wait + 0.1, 5.0))

    def record(self, actual_input: int, actual_output: int) -> None:
        """Replace the most recent placeholder with the actual usage."""
        with self._lock:
            if self._in:
                ts, _ = self._in.pop()
                self._in.append((ts, actual_input))
            if self._out:
                ts, _ = self._out.pop()
                self._out.append((ts, actual_output))

    def stats(self) -> dict:
        with self._lock:
            now = time.time()
            self._gc(now)
            return {
                "req_per_min": len(self._req),
                "in_per_min": sum(t for _, t in self._in),
                "out_per_min": sum(t for _, t in self._out),
            }
