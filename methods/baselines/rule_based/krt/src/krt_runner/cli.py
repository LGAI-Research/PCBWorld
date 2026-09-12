"""``krt-route`` console script entry point.

Forwards all argv to KRT ``route.py`` and exits with the child's
returncode. No flag parsing — pass anything KRT accepts.
"""

from __future__ import annotations

import sys

from . import run_route


def main() -> int:
    proc = run_route(sys.argv[1:])
    return int(proc.returncode)


if __name__ == "__main__":
    sys.exit(main())
