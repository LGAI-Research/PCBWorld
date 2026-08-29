# scripts/

Thin, general-purpose entrypoints only (`train.py` / `eval.py` / `profile.py`).

- [train.py](train.py) — training front door. `ppo` · `grpo` subcommands
  delegate argv to `train_{ppo,grpo}.py` under
  [methods/rl_agent/training/](../methods/rl_agent/training/).
  Needs the built C++ router (`PYTHONPATH` — see README.md).
- [eval.py](eval.py) — evaluation front door (single command, no subcommands).
  Delegates to the 3-stage pipeline [eval/pipeline.py](../eval/pipeline.py)
  (returned int → exit code). Needs the built C++ router.
- [profile.py](profile.py) — training-loop speed profiler front door.
  Delegates to [tools/diagnostics/speed_profiler/](../tools/diagnostics/speed_profiler/).

Reproduction recipes live in `experiments/`; manual tools in `tools/`.
