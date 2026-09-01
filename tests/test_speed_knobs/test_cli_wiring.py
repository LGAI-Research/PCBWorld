"""Whether the --bf16/--compile-* training CLI flags wire through to
configure_speed (lightweight, no GPU required)."""

from __future__ import annotations

import torch

from methods.rl_agent.models.v1.net import KiCadRLModel


class TestTrainCliWiring:
    def test_parser_accepts_speed_flags(self):
        from methods.rl_agent.training.train_ppo import build_arg_parser
        args = build_arg_parser().parse_args([
            "--board", "dummy.kicad_pcb",
            "--bf16", "--compile-regions", "stack,decode,heads",
            "--compile-mode", "default",
        ])
        assert args.bf16 is True
        assert args.compile_regions == "stack,decode,heads"
        assert args.compile_mode == "default"

    def test_flags_default_off(self):
        from methods.rl_agent.training.train_ppo import build_arg_parser
        args = build_arg_parser().parse_args(["--board", "dummy.kicad_pcb"])
        assert args.bf16 is False and args.compile_regions == ""
        assert args.attn == "sdpa"

    def test_attn_flag(self):
        from methods.rl_agent.training.train_ppo import build_arg_parser
        args = build_arg_parser().parse_args(
            ["--board", "dummy.kicad_pcb", "--attn", "flex"])
        assert args.attn == "flex"

    def test_attn_knob_glue(self):
        import torch._dynamo
        torch.manual_seed(0)
        # n_heads=2: d_head 16 is the flex kernel minimum (asserted by the knob).
        m = KiCadRLModel(d_model=32, n_heads=2, n_layers=2, d_ff=64,
                         max_seq_len=2000, n_freq=4, use_critic=True)
        assert m.attn_impl == "sdpa"
        m.configure_speed(attn="flex")
        assert m.attn_impl == "flex"
        # flex is compiled static per (B, L_pad): the default per-frame limit
        # (8) would be exceeded and silently fall back to eager flex.
        assert torch._dynamo.config.recompile_limit >= 128

    def test_configure_speed_glue(self):
        # Verifies the same conversion (comma split -> tuple) as _build_policy's glue.
        torch.manual_seed(0)
        m = KiCadRLModel(d_model=32, n_heads=4, n_layers=2, d_ff=64,
                         max_seq_len=2000, n_freq=4, use_critic=True)
        regions = tuple(r for r in "stack,decode".split(",") if r)
        m.configure_speed(bf16=True, compile_regions=regions)
        assert m.bf16_compute is True
        assert m._stack_fn is not m._stack_impl      # compiled wrapper installed
        assert m._decode_fn is not m._decode_impl

    def test_efficient_alias_expands_to_adopted_combo(self):
        # --compile-regions efficient == stack,decode,heads (encode excluded).
        torch.manual_seed(0)
        m = KiCadRLModel(d_model=32, n_heads=4, n_layers=2, d_ff=64,
                         max_seq_len=2000, n_freq=4, use_critic=True)
        m.configure_speed(compile_regions=("efficient",))
        assert m._stack_fn is not m._stack_impl
        assert m._decode_fn is not m._decode_impl
        # heads: whether the two compile-target helpers were replaced by wrappers
        assert type(m._combined_ptr_logits).__name__ != "method"
        # encode must not be included in the expansion (vocab encode_* stays the original bound method)
        import inspect
        assert inspect.ismethod(m.tokenizer.vocab.encode_track)
