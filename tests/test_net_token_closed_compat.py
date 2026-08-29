"""NET-token `closed` channel: 3f/4f checkpoint compatibility (pure torch).

Old checkpoints encode nets as 3f (track_width + clearance + via_diameter);
current ones add the per-episode ``closed`` flag (4f). The eval loader derives
``legacy_net_encoding`` from the saved ``net_proj.weight`` shape — mirroring
the ``legacy_pad_layer_encoding`` mechanism — so both generations stay
loadable without a config knob.
"""

from methods.rl_agent.models.loader import _policy_args_for_checkpoint

_ARGS = {
    "d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64,
    "max_seq_len": 512, "n_freq": 4, "coord_encoding": "fourier",
    "mlp_hidden": 16, "use_critic": True,
}


def _model(**overrides):
    from methods.rl_agent.models.v1.net import KiCadRLModel

    kw = {k: v for k, v in _ARGS.items() if k != "use_critic"}
    kw["use_critic"] = _ARGS["use_critic"]
    kw.update(overrides)
    return KiCadRLModel(**kw)


class TestCheckpointShapeDetection:
    def test_legacy_3f_checkpoint_detected_and_loads(self):
        legacy_sd = _model(legacy_net_encoding=True).state_dict()
        compat = _policy_args_for_checkpoint(dict(_ARGS), legacy_sd)
        assert compat["legacy_net_encoding"] is True

        from configs.loader.schema import RLPolicyConfig

        rebuilt = RLPolicyConfig.from_checkpoint(compat).build()
        rebuilt.load_state_dict(legacy_sd, strict=True)

    def test_current_4f_checkpoint_detected_and_loads(self):
        current_sd = _model(legacy_net_encoding=False).state_dict()
        compat = _policy_args_for_checkpoint(dict(_ARGS), current_sd)
        assert compat["legacy_net_encoding"] is False

        from configs.loader.schema import RLPolicyConfig

        rebuilt = RLPolicyConfig.from_checkpoint(compat).build()
        rebuilt.load_state_dict(current_sd, strict=True)

    def test_net_proj_widths(self):
        legacy = _model(legacy_net_encoding=True)
        current = _model(legacy_net_encoding=False)
        f = legacy.tokenizer.vocab.fenc_dim
        assert legacy.tokenizer.vocab.net_proj.in_features == 3 * f
        assert current.tokenizer.vocab.net_proj.in_features == 4 * f
