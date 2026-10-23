"""One machine-route sampling contract for validation, selection and inference.

Training continues to use the original iid Gaussian FM path; this module is
generation only. Existing B5/Final entry points are deliberately unchanged.
"""
SAMPLING = {"solver": "heun", "steps": 16, "samples": 16, "seed": 42,
            "noise_mode": "antithetic", "noise_scale": 1.0,
            "noise_protocol": "B5-keyed-adjacent-antithetic-v1"}


def member_spec(index):
    """Member order z0,-z0,z1,-z1,...,z7,-z7; base RNG remains B5-v1."""
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 16:
        raise ValueError("Need member index 0..15")
    return index // 2, 1 if index % 2 == 0 else -1


def member_noise(keys, length, index, device, dtype):
    from baselines.B5.flow import keyed_noise
    pair_index, sign = member_spec(index)
    # Eight independent N(0,I) draws and their exact negatives, no variance scaling.
    return sign * keyed_noise(keys, length, SAMPLING["seed"], pair_index, device, dtype)


def validate_settings(settings):
    if settings != SAMPLING:
        raise ValueError("ContextPublic machine generation requires Heun16/antithetic K16/scale1/seed42")


def sample(model, inputs, keys, *, context_enabled=False):
    """Normalized d12; supports the stable B5 and reused V6 via one noise path."""
    import torch
    from baselines.B5.flow import integrate
    fields = ("anchor", "numeric", "sex", "field_mask", "age_topcoded")
    anchor = inputs["anchor"]
    if model.training or len(keys) != len(anchor) or any(not isinstance(k, str) or not k for k in keys):
        raise ValueError("Sampling needs eval mode, stable nonempty keys and matching batch")
    with torch.no_grad():
        kwargs = {}
        if context_enabled:
            kwargs = {k: inputs[k] for k in ("context", "context_time_mask", "context_window_mask", "context_window_weights")}
        condition = model.encode_conditions(*(inputs[k] for k in fields), **kwargs)
        total = anchor.new_zeros(len(anchor), 11, anchor.shape[-1])
        for index in range(16):
            noise = member_noise(keys, anchor.shape[-1], index, anchor.device, anchor.dtype)
            total += integrate(model, condition, noise, steps=16, solver="heun")
        return torch.cat((condition.anchor_prediction, total / 16), dim=1)


def validate_machine_references(reports):
    """Reject old independent-noise scores/caches before paired selection.

    Called by the future full-record training/evaluation orchestrator; checking
    metadata cannot itself prove that generation used the claimed noise rule.
    """
    names = ("public_initial", "stable_machine", "control_initial", "context_public_initial")
    if set(reports) != set(names):
        raise ValueError("Need all four freshly evaluated machine references")
    reference = reports["stable_machine"]
    pinned = ("population_sha256", "record_keys_sha256", "scales_sha256", "slow_assets_sha256", "terminal_rules_sha256")
    for name in names:
        report = reports[name]
        validate_settings(report.get("sampling"))
        if (report.get("completed") is not True or report.get("route") != "ecg_machine_d6"
                or report.get("full_record_seconds") != 120 or report.get("slow_beta") != .75):
            raise ValueError("Incomplete or incompatible reference: " + name)
        for field in ("checkpoint_sha256", *pinned):
            value = report.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Missing SHA256 identity: " + name + ":" + field)
        if any(report[k] != reference[k] for k in pinned):
            raise ValueError("Reference populations/keys/scales/slow/context differ: " + name)
    return True
