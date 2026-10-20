"""Explicitly transfer the frozen body to a new fast predictor, without forged lineage."""
from baselines.B5_best.common import sha
from baselines.B5_best_context.body_model import BodyAdapter, BodyIIAdapter
from baselines.B5_best_context.adapters import ContextRouter, load_settings
from baselines.B5_final.profile import BASE_SHA, BODY_SHA


class FrozenBodyTransfer:
    def __init__(self, body_checkpoint, device, *, group, predictor_sha):
        if sha(body_checkpoint) != BODY_SHA:
            raise ValueError("Only Final's original body weights may transfer")
        self.adapter = BodyAdapter(body_checkpoint, BASE_SHA, device, expected_group="context")
        self.adapter.model.requires_grad_(False).eval()
        self.group, self.predictor_sha = group, predictor_sha

    @classmethod
    def from_payload(cls, payload, device, *, group, predictor_sha):
        # The package loader verifies Final's complete payload lineage first.
        instance = cls.__new__(cls)
        adapter = BodyAdapter.__new__(BodyAdapter)
        adapter.model = BodyIIAdapter().to(device)
        adapter.model.load_state_dict(payload["model"], strict=True)
        adapter.model.requires_grad_(False).eval()
        adapter.payload, adapter.device, adapter.mode, adapter.cache = payload, device, "correct", {}
        instance.adapter, instance.group, instance.predictor_sha = adapter, group, predictor_sha
        return instance

    def __call__(self, prediction, observed_i, history):
        result, status = self.adapter(prediction, observed_i, history)
        return result, {**status, "transfer_mode": "original_Final" if self.group == "F0" else "frozen_body_transfer",
                        "body_source_base_sha256": BASE_SHA, "body_sha256": BODY_SHA,
                        "new_fast_group": self.group, "new_fast_predictor_sha256": self.predictor_sha}


def routers(body_checkpoint, device, *, group, predictor_sha, body_payload=None, settings=None):
    settings = load_settings() if settings is None else settings
    transferred = (FrozenBodyTransfer(body_checkpoint, device, group=group, predictor_sha=predictor_sha)
                   if body_payload is None else FrozenBodyTransfer.from_payload(
                       body_payload, device, group=group, predictor_sha=predictor_sha))
    return {"baseline": ContextRouter(settings, machine_mode="none"),
            "d6_only": ContextRouter(settings, machine_mode="fir25"),
            "both": ContextRouter(settings, machine_mode="fir25", body_adapter=transferred)}
