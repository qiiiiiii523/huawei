"""Local CPU smoke test for held-out body I-to-II relation learning."""
from __future__ import annotations

import numpy as np
import torch

from baselines.B5.context_body_lead_relation import BodyContextEncoderLeadRelation
from scripts.evaluate_b5_body_diagnostics import body_pairs
from scripts.train_b5_body_context import BodySubset


def main() -> None:
    dataset = BodySubset("validation", "A_raw_window")
    try:
        pair = list(body_pairs(dataset).values())[5]
        raw = pair.context.raw_uV[:2].astype(np.float32)
        mask = pair.context.time_mask[:2]
        baseline = np.asarray([np.median(raw[:, c][mask[:, c]]) for c in range(6)], dtype=np.float32)
        values = np.where(mask, (raw - baseline[None, :, None]) / 1000, 0)
        context = torch.from_numpy(values[None].copy())
        valid = torch.from_numpy(mask[None].copy())
        counts = valid.sum(dim=(2, 3)).float()
        weights = counts / counts.sum(dim=1, keepdim=True)
        available = torch.tensor([True])
        encoder = BodyContextEncoderLeadRelation()
        for module in (encoder.cnn, encoder.project, encoder.attention, encoder.fuse):
            module.requires_grad_(False)
        embedding = encoder(context, valid, weights, available)
        auxiliary = encoder.last_auxiliary_loss
        if (embedding.shape != (1, 128) or encoder.last_valid_records != 1
                or auxiliary is None or not torch.isfinite(auxiliary) or auxiliary <= 0):
            raise ValueError("Relation encoder shape/coverage/loss test failed")
        (embedding.square().mean() + 0.02 * auxiliary).backward()
        grads = {name: sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
                 for name, module in (("relation_encoder", encoder.relation_encoder),
                                      ("transfer_head", encoder.transfer_head))}
        gate_grad = float(encoder.relation_gate.grad.abs())
        if min(*grads.values(), gate_grad) <= 0:
            raise ValueError(f"Relation branch gradient test failed: {grads}, gate={gate_grad}")
        beat, present = encoder._features(context, valid)
        support, support_mask, query, query_mask = encoder._relation_beats(beat, present)
        relation = encoder._encode_relation(support, support_mask)
        changed = query.clone()
        changed[:, :, 1] = -changed[:, :, 1]
        with torch.no_grad():
            original_loss = encoder._held_out_loss(query, query_mask, relation, available)
            changed_loss = encoder._held_out_loss(changed, query_mask, relation, available)
        if torch.isclose(original_loss, changed_loss, rtol=1e-5, atol=1e-7):
            raise ValueError("Held-out query II did not change the auxiliary target")
        print({"smoke": "passed", "auxiliary_loss": float(auxiliary.detach()),
               "valid_records": encoder.last_valid_records, "gradients": grads,
               "gate_gradient": gate_grad, "beats": int(present.sum())})
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
