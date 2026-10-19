"""CPU A/B contracts, preparation/resume, inference policy and actual five-route scoring."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from baselines.B5_best.common import read, write, sha
from baselines.B5_best_context.adapters import HistoryRecord, history_from_sample, _visible_digest
from baselines.B5_best_context.body_features import BODY_PROTOCOL, BEAT16_ENCODING, relation_features
from baselines.B5_best_context.body_history import A_POLICY, B_POLICY, history_policy, process_history
from baselines.B5_best_context.body_prepare import load_cache
from baselines.B5_best_context.body_ab_prepare import prepare, check_pair
from baselines.B5_best_context.body_ab_study import check_training_pair, cases, validate_features
from baselines.B5_best_context.body_alpha_core import SharedCorrection, FixedAlphaAdapter
from baselines.B5_best_context.body_study import run_study
from ecg12gen.record_context import window_context_record
from .test_body_encoding import mock_signal
from .test_context import StudyTests


def fake_filter_module():
    m = ModuleType("ecg12gen.body_scale_detrend")
    m.FILTER_SPEC = B_POLICY["filter"]
    def transform(raw, mask):
        mask = np.asarray(mask, bool).copy()
        raw = np.asarray(raw, np.float32).copy()
        # Synthetic transform deliberately changes shape; not a test of SciPy frequency response.
        for lead in range(6):
            raw[lead, mask[lead]] -= np.linspace(-30, 30, raw.shape[1], dtype=np.float32)[mask[lead]]
        raw[~mask] = np.nan
        return raw, mask, {}
    m.detrend_body_scale_record = transform
    return m


def make_dataset_source(root):
    source, overlay = root / "source", root / "overlay"
    write(overlay / "body_scale_ablation_manifest.json", {"synthetic": True})
    config = root / "data.yaml"
    config.write_text("synthetic data config", encoding="utf-8")
    samples, records = {}, []
    for split in ("train", "validation"):
        t = np.arange(10000)
        raw = np.vstack([(100 + 20*c) * np.sin(t / 50) + np.linspace(-20, 20, len(t)) for c in range(6)]).astype(np.float32)
        context = window_context_record(raw, "body_scale_d6", np.ones_like(raw, bool))
        context_sample = SimpleNamespace(context=context, context_source_type="body_scale_d6",
            context_record_id="history_" + split, input_quality_mask=np.ones(6, bool))
        history = history_from_sample(context_sample)
        filtered = process_history(history, B_POLICY)
        b_context = window_context_record(filtered.raw_uV, "body_scale_d6", filtered.valid_mask)
        anchor = (150 * np.sin(np.arange(60000) / 50)).astype(np.float32)
        target = np.vstack([anchor, anchor*.8, *[anchor*.5 for _ in range(10)]]).astype(np.float32)
        q = np.ones(12, bool)
        arrays = dict(anchor=anchor, base_ii=anchor*.7, target_ii=target[1], quality=q,
                      **dict(zip(("beat", "stats", "available"), relation_features(history))))
        file = split + "/record.npz"
        (source / split).mkdir(parents=True)
        np.savez_compressed(source / file, **arrays)
        records.append(dict(split=split, pair_id=split, subject_id=split, target_record_id=split,
            history_record_id=history.record_id, history_visible_sha256=_visible_digest(history),
            history_available=bool(arrays["available"]), I_II_eligible=True, file=file, sha256=sha(source/file)))
        samples[split] = (context, b_context, target, q)
    write(source / "cache.json", dict(protocol=BODY_PROTOCOL, completed=True, records=records,
                                      frozen={"base_checkpoint_sha256": "same_P"}))
    class Dataset:
        def __init__(self, cfg, task, split, body_scale_variant="A_raw_window"):
            self.split, self.variant = split, body_scale_variant
            self._indices = list(range(12))
            self._rows = [dict(pair_id=split, subject_id=split, target_record_id=split,
                input_type="body_scale_d6", start_sample_500hz=str(i*5000)) for i in range(12)]
        def __getitem__(self, i):
            a, b, target, q = samples[self.split]
            return SimpleNamespace(context=b if self.variant == B_POLICY["variant"] else a,
                context_source_type="body_scale_d6", context_record_id="history_"+self.split,
                input_quality_mask=np.ones(6, bool), anchor_i_ecg=target[:1, i*5000:(i+1)*5000],
                Y_12lead=target[:, i*5000:(i+1)*5000], target_quality_mask=q)
        def close(self):
            pass
    return source, overlay, config, Dataset


class BodyABTests(unittest.TestCase):
    def test_policy_defaults_legacy_and_B_transforms_only_visible_history_without_mutation(self):
        self.assertEqual(history_policy({}), A_POLICY)
        with self.assertRaises(ValueError):
            history_policy({"history_preprocessing": {"variant": "wrong"}})
        raw = np.arange(120, dtype=np.float32).reshape(6, 20)
        mask = np.ones(raw.shape, bool); mask[:, 8:10] = False
        h = HistoryRecord("body_scale_d6", "visible", raw, mask)
        self.assertIs(process_history(h, A_POLICY), h)
        original = raw.copy()
        with patch.dict("sys.modules", {"ecg12gen.body_scale_detrend": fake_filter_module()}):
            output = process_history(h, B_POLICY)
        np.testing.assert_array_equal(h.raw_uV, original)
        np.testing.assert_array_equal(output.valid_mask, mask)
        self.assertEqual(output.raw_uV.dtype, np.float32)
        self.assertTrue(np.isfinite(output.raw_uV).all())
        self.assertFalse(np.array_equal(output.raw_uV, raw))
        self.assertIsNone(process_history(None, B_POLICY))
        with self.assertRaises(ValueError):
            process_history(HistoryRecord("ecg_machine_d6", "wrong_device", raw, mask), B_POLICY)

    def test_training_pair_requires_matching_initialization_budget_loss_and_history_protocol(self):
        base = dict(protocol=BODY_PROTOCOL, group="context", seed=42, epochs=30, initial_weights_sha256="same",
                    feature_encoding=BEAT16_ENCODING, loss="same", base_checkpoint_sha256="P")
        a = {**base, "history_preprocessing": A_POLICY, "cache_sha256": "A"}
        b = {**base, "history_preprocessing": B_POLICY, "cache_sha256": "B"}
        check_training_pair(a, b, "A", "B")
        for key, value in (("seed", 43), ("epochs", 10), ("initial_weights_sha256", "other"),
                           ("history_preprocessing", A_POLICY), ("base_checkpoint_sha256", "old")):
            with self.assertRaises(ValueError):
                check_training_pair(a, {**b, key: value}, "A", "B")
        with self.assertRaises(ValueError):
            check_training_pair(a, b, "A", "B", preflight=True)
        check_training_pair({**a, "ready": True, "optimizer_updates": 0},
                            {**b, "ready": True, "optimizer_updates": 0}, "A", "B", preflight=True)

    def test_prepare_copies_P_arrays_exactly_checks_live_B_and_resumes_without_source_write(self):
        mods = {**mock_signal(), "ecg12gen.body_scale_detrend": fake_filter_module()}
        with tempfile.TemporaryDirectory() as temp, patch.dict("sys.modules", mods):
            root = Path(temp)
            source, overlay, cfg, dataset = make_dataset_source(root)
            source_hash = sha(source / "cache.json")
            args = SimpleNamespace(cache=source, output=root/"prepared", overlay=overlay, config=cfg, resume=False)
            with patch("baselines.B5.config.load_config", return_value={}), \
                    patch("baselines.B5.data.common_config", return_value=SimpleNamespace(raw={"paths": {}})), \
                    patch("ecg12gen.dataset.JointAnchorDataset", dataset):
                prepare(args)
                a, b, changes = check_pair(root/"prepared/A_cache", root/"prepared/B_cache")
                self.assertEqual(len(changes), 2)
                self.assertEqual(sha(source / "cache.json"), source_hash)
                self.assertEqual(a["frozen"], b["frozen"])
                before = {str(p.relative_to(args.output)): sha(p) for p in args.output.rglob("*.npz")}
                args.resume = True
                prepare(args)
                self.assertEqual(before, {str(p.relative_to(args.output)): sha(p) for p in args.output.rglob("*.npz")})
                self.assertTrue(read(args.output / "prepared.json")["B_overlay_runtime_features_exactly_matched"])
                # Changed data/config must fail resume before training.
                cfg.write_text("modified", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Resume changed"):
                    prepare(args)

    def test_prepare_rejects_B_overlay_that_does_not_match_runtime_filter(self):
        mods = {**mock_signal(), "ecg12gen.body_scale_detrend": fake_filter_module()}
        with tempfile.TemporaryDirectory() as temp, patch.dict("sys.modules", mods):
            root = Path(temp)
            source, overlay, cfg, dataset = make_dataset_source(root)
            class WrongB(dataset):
                def __getitem__(self, i):
                    s = super().__getitem__(i)
                    if self.variant == B_POLICY["variant"]:
                        s.context.raw_uV[0, 0, 0] += 1
                    return s
            args = SimpleNamespace(cache=source, output=root/"prepared", overlay=overlay, config=cfg, resume=False)
            with patch("baselines.B5.config.load_config", return_value={}), \
                    patch("baselines.B5.data.common_config", return_value=SimpleNamespace(raw={"paths": {}})), \
                    patch("ecg12gen.dataset.JointAnchorDataset", WrongB):
                with self.assertRaisesRegex(ValueError, "reproduce uploaded overlay"):
                    prepare(args)
            self.assertFalse(read(root / "prepared/B_cache/cache.json")["completed"])

    def test_five_actual_score_routes_preserve_task1_machine_and_frozen_arrays(self):
        class Engine:
            def __init__(self, scale): self.scale = scale
            def __call__(self, p, i, h):
                p[1] += self.scale * i[0]
                return p, {}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            StudyTests().baseline(root / "base")
            before = sha(root / "base/task2/prediction_uV.npy")
            adapters = {name: FixedAlphaAdapter(SharedCorrection(Engine(scale)), 1.)
                        for name, scale in (("old", .01), ("A", .02), ("B", .03))}
            rows = run_study(root/"base", root/"study", lambda r: None, None, None,
                             case_specs=cases(), adapter_map=adapters, additive_routes=[], notes=["synthetic A/B"])
            self.assertEqual([r["group"] for r in rows], ["baseline", "d6_only", "old_O16_both", "A_both", "B_both"])
            for row in rows:
                self.assertEqual(row["task1_r"], rows[0]["task1_r"])
            for row in rows[2:]:
                self.assertEqual(row["machine_r"], rows[1]["machine_r"])
            self.assertEqual(sha(root / "base/task2/prediction_uV.npy"), before)
            self.assertEqual(adapters["B"].shared.forward_calls, 1)


if __name__ == "__main__":
    unittest.main()
