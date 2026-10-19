"""Read-only checkpoint selection, encoding guards and immutable continuation staging."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from baselines.B5_best.common import read, write, sha
from baselines.B5_best_context.body_alpha_study import check_alpha_checkpoint
from baselines.B5_best_context.body_features import BODY_PROTOCOL, SHAPE_ENCODING, LEGACY_ENCODING, BEAT16_ENCODING
from baselines.B5_best_context.body_gate_features import GATE_PROTOCOL
from baselines.B5_best_context.body_verify_prepare import completed_run, find_e24, extend


def fixture(root, cache_sha, encoding, *, epochs, best_epoch, seed=42, score=.786604):
    root.mkdir(parents=True)
    cfg = {"protocol": BODY_PROTOCOL, "group": "context", "feature_encoding": encoding,
        "cache_sha256": cache_sha, "seed": seed, "epochs": epochs, "batch_size": 16,
        "learning_rate": .0002, "initial_weights_sha256": "same"}
    (root / "best.pt").write_bytes(b"synthetic saved best; not a torch execution")
    (root / "last.pt").write_bytes(b"synthetic saved model + optimizer + RNG")
    write(root / "config.json", cfg)
    write(root / "completed.json", {"completed": True, "epochs": epochs, "training_seed": seed,
        "best_epoch": best_epoch, "best_II_r": score, "best_checkpoint_sha256": sha(root / "best.pt")})
    write(root / "validation" / f"epoch_{best_epoch:03d}.json", {"epoch": best_epoch})
    return cfg


class VerificationTests(unittest.TestCase):
    def test_alpha_E_and_C_require_their_own_cache_protocol_and_epoch(self):
        e = {"protocol": BODY_PROTOCOL, "group": "context", "feature_encoding": SHAPE_ENCODING,
             "cache_sha256": "encoded", "epoch": 24}
        check_alpha_checkpoint(e, {"feature_encoding": SHAPE_ENCODING}, "encoded", "E", 24)
        c = {"protocol": GATE_PROTOCOL, "group": "context", "record_gate": "C", "cache_sha256": "old", "epoch": 26}
        check_alpha_checkpoint(c, {}, "old", "C", 26)
        for cache, digest, kind, epoch in (({}, "encoded", "E", 24),
            ({"feature_encoding": SHAPE_ENCODING}, "other", "E", 24),
            ({"feature_encoding": SHAPE_ENCODING}, "encoded", "C", 24),
            ({"feature_encoding": SHAPE_ENCODING}, "encoded", "E", 26)):
            with self.assertRaises(ValueError):
                check_alpha_checkpoint(e, cache, digest, kind, epoch)
        with self.assertRaises(ValueError):
            check_alpha_checkpoint({**c, "record_gate": "G"}, {}, "old", "C", 26)

    def test_E24_discovery_checks_report_and_actual_hash_and_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            e = root / "reported/E"
            fixture(e, "encoded", SHAPE_ENCODING, epochs=30, best_epoch=24)
            fixture(root / "wrong_seed/E", "encoded", SHAPE_ENCODING, epochs=30, best_epoch=24, seed=43)
            fixture(root / "wrong_budget/E", "encoded", SHAPE_ENCODING, epochs=10, best_epoch=10)
            self.assertEqual(find_e24(root, "encoded"), e.resolve())
            fixture(root / "second/E", "encoded", SHAPE_ENCODING, epochs=30, best_epoch=24)
            with self.assertRaisesRegex(ValueError, "uniquely"):
                find_e24(root, "encoded")
            (e / "best.pt").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                completed_run(e, "encoded", SHAPE_ENCODING, seed=42, epochs=30, best_epoch=24)

    def test_extension_copies_last_optimizer_blob_best_and_logs_without_source_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root / "cache"
            write(cache / "cache.json", {"synthetic": True})
            source = root / "source/O16"
            original = fixture(source, sha(cache / "cache.json"), BEAT16_ENCODING, epochs=10, best_epoch=10)
            before = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
            args = SimpleNamespace(source_run=source, output=root / "new/O16", cache=cache, epochs=30, resume=False)
            extend(args)
            self.assertEqual(read(args.output / "config.json"), {**original, "epochs": 30})
            self.assertEqual(sha(args.output / "last.pt"), sha(source / "last.pt"))
            self.assertEqual(sha(args.output / "best.pt"), sha(source / "best.pt"))
            self.assertFalse((args.output / "completed.json").exists())
            self.assertTrue(read(args.output / "extension.json")["optimizer_and_RNG_preserved"])
            self.assertTrue((args.output / "validation/epoch_010.json").is_file())
            self.assertEqual(before, {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()})
            # Once the staged run has trained, resume must preserve its NEW last state.
            (args.output / "last.pt").write_bytes(b"new training state")
            args.resume = True
            extend(args)
            self.assertEqual((args.output / "last.pt").read_bytes(), b"new training state")
            (source / "last.pt").write_bytes(b"changed original optimizer state")
            with self.assertRaises(ValueError):
                extend(args)

    def test_extension_rejects_equal_budget_and_source_output_alias(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root / "cache"
            write(cache / "cache.json", {})
            source = root / "source"
            fixture(source, sha(cache / "cache.json"), BEAT16_ENCODING, epochs=10, best_epoch=10)
            args = SimpleNamespace(source_run=source, output=root / "new", cache=cache, epochs=10, resume=False)
            with self.assertRaises(ValueError):
                extend(args)
            args.epochs, args.output = 30, source / "nested"
            with self.assertRaises(ValueError):
                extend(args)


if __name__ == "__main__":
    unittest.main()
