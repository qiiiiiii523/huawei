import tempfile
import unittest
from pathlib import Path
from baselines.B5_context_public.audit import partition, make_pairs, audit, inspect_header


class AuditTests(unittest.TestCase):
    def test_pairing_is_target_uniform_and_patient_disjoint(self):
        records = [dict(subject_id=s, record_id=f"{s}/{i}", seconds=115.)
                   for s, n in (("patient1", 3), ("patient2", 2), ("patient3", 1)) for i in range(n)]
        split = partition([r["subject_id"] for r in records])
        pairs = make_pairs(records, split)
        self.assertEqual(len(pairs), 5)
        self.assertEqual(len({p["target_record_id"] for p in pairs}), 5)
        for p in pairs:
            self.assertNotEqual(p["target_record_id"], p["history_record_id"])
            self.assertEqual(p["target_record_id"].split('/')[0], p["history_record_id"].split('/')[0])
            self.assertEqual(p["split"], split[p["subject_id"]])

    def test_audit_never_claims_qc_or_training_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            for s in range(4):
                folder = root / f"patient{s}"; folder.mkdir(parents=True)
                for i in range(2):
                    name = f"r{i}"
                    (folder / f"{name}.dat").write_bytes(bytes([s, i]) * 120000)
                    leads = "I II III aVR aVL aVF V1 V2 V3 V4 V5 V6".split()
                    header = f"{name} 12 1000 10000\n" + ''.join(
                        f"{name}.dat 16 2000/mV 16 0 0 0 0 {lead}\n" for lead in leads)
                    (folder / f"{name}.hea").write_text(header)
            report = audit(root, Path(directory) / "out")
            self.assertEqual(report["eligible_headers"], 8)
            self.assertFalse(report["training_ready"])
            self.assertFalse(report["waveform_QC_completed"])
            with self.assertRaises(FileExistsError):
                audit(root, Path(directory) / "out")

    def test_truncated_source_and_output_inside_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "patient1"; folder.mkdir()
            path = folder / "r.hea"
            path.write_text("r 12 1000 10000\n" + ''.join(
                f"r.dat 16 2000 16 0 0 0 0 {lead}\n"
                for lead in "I II III aVR aVL aVF V1 V2 V3 V4 V5 V6".split()))
            (folder / "r.dat").write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "byte length"):
                inspect_header(path, root, 10)
            with self.assertRaisesRegex(ValueError, "outside"):
                audit(root, root / "out")

    def test_empty_pairs_still_produces_csv_header(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"; root.mkdir()
            folder = root / "patient1"; folder.mkdir()
            (folder / "r.hea").write_text("invalid\n")
            out = Path(directory) / "out"
            report = audit(root, out)
            self.assertEqual(report["counts"]["train"]["targets"], 0)
            self.assertTrue((out / "candidate_pairs.csv").read_text().startswith("subject_id,"))


if __name__ == "__main__":
    unittest.main()
