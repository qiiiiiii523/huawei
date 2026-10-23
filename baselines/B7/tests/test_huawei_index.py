"""Regression checks for the metadata bridge; no GPU or raw private data needed."""
import csv
from pathlib import Path
import tempfile
import unittest

from baselines.B7.huawei_index import build_index


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class HuaweiIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.metadata = self.root / "metadata"
        self.metadata.mkdir()
        self.data = self.root / "data"
        self.d12 = self.data / "心电图机d12"
        self.d12.mkdir(parents=True)
        splits, raw, qc, task1, task2 = [], [], [], [], []
        for split in ("train", "validation"):
            target = f"d12:{split}"
            (self.d12 / f"{split}.xml").write_text("<test/>", encoding="utf-8")
            splits.append({"subject_id": split, "split": split})
            raw.append({"record_id": target, "subject_id": split, "split": split,
                        "quality_status": "usable", "source_path": f"Data/old-name/{split}.xml"})
            qc.append({"record_id": target, "device_type": "ecg_machine_d12", "split": split,
                       "target_lead_mask": "|".join(["1"] * 12), "d12_direct_supervision_eligible": "true"})
            for group in ("watch_ecg", "ecg_machine_d6", "body_scale_d6"):
                row = {"pair_id": f"{group}:{split}", "subject_id": split, "split": split,
                       "target_record_id": target, "input_type": group, "input_record_id": "context-only",
                       "pair_status": "paired", "target_quality_status": "usable", "training_policy": "keep",
                       "target_path": f"Data\\old-name\\{split}.xml"}
                (task1 if group == "watch_ecg" else task2).append(row)
        for name, rows in (("subject_split", splits), ("raw_record_manifest", raw),
                           ("device_interpretation_qc", qc), ("pair_manifest_task1", task1),
                           ("pair_manifest_task2", task2)):
            write_csv(self.metadata / (name + ".csv"), rows)

    def build(self, source="same-record-target-I"):
        return build_index(self.metadata, self.data, source)

    def test_relocated_paths_repeat_targets_and_explicit_anchor(self):
        # Same basename in a different device must never select that device.
        d6 = self.data / "心电图机d6"
        d6.mkdir()
        (d6 / "train.xml").write_text("<wrong/>", encoding="utf-8")
        rows, audit = self.build()
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({r["target_record_id"] for r in rows}), 2)
        self.assertEqual(audit["counts"]["validation/body"], 1)
        self.assertTrue(all(Path(r["path"]).parent == self.d12 for r in rows))
        self.assertTrue(all(r["anchor_source"] == "same_record_target_I" and
                            not r["original_context_used"] for r in rows))

    def test_duplicate_XML_in_d12_is_rejected(self):
        sub = self.d12 / "copy"
        sub.mkdir()
        (sub / "train.xml").write_text("<test/>", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "found 2"):
            self.build()

    def test_split_mismatch_is_rejected(self):
        from baselines.B7.huawei_index import read_csv
        path = self.metadata / "pair_manifest_task1.csv"
        rows = read_csv(path)
        rows[0]["split"] = "validation"
        write_csv(path, rows)
        with self.assertRaisesRegex(ValueError, "authoritative split"):
            self.build()

    def test_nonbinary_QC_is_rejected(self):
        from baselines.B7.huawei_index import read_csv
        path = self.metadata / "device_interpretation_qc.csv"
        rows = read_csv(path)
        rows[0]["target_lead_mask"] = "0.5|" + "|".join(["1"] * 11)
        write_csv(path, rows)
        with self.assertRaisesRegex(ValueError, "fixed lead QC"):
            self.build()

    def test_unsynchronized_context_cannot_be_declared_as_anchor(self):
        with self.assertRaisesRegex(ValueError, "synchronous target-I"):
            self.build("watch-I")


if __name__ == "__main__":
    unittest.main()
