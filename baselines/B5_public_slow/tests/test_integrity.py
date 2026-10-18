"""Standard-library tests of missing/corrupt files and read-only PTB audit."""
from pathlib import Path
import tempfile
import unittest

from scripts.check_ptb_integrity import audit, digest, safe_path


def fixture(parent):
    root = parent / "data"
    patient = root / "patient001"
    patient.mkdir(parents=True)
    names = ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6", "vx", "vy", "vz")
    lines = ["s0001 15 1000 100"]
    for i, name in enumerate(names):
        filename = "s0001.dat" if i < 12 else "s0001.xyz"
        lines.append(filename + " 16 2000 16 0 0 0 0 " + name)
    (patient / "s0001.hea").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (patient / "s0001.dat").write_bytes(bytes(100 * 12 * 2))
    (patient / "s0001.xyz").write_bytes(bytes(100 * 3 * 2))
    (root / "RECORDS").write_text("patient001/s0001\n", encoding="utf-8")
    (root / "README").write_text("source metadata", encoding="utf-8")
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    (root / "SHA256SUMS.txt").write_text("\n".join(digest(p)[0] + " " + p.relative_to(root).as_posix() for p in paths) + "\n", encoding="utf-8")
    return root


class IntegrityTests(unittest.TestCase):
    def test_complete_data_is_read_only_and_short_record_is_not_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary); root = fixture(parent)
            before = {p.relative_to(root).as_posix(): digest(p)[0] for p in root.rglob("*") if p.is_file()}
            result = audit(root, parent / "report")
            self.assertTrue(result["ready_for_preprocessing"])
            self.assertEqual(result["duration_counts"], {"under_50s": 1})
            self.assertEqual(before, {p.relative_to(root).as_posix(): digest(p)[0] for p in root.rglob("*") if p.is_file()})

    def test_missing_header_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary); root = fixture(parent)
            (root / "patient001/s0001.hea").unlink()
            result = audit(root, parent / "report")
            self.assertFalse(result["ready_for_preprocessing"])
            self.assertEqual(result["missing_required_files"], ["patient001/s0001.hea"])

    def test_byte_corruption_and_truncation_are_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary); root = fixture(parent)
            path = root / "patient001/s0001.dat"
            path.write_bytes(b"x" + path.read_bytes()[1:])
            result = audit(root, parent / "corrupt")
            self.assertEqual(result["checksum_mismatches"], ["patient001/s0001.dat"])
            path.write_bytes(b"x")
            result = audit(root, parent / "truncated")
            self.assertEqual(result["invalid_records"], ["patient001/s0001"])

    def test_optional_root_files_do_not_block_waveform_check_unless_strict(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary); root = fixture(parent)
            (root / "README").unlink()
            normal = audit(root, parent / "normal")
            self.assertTrue(normal["ready_for_preprocessing"])
            self.assertEqual(normal["optional_missing_files"], ["README"])
            self.assertFalse(audit(root, parent / "strict", strict_root=True)["ready_for_preprocessing"])

    def test_missing_manifest_creates_actionable_failure_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary); root = fixture(parent)
            (root / "SHA256SUMS.txt").unlink()
            result = audit(root, parent / "report")
            self.assertFalse(result["ready_for_preprocessing"])
            self.assertIn("Missing required root manifest: SHA256SUMS.txt", result["errors"])

    def test_reports_cannot_modify_dataset_and_unsafe_manifest_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary); root = fixture(parent)
            with self.assertRaisesRegex(ValueError, "OUTSIDE"):
                audit(root, root / "report")
            with self.assertRaises(ValueError): safe_path(root, "../private")
            with (root / "SHA256SUMS.txt").open("a", encoding="utf-8") as handle:
                handle.write("0" * 64 + " ../outside\n")
            self.assertFalse(audit(root, parent / "report")["ready_for_preprocessing"])


if __name__ == "__main__":
    unittest.main()
