"""Shell CLI forwarding checks with a fake Python executable: never trains."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


class SeedScriptTests(unittest.TestCase):
    def test_forwards_group_seed_and_rejects_invalid_seed(self):
        repo = Path(__file__).resolve().parents[3]
        bash = shutil.which('bash')
        if os.name == 'nt':
            candidate = Path(r'C:\Program Files\Git\bin\bash.exe')
            bash = str(candidate) if candidate.is_file() else None
        if bash is None:
            self.skipTest('Bash unavailable')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'shared').mkdir()
            (root / 'shared/preflight.json').write_text('{}', encoding='utf-8')
            calls = root / 'calls.txt'
            environment = dict(os.environ, CALLS=calls.as_posix())
            fake = 'python() { printf "%s\\n" "$*" >> "$CALLS"; }; export -f python; '
            command = fake + 'bash scripts/run_b5_slow_loss.sh --gpu 1 --group F --seed 43 --root ' + shlex.quote(root.as_posix())
            result = subprocess.run([bash, '-c', command], cwd=repo, env=environment,
                                    capture_output=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = calls.read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(lines), 4)  # One train + three evaluate commands.
            self.assertIn('--group F --seed 43', lines[0])
            self.assertIn('baselines.B5_exp.train_loss', lines[0])
            self.assertTrue(all('baselines.B5_exp.evaluate' in line for line in lines[1:]))
            before = calls.read_bytes()
            for invalid in ('-1', '4294967296', 'abc'):
                command = fake + 'bash scripts/run_b5_slow_loss.sh --group C --seed ' + invalid
                result = subprocess.run([bash, '-c', command], cwd=repo, env=environment,
                                        capture_output=True, encoding='utf-8')
                self.assertEqual(result.returncode, 2)
            self.assertEqual(calls.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
