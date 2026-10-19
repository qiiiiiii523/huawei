"""Check frozen CSV metadata across Git CRLF/LF checkouts without changing data."""
from __future__ import annotations

import hashlib
from pathlib import Path


def metadata_sha256_matches(path, expected):
    """Accept exact bytes or a newline-only equivalent of the recorded SHA256.

    No CSV parsing/reordering, whitespace stripping, BOM removal, path replacement
    or value conversion is allowed. A real field/row change still fails closed.
    Array hashes must continue to use their exact-byte hash checker.
    """
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() == expected:
        return True
    lf = raw.replace(b"\r\n", b"\n")
    return any(hashlib.sha256(candidate).hexdigest() == expected
               for candidate in (lf, lf.replace(b"\n", b"\r\n")))
