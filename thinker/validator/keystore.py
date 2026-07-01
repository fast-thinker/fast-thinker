from __future__ import annotations

import os
import stat
from pathlib import Path

from thinker.submission.crypto import generate_keypair, public_key_from_private


def load_or_create_keypair(path: Path | str) -> tuple[bytes, bytes]:
    p = Path(path)
    if p.exists():
        raw = bytes.fromhex(p.read_text(encoding="utf-8").strip())
        if len(raw) != 32:
            raise ValueError(f"keypair file {p} does not contain a 32-byte X25519 key")
        return raw, public_key_from_private(raw)

    priv_bytes, pub_bytes = generate_keypair()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(priv_bytes.hex(), encoding="utf-8")
    try:
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return priv_bytes, pub_bytes
