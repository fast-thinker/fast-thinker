from __future__ import annotations

import hashlib
import threading
from pathlib import Path


def instance_hash(track: str, prompt: str) -> str:
    normalized = prompt.strip()
    return hashlib.sha256(f"{track}:{normalized}".encode("utf-8")).hexdigest()


class DecontaminationStore:
    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as f:
                self._seen.update(line.strip() for line in f if line.strip())

    def __len__(self) -> int:
        return len(self._seen)

    def seen(self, digest: str) -> bool:
        return digest in self._seen

    def _record_unlocked(self, digest: str) -> None:
        self._seen.add(digest)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(digest + "\n")

    def record(self, digest: str) -> None:
        with self._lock:
            if digest not in self._seen:
                self._record_unlocked(digest)

    def check_and_record(self, track: str, prompt: str) -> bool:
        digest = instance_hash(track, prompt)
        with self._lock:
            if digest in self._seen:
                return False
            self._record_unlocked(digest)
            return True
