from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CachedEval:
    adapter_hash: str
    eval_key: str
    score: float
    epoch: int
    metadata: dict[str, Any]


class EvaluationCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._latest: dict[tuple[str, str], CachedEval] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    cached = CachedEval(
                        adapter_hash=str(record["adapter_hash"]),
                        eval_key=str(record["eval_key"]),
                        score=float(record["score"]),
                        epoch=int(record.get("epoch", 0)),
                        metadata=dict(record.get("metadata") or {}),
                    )
                except Exception:
                    continue
                key = (cached.adapter_hash, cached.eval_key)
                previous = self._latest.get(key)
                if previous is None or cached.epoch >= previous.epoch:
                    self._latest[key] = cached

    def get(self, adapter_hash: str, eval_key: str) -> CachedEval | None:
        return self._latest.get((adapter_hash, eval_key))

    def put(
        self,
        adapter_hash: str,
        eval_key: str,
        score: float,
        epoch: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        cached = CachedEval(
            adapter_hash=adapter_hash,
            eval_key=eval_key,
            score=float(score),
            epoch=int(epoch),
            metadata=dict(metadata or {}),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "adapter_hash": cached.adapter_hash,
                        "eval_key": cached.eval_key,
                        "score": cached.score,
                        "epoch": cached.epoch,
                        "metadata": cached.metadata,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        self._latest[(adapter_hash, eval_key)] = cached
