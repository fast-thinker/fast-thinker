from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChampionRecord:
    epoch: int
    miner_id: str
    score: float


@dataclass
class MinerRoundState:
    last_adapter_hash: str
    rounds_without_full_eval: int = 0


class RoundStateStore:
    """Tracks cross-round state needed by the staged evaluation:

    - the last `champion_history_rounds` round winners, kept by miner_id so a
      miner stays eligible for full evaluation even after resubmitting a new
      adapter.
    - per-miner streaks of rounds without a full-evaluation slot, keyed by
      miner_id but reset whenever the miner's adapter hash changes, so a
      miner can only be skipped for being stagnant on an *unchanged*
      submission.
    """

    def __init__(self, path: str | Path, *, champion_history_rounds: int = 5):
        self.path = Path(path)
        self._champion_history_rounds = max(0, champion_history_rounds)
        self._champions: list[ChampionRecord] = []
        self._miner_state: dict[str, MinerRoundState] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        for record in data.get("champions", []) or []:
            try:
                self._champions.append(
                    ChampionRecord(
                        epoch=int(record["epoch"]),
                        miner_id=str(record["miner_id"]),
                        score=float(record["score"]),
                    )
                )
            except Exception:
                continue
        for miner_id, record in (data.get("miner_state") or {}).items():
            try:
                self._miner_state[str(miner_id)] = MinerRoundState(
                    last_adapter_hash=str(record["last_adapter_hash"]),
                    rounds_without_full_eval=int(record.get("rounds_without_full_eval", 0)),
                )
            except Exception:
                continue

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "champions": [
                {"epoch": record.epoch, "miner_id": record.miner_id, "score": record.score}
                for record in self._champions
            ],
            "miner_state": {
                miner_id: {
                    "last_adapter_hash": state.last_adapter_hash,
                    "rounds_without_full_eval": state.rounds_without_full_eval,
                }
                for miner_id, state in self._miner_state.items()
            },
        }
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)

    def recent_champions(self) -> set[str]:
        return {record.miner_id for record in self._champions}

    def record_champion(self, epoch: int, miner_id: str, score: float) -> None:
        self._champions.append(ChampionRecord(epoch=epoch, miner_id=miner_id, score=score))
        if self._champion_history_rounds > 0:
            self._champions = self._champions[-self._champion_history_rounds :]
        else:
            self._champions = []

    def should_skip_qualification(
        self, miner_id: str, adapter_hash: str, *, skip_after_rounds: int
    ) -> bool:
        if skip_after_rounds <= 0:
            return False
        state = self._miner_state.get(miner_id)
        if state is None or state.last_adapter_hash != adapter_hash:
            return False
        return state.rounds_without_full_eval >= skip_after_rounds

    def record_round(
        self, miner_id: str, adapter_hash: str, *, was_selected_for_full_eval: bool
    ) -> None:
        state = self._miner_state.get(miner_id)
        if state is None or state.last_adapter_hash != adapter_hash:
            state = MinerRoundState(last_adapter_hash=adapter_hash, rounds_without_full_eval=0)
        if was_selected_for_full_eval:
            state.rounds_without_full_eval = 0
        else:
            state.rounds_without_full_eval += 1
        self._miner_state[miner_id] = state


__all__ = ["ChampionRecord", "MinerRoundState", "RoundStateStore"]
