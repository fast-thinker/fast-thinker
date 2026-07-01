from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

from thinker.submission.commitments import read_submission
from thinker.validator.epoch_loop import MinerSubmissionPointer


# A replacement commitment receives a new chain block, resetting this delay.
# Six subnet epochs are approximately eight hours with the production tempo.
MODEL_EVALUATION_DELAY_EPOCHS = 6


@dataclass(frozen=True)
class MaturingSubmission:
    """A valid submission that is waiting for its chain-age delay."""

    miner_id: str
    uid: int
    committed_block: int
    eligible_block: int
    eligible_epoch: int
    remaining_blocks: int


def _operation_result(result: Any) -> tuple[bool, str]:
    if isinstance(result, tuple):
        return (
            bool(result[0]) if result else False,
            str(result[1]) if len(result) > 1 else str(result),
        )
    if isinstance(result, bool):
        return result, str(result)
    if hasattr(result, "success"):
        success = bool(result.success)
        message = getattr(result, "message", None)
        if not message and not success:
            # bittensor's ExtrinsicResponse leaves message=None whenever a
            # call fails before it ever reaches the chain (e.g. set_weights
            # exits its internal retry loop without making an attempt when
            # the weights-rate-limit hasn't elapsed yet) -- .error or a full
            # repr is the only way to see why in that case. repr(), not
            # str(): an exception raised with multiple positional args
            # (e.g. SomeError(False, None)) stringifies as just "(False,
            # None)" with no type name, which is indistinguishable from a
            # bare tuple -- repr() keeps the exception's class name visible.
            error = getattr(result, "error", None)
            message = repr(error) if error is not None else repr(result)
        return success, str(message)
    if hasattr(result, "is_success"):
        return bool(result.is_success), str(getattr(result, "error_message", result))
    return False, f"unexpected chain response: {result!r}"


def current_epoch(subtensor: Any, epoch_blocks: int) -> int:
    if epoch_blocks <= 0:
        raise ValueError("epoch_blocks must be positive")
    return subtensor.get_current_block() // epoch_blocks


def discover_miner_pointers(
    subtensor: Any,
    netuid: int,
    metagraph: Any,
    epoch: int,
    *,
    epoch_blocks: int = 360,
    evaluation_delay_epochs: int = MODEL_EVALUATION_DELAY_EPOCHS,
    current_block: int | None = None,
    maturing_submissions: list[MaturingSubmission] | None = None,
) -> dict[str, MinerSubmissionPointer]:
    if epoch_blocks <= 0:
        raise ValueError("epoch_blocks must be positive")
    if evaluation_delay_epochs < 0:
        raise ValueError("evaluation_delay_epochs must be non-negative")
    if evaluation_delay_epochs and current_block is None:
        current_block = subtensor.get_current_block()
    if current_block is not None and current_block < 0:
        raise ValueError("current_block must be non-negative")

    pointers: dict[str, MinerSubmissionPointer] = {}
    for uid, hotkey in enumerate(metagraph.hotkeys):
        if not hotkey:
            continue
        commitment = read_submission(subtensor, netuid, hotkey, epoch)
        if commitment is None:
            continue
        # Do not trust commitment.epoch for age: miners choose that field and
        # could backdate it. The chain records when the current commitment was
        # actually published, so replacing a model starts a fresh wait.
        if evaluation_delay_epochs:
            if commitment.block is None:
                continue
            required_age_blocks = evaluation_delay_epochs * epoch_blocks
            if current_block is None:
                continue
            eligible_block = commitment.block + required_age_blocks
            if current_block < eligible_block:
                if maturing_submissions is not None:
                    maturing_submissions.append(
                        MaturingSubmission(
                            miner_id=hotkey,
                            uid=uid,
                            committed_block=commitment.block,
                            eligible_block=eligible_block,
                            eligible_epoch=eligible_block // epoch_blocks,
                            remaining_blocks=eligible_block - current_block,
                        )
                    )
                continue
        pointers[hotkey] = MinerSubmissionPointer(
            miner_id=hotkey,
            epoch=commitment.epoch,
            repo_id=commitment.repo_id,
            sha256=commitment.sha256,
        )
    return pointers


class BittensorWeightSetter:
    def __init__(self, subtensor: Any, wallet: Any, netuid: int):
        self._subtensor = subtensor
        self._wallet = wallet
        self._netuid = netuid

    def set_weights(self, scores: dict[str, float]) -> None:
        metagraph = self._subtensor.metagraph(self._netuid)
        hotkey_to_uid = {hotkey: uid for uid, hotkey in enumerate(metagraph.hotkeys)}
        clipped = {
            hotkey: max(0.0, score)
            for hotkey, score in scores.items()
            if hotkey in hotkey_to_uid
        }
        total = sum(clipped.values())
        uids: list[int] = []
        weights: list[float] = []
        if total <= 0:
            uids.append(0)
            weights.append(1.0)
        else:
            for hotkey, score in clipped.items():
                uids.append(hotkey_to_uid[hotkey])
                weights.append(score / total)
        result = self._subtensor.set_weights(
            wallet=self._wallet,
            netuid=self._netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=False,
            wait_for_finalization=False,
        )
        success, message = _operation_result(result)
        if not success:
            raise RuntimeError(f"failed to set evaluation weights: {message}")


class NullWeightSetter:
    """No-op WeightSetter for --no-set-weights: scores are still computed by
    the epoch loop but never written to chain. Exposes the same
    start()/stop()/set_weights() surface as PeriodicWeightSetter so cli.py
    doesn't need to special-case it."""

    def __init__(self, log: Callable[[str], None] | None = None):
        self._log = log or (lambda message: None)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def set_weights(self, scores: dict[str, float]) -> None:
        self._log(f"weight-set skipped (--no-set-weights): {len(scores)} score(s) computed")


class PeriodicWeightSetter:
    """Decouples "deciding what the weights should be" (the evaluation
    loop's pace, governed by epoch_blocks -- could be an hour or more
    between epochs) from "writing weights to chain" (governed by
    Bittensor's weights_rate_limit, a fixed block interval per hotkey that
    has nothing to do with how often the validator re-scores miners).

    Implements the same set_weights(scores) -> None interface as
    BittensorWeightSetter (the WeightSetter Protocol epoch_loop.py expects),
    so it's a drop-in replacement for EpochLoop -- but here set_weights only
    records the latest scores; a dedicated background thread is the only
    thing that actually calls the chain, on its own fixed cadence,
    independent of evaluation. The timer thread retries on a fixed interval
    and logs-and-continues on failure (for example, a rate-limit rejection)
    rather than crashing or blocking evaluation on a chain write. No chain
    write occurs until the epoch loop provides its first set of evaluation
    scores."""

    def __init__(
        self,
        inner: BittensorWeightSetter,
        *,
        interval_seconds: float = 600,
        log: Callable[[str], None] | None = None,
    ):
        self._inner = inner
        self._interval_seconds = interval_seconds
        self._log = log or (lambda message: None)
        self._lock = threading.Lock()
        self._latest_scores: dict[str, float] | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def set_weights(self, scores: dict[str, float]) -> None:
        with self._lock:
            self._latest_scores = dict(scores)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="thinker-weight-setter", daemon=True
        )
        self._thread.start()

    def stop(self, *, timeout: float | None = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._write_once()
            self._stop_event.wait(self._interval_seconds)

    def _write_once(self) -> None:
        with self._lock:
            scores = dict(self._latest_scores) if self._latest_scores is not None else None
        if scores is None:
            return
        try:
            self._inner.set_weights(scores)
            self._log(f"weight-set succeeded: {len(scores)} score(s) written")
        except Exception as exc:
            self._log(
                f"weight-set rejected (will retry in {self._interval_seconds:.0f}s): "
                f"{type(exc).__name__}: {exc}"
            )
