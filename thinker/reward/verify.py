from __future__ import annotations

import logging
import multiprocessing as mp
import queue
from typing import Literal

try:
    import resource
except ImportError:  # Windows
    resource = None

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 5.0
MAX_OUTPUT_CHARS = 20_000
_WORKER_MEMORY_BYTES = 512 * 1024 * 1024

_WorkerStatus = Literal[
    "ok",
    "gold_parse_error",
    "pred_parse_error",
    "error",
]


def _apply_worker_limits() -> None:
    if resource is None:
        return
    try:
        resource.setrlimit(
            resource.RLIMIT_AS,
            (_WORKER_MEMORY_BYTES, _WORKER_MEMORY_BYTES),
        )
    except (OSError, ValueError, AttributeError):
        pass


def _math_verify_worker(
    gold: str,
    output: str,
    response_queue: mp.Queue,
) -> None:
    _apply_worker_limits()
    try:
        from math_verify import parse
        from math_verify import verify as mv_verify

        gold_parsed = parse(gold, parsing_timeout=None)
        if not gold_parsed:
            response_queue.put(("gold_parse_error", False))
            return

        pred_parsed = parse(output, parsing_timeout=None)
        if not pred_parsed:
            response_queue.put(("pred_parse_error", False))
            return

        result = mv_verify(gold_parsed, pred_parsed, timeout_seconds=None)
        response_queue.put(("ok", bool(result)))
    except Exception as exc:
        response_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _verify_in_subprocess(
    gold: str,
    output: str,
    *,
    timeout: float,
) -> tuple[_WorkerStatus, bool | str | None]:
    context = mp.get_context("spawn")
    response_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_math_verify_worker,
        args=(gold, output, response_queue),
    )
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join()
        response_queue.close()
        response_queue.join_thread()
        logger.warning(
            "math-verify worker exceeded %ss timeout; treating as unverified",
            timeout,
        )
        return "error", None

    try:
        status, payload = response_queue.get_nowait()
    except queue.Empty:
        logger.warning(
            "math-verify worker exited without a result; treating as unverified"
        )
        return "error", None
    finally:
        response_queue.close()
        response_queue.join_thread()
    return status, payload


def check_equivalence(
    gold: str,
    output: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> bool:
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS]

    status, payload = _verify_in_subprocess(gold, output, timeout=timeout)
    if status == "gold_parse_error":
        raise ValueError("gold answer failed to parse")
    if status == "ok":
        return bool(payload)
    if status == "error":
        logger.debug("math-verify worker failed: %s", payload)
        return False
    return False
