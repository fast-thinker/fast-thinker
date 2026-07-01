from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

from math_verify import parse
from math_verify import verify as _mv_verify

logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=4)
DEFAULT_TIMEOUT_S = 5.0
MAX_OUTPUT_CHARS = 20_000


def _run_with_timeout(fn, /, *args, timeout: float, **kwargs):
    future = _EXECUTOR.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError:
        logger.warning("math-verify call exceeded %ss timeout; treating as unverified", timeout)
        return None


def check_equivalence(
    gold: str,
    output: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> bool:
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS]

    gold_parsed = _run_with_timeout(parse, gold, parsing_timeout=None, timeout=timeout)
    if not gold_parsed:
        raise ValueError("gold answer failed to parse")

    pred_parsed = _run_with_timeout(parse, output, parsing_timeout=None, timeout=timeout)
    if not pred_parsed:
        return False

    result = _run_with_timeout(
        _mv_verify, gold_parsed, pred_parsed, timeout_seconds=None, timeout=timeout
    )
    return bool(result)
