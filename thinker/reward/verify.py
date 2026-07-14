from __future__ import annotations

DEFAULT_TIMEOUT_S = 5.0
MAX_OUTPUT_CHARS = 20_000


def _strict_normalize(text: str) -> str:
    return text.strip()


def check_equivalence(
    gold: str,
    output: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> bool:
    if len(output) > MAX_OUTPUT_CHARS:
        return False

    return _strict_normalize(gold) == _strict_normalize(output)
