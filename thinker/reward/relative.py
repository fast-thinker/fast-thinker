from __future__ import annotations


EFFICIENCY_WEIGHT = 0.5
EFFICIENCY_CLAMP = 0.5


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _baseline_efficiency_score(
    *,
    original_completion_len: int,
    miner_completion_len: int,
) -> float:
    denominator = max(1, int(original_completion_len))
    numerator = max(0, int(miner_completion_len))
    savings_ratio = (denominator - numerator) / denominator
    return _clamp(savings_ratio, -EFFICIENCY_CLAMP, EFFICIENCY_CLAMP)


def _correct_reward(efficiency_score: float) -> float:
    return 1.0 + (EFFICIENCY_WEIGHT * efficiency_score)


def relative_reasoning_reward(
    *,
    original_verified: bool,
    miner_verified: bool,
    original_completion_len: int,
    miner_completion_len: int,
) -> float:
    if not miner_verified:
        return -1.0
    if not original_verified:
        return _correct_reward(1.0)

    return _correct_reward(
        _baseline_efficiency_score(
            original_completion_len=original_completion_len,
            miner_completion_len=miner_completion_len,
        )
    )


def peer_completion_efficiency_rewards(
    *,
    original_verified: bool,
    miner_verified: list[bool],
    miner_completion_lens: list[int],
    base_rewards: list[float],
) -> list[float]:
    """Apply peer-relative token efficiency when miners fix a baseline miss.

    Correct miners are compared only with other correct miners for the same
    problem. The shortest correct completion gets full efficiency credit and
    the longest correct completion gets the base correct score. Incorrect
    rewards are left untouched.
    """
    if not (
        len(miner_verified)
        == len(miner_completion_lens)
        == len(base_rewards)
    ):
        raise ValueError("miner reward inputs must have matching lengths")

    rewards = list(base_rewards)
    if original_verified:
        return rewards

    correct_indexes = [
        index for index, verified in enumerate(miner_verified) if verified
    ]
    if not correct_indexes:
        return rewards

    token_counts = [
        max(0, int(miner_completion_lens[index])) for index in correct_indexes
    ]
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    if max_tokens == min_tokens:
        for index in correct_indexes:
            rewards[index] = _correct_reward(1.0)
        return rewards

    for index in correct_indexes:
        scaled = (
            max(0, int(miner_completion_lens[index])) - min_tokens
        ) / (max_tokens - min_tokens)
        rewards[index] = _correct_reward(1.0 - scaled)
    return rewards


__all__ = [
    "EFFICIENCY_CLAMP",
    "EFFICIENCY_WEIGHT",
    "peer_completion_efficiency_rewards",
    "relative_reasoning_reward",
]
