from __future__ import annotations


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
        return 1.0

    denominator = max(1, int(original_completion_len))
    numerator = max(0, int(miner_completion_len))
    return 1.0 - (numerator / denominator)


def peer_completion_efficiency_rewards(
    *,
    original_verified: bool,
    miner_verified: list[bool],
    miner_completion_lens: list[int],
    base_rewards: list[float],
) -> list[float]:
    """Apply peer-relative token efficiency when miners fix a baseline miss.

    The ordinary reward gives every correct miner 1.0 when the original model
    is wrong. In cross-miner batches we instead compare only the correct miners
    for that same problem, min-max scale their completion lengths, and assign
    1 - scaled_tokens. Incorrect or otherwise special-cased base rewards are
    left untouched.
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
            rewards[index] = 1.0
        return rewards

    for index in correct_indexes:
        scaled = (
            max(0, int(miner_completion_lens[index])) - min_tokens
        ) / (max_tokens - min_tokens)
        rewards[index] = 1.0 - scaled
    return rewards


__all__ = ["relative_reasoning_reward", "peer_completion_efficiency_rewards"]
