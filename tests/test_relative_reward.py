from __future__ import annotations

import unittest

from thinker.reward.relative import (
    peer_completion_efficiency_rewards,
    relative_reasoning_reward,
)


class RelativeReasoningRewardTest(unittest.TestCase):
    def test_wrong_miner_is_penalized(self) -> None:
        self.assertEqual(
            relative_reasoning_reward(
                original_verified=True,
                miner_verified=False,
                original_completion_len=100,
                miner_completion_len=10,
            ),
            -1.0,
        )

    def test_original_correct_uses_bounded_efficiency_bonus(self) -> None:
        self.assertEqual(
            relative_reasoning_reward(
                original_verified=True,
                miner_verified=True,
                original_completion_len=100,
                miner_completion_len=0,
            ),
            1.25,
        )
        self.assertEqual(
            relative_reasoning_reward(
                original_verified=True,
                miner_verified=True,
                original_completion_len=100,
                miner_completion_len=100,
            ),
            1.0,
        )
        self.assertEqual(
            relative_reasoning_reward(
                original_verified=True,
                miner_verified=True,
                original_completion_len=100,
                miner_completion_len=1000,
            ),
            0.75,
        )

    def test_original_wrong_single_correct_gets_full_baseline_miss_credit(self) -> None:
        self.assertEqual(
            relative_reasoning_reward(
                original_verified=False,
                miner_verified=True,
                original_completion_len=100,
                miner_completion_len=1000,
            ),
            1.5,
        )

    def test_peer_efficiency_scores_only_correct_miners_on_baseline_miss(self) -> None:
        rewards = peer_completion_efficiency_rewards(
            original_verified=False,
            miner_verified=[True, True, True, False],
            miner_completion_lens=[20, 30, 50, 1],
            base_rewards=[1.5, 1.5, 1.5, -1.0],
        )
        self.assertEqual(len(rewards), 4)
        for actual, expected in zip(
            rewards,
            [1.5, 1.3333333333333333, 1.0, -1.0],
        ):
            self.assertAlmostEqual(actual, expected)

    def test_peer_efficiency_does_not_override_original_correct_rewards(self) -> None:
        self.assertEqual(
            peer_completion_efficiency_rewards(
                original_verified=True,
                miner_verified=[True, True],
                miner_completion_lens=[10, 100],
                base_rewards=[1.25, 1.0],
            ),
            [1.25, 1.0],
        )


if __name__ == "__main__":
    unittest.main()
