import unittest

from thinker.validator.scoring import RolloutResult, stratified_score


class StratifiedScoreTest(unittest.TestCase):
    def test_band_mean_uses_sample_weights(self) -> None:
        score = stratified_score(
            [
                RolloutResult("math", "easy", "band-0", -1.0, sample_weight=0.05),
                RolloutResult("math", "rare", "band-0", 1.0, sample_weight=0.95),
            ]
        )

        self.assertAlmostEqual(score.per_band["band-0"], 0.9)
        self.assertAlmostEqual(score.overall, 0.9)

    def test_zero_weights_fall_back_to_unweighted_mean(self) -> None:
        score = stratified_score(
            [
                RolloutResult("math", "a", "band-0", -1.0, sample_weight=0.0),
                RolloutResult("math", "b", "band-0", 1.0, sample_weight=0.0),
            ]
        )

        self.assertEqual(score.per_band["band-0"], 0.0)


if __name__ == "__main__":
    unittest.main()
