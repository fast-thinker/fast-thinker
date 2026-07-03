import unittest

from thinker.problems.interface import extract_final_boxed_answer
from thinker.problems.tracks import constructive, depth_control, olympiad
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


class FinalBoxedAnswerTest(unittest.TestCase):
    def test_accepts_one_final_box_with_nested_latex(self) -> None:
        self.assertEqual(
            extract_final_boxed_answer(
                "Reasoning first.\n\\boxed{\\frac{1}{2}}"
            ),
            r"\frac{1}{2}",
        )

    def test_rejects_missing_multiple_or_nonfinal_box(self) -> None:
        invalid = (
            "10",
            r"\boxed{5} then \boxed{10}",
            r"\boxed{10} trailing text",
            r"\boxed{}",
            r"\boxed{10",
        )
        for output in invalid:
            with self.subTest(output=output):
                self.assertIsNone(extract_final_boxed_answer(output))

    def test_builtin_tracks_reject_unboxed_gold_answers(self) -> None:
        cases = (
            (
                constructive._track,
                "constructive-box-test",
                constructive._track._instance(
                    "constructive-box-test"
                ).get_solution(),
            ),
            (
                depth_control._track,
                "depth-box-test",
                ", ".join(
                    depth_control._track._build_task("depth-box-test").query_list
                ),
            ),
            (
                olympiad._track,
                "olympiad-box-test",
                str(olympiad._track._instance("olympiad-box-test").answer),
            ),
        )
        for track, seed, answer in cases:
            with self.subTest(track=track.track):
                self.assertTrue(track.verify(seed, rf"\boxed{{{answer}}}"))
                self.assertFalse(track.verify(seed, answer))
                self.assertFalse(track.verify(seed, rf"\boxed{{{answer}}} trailing"))

    def test_constructive_track_excludes_fixed_witness_families(self) -> None:
        active = {family.__name__ for family in constructive._FAMILIES}
        self.assertNotIn("_egyptian_fraction", active)
        self.assertNotIn("_gcd_lcm", active)


if __name__ == "__main__":
    unittest.main()
