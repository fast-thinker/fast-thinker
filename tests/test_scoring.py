import os
import unittest
from unittest import mock

from thinker.common_seed import COMMON_SAMPLE_RATE, build_sample_seed_plan
from thinker.config import ThinkerConfig
from thinker.problems.interface import extract_final_boxed_answer
from thinker.problems.tracks import constructive, depth_control, olympiad
from thinker.reward import verify as reward_verify
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


class StrictMatchingTests(unittest.TestCase):
    def test_exact_answer_is_accepted(self) -> None:
        self.assertTrue(reward_verify.check_equivalence(r"\boxed{12}", r"\boxed{12}"))

    def test_surrounding_whitespace_is_ignored(self) -> None:
        self.assertTrue(
            reward_verify.check_equivalence(
                "  " + r"\boxed{12}" + "\n",
                "\n" + r"\boxed{12}" + "  ",
            )
        )

    def test_different_formatting_is_rejected(self) -> None:
        rejected = (
            ("12", r"\boxed{12}"),
            (r"\boxed{\frac{1}{2}}", r"\boxed{1/2}"),
            (r"\boxed{12}", r"\boxed{12.0}"),
        )
        for gold, output in rejected:
            with self.subTest(gold=gold, output=output):
                self.assertFalse(reward_verify.check_equivalence(gold, output))

    def test_long_output_is_rejected_before_matching(self) -> None:
        self.assertFalse(
            reward_verify.check_equivalence(
                "x" * (reward_verify.MAX_OUTPUT_CHARS + 1),
                "x" * (reward_verify.MAX_OUTPUT_CHARS + 1),
            )
        )
        self.assertFalse(
            reward_verify.check_equivalence(
                "x" * reward_verify.MAX_OUTPUT_CHARS,
                "x" * (reward_verify.MAX_OUTPUT_CHARS + 1),
            )
        )


class EvaluationDefaultsTest(unittest.TestCase):
    def test_full_evaluation_defaults_are_thirty_math_twenty_long_qa(self) -> None:
        with mock.patch.dict(os.environ, {"USERPROFILE": "C:\\Users\\test"}, clear=True):
            config = ThinkerConfig()

        self.assertEqual(config.n_problems_per_epoch, 30)
        self.assertEqual(config.n_long_context_qa_per_epoch, 20)

    def test_common_seed_rate_uses_eighty_percent_of_batch(self) -> None:
        self.assertEqual(COMMON_SAMPLE_RATE, 0.8)

        math_plan = build_sample_seed_plan(
            30,
            private_seed="private",
            epoch=7,
            namespace="full_evaluation:math",
            common_seed="0" * 64,
        )
        long_qa_plan = build_sample_seed_plan(
            20,
            private_seed="private",
            epoch=7,
            namespace="full_evaluation:long_context_qa",
            common_seed="0" * 64,
        )

        self.assertEqual(math_plan.common_count, 24)
        self.assertEqual(long_qa_plan.common_count, 16)


if __name__ == "__main__":
    unittest.main()
