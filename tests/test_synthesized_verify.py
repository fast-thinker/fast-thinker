from __future__ import annotations

import unittest
from unittest.mock import patch

from thinker.problems.tracks import synthesized


class _LLM:
    def complete(self, prompt: str, *, temperature: float = 0.0, seed: int | None = None) -> str:
        raise AssertionError("test should not generate synthesized prompts")


def _track_with_gold(gold: str) -> synthesized.SynthesizedTrack:
    track = synthesized.SynthesizedTrack(_LLM())
    track._cache["seed"] = synthesized._SynthesizedInstance(
        gold_answer=gold,
        source_problem="problem",
        dataset_index=0,
        problem_id="problem-0",
        prompt="prompt",
        transform="test",
    )
    return track


class SynthesizedVerifyTests(unittest.TestCase):
    def test_render_uses_dataset_problem_without_llm_rewrite(self) -> None:
        track = synthesized.SynthesizedTrack(_LLM())
        track._source_cache["seed"] = synthesized._SynthesizedSource(
            gold_answer="42",
            source_problem="What is 40 + 2?",
            dataset_index=0,
            problem_id="problem-0",
        )

        self.assertEqual(track.render("seed"), "What is 40 + 2?")
        self.assertEqual(track._instance("seed").transform, "dataset_problem")

    def test_verify_compares_only_final_boxed_answer(self) -> None:
        track = _track_with_gold("42")
        output = "Final answer: \\boxed{42}"
        calls = []

        def fake_check(gold: str, pred: str) -> bool:
            calls.append((gold, pred))
            return True

        with patch.object(synthesized, "check_equivalence", side_effect=fake_check):
            self.assertTrue(track.verify("seed", output))

        self.assertEqual(calls, [(r"\boxed{42}", r"\boxed{42}")])

    def test_verify_handles_nested_braces_in_last_boxed_answer(self) -> None:
        track = _track_with_gold(r"\frac{1}{2}")
        output = r"After simplification, the answer is \boxed{\frac{1}{2}}"
        calls = []

        def fake_check(gold: str, pred: str) -> bool:
            calls.append((gold, pred))
            return True

        with patch.object(synthesized, "check_equivalence", side_effect=fake_check):
            self.assertTrue(track.verify("seed", output))

        self.assertEqual(calls, [(r"\boxed{\frac{1}{2}}", r"\boxed{\frac{1}{2}}")])

    def test_verify_rejects_multiple_boxed_answers(self) -> None:
        track = _track_with_gold("42")

        with patch.object(synthesized, "check_equivalence") as check:
            self.assertFalse(
                track.verify("seed", r"first \boxed{41}. Final answer: \boxed{42}")
            )

        check.assert_not_called()

    def test_verify_accepts_exact_boxed_answer(self) -> None:
        track = _track_with_gold(r"\frac{1}{2}")

        self.assertTrue(track.verify("seed", r"Final answer: \boxed{\frac{1}{2}}"))

    def test_verify_rejects_semantically_equivalent_format(self) -> None:
        track = _track_with_gold(r"\frac{1}{2}")

        self.assertFalse(track.verify("seed", r"Final answer: \boxed{1/2}"))

    def test_verify_rejects_unboxed_output(self) -> None:
        track = _track_with_gold("42")
        output = "Final answer: 42"

        with patch.object(synthesized, "check_equivalence") as check:
            self.assertFalse(track.verify("seed", output))

        check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
