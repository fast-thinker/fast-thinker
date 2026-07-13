import importlib
import importlib.machinery
import importlib.util
import queue
import sys
import types
import unittest
from unittest.mock import patch

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


class _EmptyQueue:
    def __init__(self) -> None:
        self.closed = False
        self.joined = False

    def get_nowait(self):
        raise queue.Empty

    def close(self) -> None:
        self.closed = True

    def join_thread(self) -> None:
        self.joined = True


class _HangingProcess:
    def __init__(self) -> None:
        self.started = False
        self.terminated = False
        self.killed = False
        self.join_timeouts: list[float | None] = []
        self._alive = True

    def start(self) -> None:
        self.started = True

    def join(self, timeout=None) -> None:
        self.join_timeouts.append(timeout)
        if self.terminated:
            self._alive = False

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self._alive = False


class _FakeContext:
    def __init__(self) -> None:
        self.queue = _EmptyQueue()
        self.process = _HangingProcess()

    def Queue(self, maxsize=0):
        return self.queue

    def Process(self, target, args):
        self.target = target
        self.args = args
        return self.process


class MathVerifyIsolationTests(unittest.TestCase):
    def test_subprocess_timeout_terminates_worker(self) -> None:
        context = _FakeContext()
        with patch.object(reward_verify.mp, "get_context", return_value=context):
            status, payload = reward_verify._verify_in_subprocess(
                r"\boxed{1}",
                r"\boxed{1}",
                timeout=0.01,
            )

        self.assertEqual(status, "error")
        self.assertIsNone(payload)
        self.assertTrue(context.process.started)
        self.assertTrue(context.process.terminated)
        self.assertEqual(context.process.join_timeouts[:2], [0.01, 1.0])
        self.assertTrue(context.queue.closed)
        self.assertTrue(context.queue.joined)

    def test_gold_parse_error_is_not_silently_accepted(self) -> None:
        with patch.object(
            reward_verify,
            "_verify_in_subprocess",
            return_value=("gold_parse_error", False),
        ):
            with self.assertRaises(ValueError):
                reward_verify.check_equivalence("bad", r"\boxed{1}")

    def test_worker_failure_marks_prediction_unverified(self) -> None:
        with patch.object(
            reward_verify,
            "_verify_in_subprocess",
            return_value=("error", "boom"),
        ):
            self.assertFalse(reward_verify.check_equivalence(r"\boxed{1}", "boom"))


def _import_procedural_track_module():
    if (
        "reasoning_gym" not in sys.modules
        and importlib.util.find_spec("reasoning_gym") is None
    ):
        fake_reasoning_gym = types.ModuleType("reasoning_gym")
        fake_reasoning_gym.__spec__ = importlib.machinery.ModuleSpec(
            "reasoning_gym",
            loader=None,
        )
        fake_reasoning_gym.create_dataset = None
        sys.modules["reasoning_gym"] = fake_reasoning_gym
    return importlib.import_module("thinker.problems.tracks.procedural")


class ProceduralVerifyIsolationTests(unittest.TestCase):
    def test_procedural_timeout_terminates_worker(self) -> None:
        procedural = _import_procedural_track_module()

        context = _FakeContext()
        with patch.object(procedural.mp, "get_context", return_value=context):
            score = procedural._score_answer_with_timeout(
                "seed",
                ("polynomial_equations",),
                "1",
                timeout=0.01,
            )

        self.assertIsNone(score)
        self.assertTrue(context.process.started)
        self.assertTrue(context.process.terminated)
        self.assertEqual(context.process.join_timeouts[:2], [0.01, 1.0])
        self.assertTrue(context.queue.closed)
        self.assertTrue(context.queue.joined)


if __name__ == "__main__":
    unittest.main()
