import re
import unittest

from thinker.validator.multiple_choice import MultipleChoiceEvaluator, _shuffle_options


class MultipleChoiceOptionShuffleTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "problem_id": "planet",
            "input": (
                "Which planet is second from the Sun?\n"
                "A. Mercury\n"
                "B. Venus\n"
                "C. Earth\n"
                "D. Mars"
            ),
            "expected_answer": "B",
        }

    def test_shuffle_is_deterministic_and_remaps_gold_label(self):
        first = _shuffle_options(self.row["input"], "B", "seed-1")
        second = _shuffle_options(self.row["input"], "B", "seed-1")
        self.assertEqual(first, second)

        prompt, ground_truth = first
        option_by_label = dict(
            re.findall(r"(?m)^([A-Z])\) (.+)$", prompt)
        )
        self.assertEqual(option_by_label[ground_truth], "Venus")
        self.assertEqual(
            set(option_by_label.values()), {"Mercury", "Venus", "Earth", "Mars"}
        )
        self.assertTrue(set(option_by_label).isdisjoint({"A", "B", "C", "D"}))

    def test_instance_uses_seeded_layout_and_remapped_answer(self):
        evaluator = MultipleChoiceEvaluator(inference=None, rows=[self.row])
        instance = evaluator.generate_instances(["seed-2"])[0]
        option_by_label = dict(
            re.findall(r"(?m)^([A-Z])\) (.+)$", instance.prompt)
        )
        self.assertEqual(option_by_label[instance.ground_truth], "Venus")
        self.assertEqual(instance.problem_id, "planet")

    def test_multiline_option_text_is_preserved(self):
        prompt = "Pick the true statement.\nA: First line\ncontinued\nB: Other"
        shuffled_prompt, ground_truth = _shuffle_options(prompt, "A", "seed-3")
        self.assertIn("First line\ncontinued", shuffled_prompt)
        options = dict(re.findall(r"(?ms)^([A-Z])\) (.*?)(?=^[A-Z]\) |\Z)", shuffled_prompt))
        self.assertIn("First line\ncontinued", options[ground_truth])

    def test_inline_dataset_option_format_is_supported(self):
        prompt = (
            "Solve the following problem. Which material is conductive? "
            "A: Glass B: Copper C: Rubber D: Wood"
        )
        shuffled_prompt, ground_truth = _shuffle_options(prompt, "B", "seed-4")
        options = dict(re.findall(r"(?m)^([A-Z])\) (.+)$", shuffled_prompt))
        self.assertEqual(options[ground_truth], "Copper")
        self.assertEqual(set(options.values()), {"Glass", "Copper", "Rubber", "Wood"})

    def test_answer_preserves_full_response_for_test_mode_logging(self):
        response = "Reasoning over every option.\nTherefore \\boxed{Q}."
        answer = MultipleChoiceEvaluator._answer(response, 17, "Q")
        self.assertTrue(answer.verified)
        self.assertEqual(answer.text, "Q")
        self.assertEqual(answer.response, response)
        self.assertEqual(answer.completion_len, 17)


if __name__ == "__main__":
    unittest.main()
