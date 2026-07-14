import unittest

from thinker.validator.cli import _run_mode_tags, build_parser


class ValidatorCliTests(unittest.TestCase):
    def test_run_accepts_no_wandb_with_no_set_weights(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--wallet",
                "validator-wallet",
                "--hotkey",
                "validator-hotkey",
                "--evaluation-delay-epochs",
                "6",
                "--burn-rate",
                "1",
                "--no-set-weights",
                "--no-wandb",
            ]
        )

        self.assertTrue(args.no_set_weights)
        self.assertTrue(args.no_wandb)
        self.assertIsNone(args.test_mode)
        self.assertEqual(
            _run_mode_tags(args, args.test_mode),
            ("no_set_weights", "no_wandb"),
        )

    def test_test_mode_tags_disable_weights_and_wandb(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--test-mode", "math"])

        self.assertEqual(
            _run_mode_tags(args, args.test_mode),
            ("test_mode", "no_set_weights", "no_wandb"),
        )


if __name__ == "__main__":
    unittest.main()
