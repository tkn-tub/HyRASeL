import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import sys
from pathlib import Path

# The modules of the evaluation chain live in src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import post_processing
from config_defaults import default_args


REQUIRED_CLI_ARGUMENTS = [
    "post_processing",
    "--method", "CENTRALIZED",
    "--experiment_path", "data/example",
    "--sweep_parameters", "interAnchorDist:130",
    "--results_dir", "results/example",
    *default_args("postprocess"),
]


class TdoaEstimatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.anchors = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [10.0, 10.0, 0.0],
            ]
        )
        target = np.array([3.0, 4.0, 0.0])
        ranges = np.linalg.norm(self.anchors - target, axis=1)
        self.time_differences = (
            ranges[1:] - ranges[0]
        ) / post_processing.SPEED_OF_LIGHT
        self.bounds = post_processing._build_localization_bounds_by_dimension(
            self.anchors
        )

    def _solve(self, **arguments):
        return post_processing.solve_tdoa_least_squares(
            [(self.anchors, self.time_differences)],
            localization_bounds_by_dimension=self.bounds,
            **{
                "solver_profile": "robust",
                "dimension_mode": "auto",
                "loss": "huber",
                "huber_f_scale_m": None,
                **arguments,
            },
        )

    def test_legacy_executes_one_fit(self) -> None:
        original_least_squares = post_processing.least_squares
        with patch.object(
            post_processing,
            "least_squares",
            wraps=original_least_squares,
        ) as least_squares_mock:
            _position, info = self._solve(solver_profile="legacy")

        self.assertEqual(least_squares_mock.call_count, 1)
        self.assertEqual(info["solver_profile"], "legacy")
        self.assertEqual(info["loss"], "linear")

    def test_huber_starts_from_a_linear_fit(self) -> None:
        original_least_squares = post_processing.least_squares
        with patch.object(
            post_processing,
            "least_squares",
            wraps=original_least_squares,
        ) as least_squares_mock:
            _position, info = self._solve()

        self.assertEqual(least_squares_mock.call_count, 2)
        self.assertEqual(info["loss"], "huber")

    def test_result_is_stored_on_the_event_row(self) -> None:
        position, info = self._solve()
        dataframe = pd.DataFrame(index=[0])
        post_processing._init_results_columns(dataframe)
        post_processing._store_localization_result(
            dataframe,
            current_index=0,
            estimated_pos=position,
            true_pos=np.array([3.0, 4.0, 0.0]),
            info=info,
            used_anchors_or_pairs=[(1, 2), (1, 3), (1, 4)],
        )
        merged = post_processing._merge_localization_results(dataframe)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.at[0, "localization_method"], "self_loc")
        self.assertAlmostEqual(merged.at[0, "est_position_error"], 0.0)
        self.assertEqual(merged.at[0, "tdoa_dimension"], 2)

    def test_cli_accepts_the_legacy_profile(self) -> None:
        with patch(
            "sys.argv",
            [*REQUIRED_CLI_ARGUMENTS, "--tdoa_solver_profile", "legacy"],
        ):
            arguments = post_processing._get_arguments_from_parser()
        self.assertEqual(arguments.tdoa_solver_profile, "legacy")

    def test_cli_rejects_unknown_options(self) -> None:
        with patch(
            "sys.argv",
            [*REQUIRED_CLI_ARGUMENTS, "--tdoa-loss", "linear"],
        ):
            with self.assertRaises(SystemExit):
                post_processing._get_arguments_from_parser()


if __name__ == "__main__":
    unittest.main()
