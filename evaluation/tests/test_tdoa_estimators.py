import unittest
from unittest.mock import patch

import numpy as np

import sys
from pathlib import Path

# The modules of the evaluation chain live in src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import localization
import post_processing
import solvers
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
        ) / solvers.SPEED_OF_LIGHT
        self.bounds = solvers.localization_bounds_by_dimension(
            self.anchors
        )

    def _solve(self, **arguments):
        return solvers.solve_tdoa_least_squares(
            [(self.anchors, self.time_differences)],
            localization_bounds_by_dimension=self.bounds,
            **{
                "dimension_mode": "auto",
                "loss": "huber",
                "huber_f_scale_m": None,
                **arguments,
            },
        )

    def test_linear_loss_executes_one_fit(self) -> None:
        original_least_squares = solvers.least_squares
        with patch.object(
            solvers,
            "least_squares",
            wraps=original_least_squares,
        ) as least_squares_mock:
            _position, info = self._solve(loss="linear")

        self.assertEqual(least_squares_mock.call_count, 1)
        self.assertEqual(info["tdoa_loss"], "linear")

    def test_huber_starts_from_a_linear_fit(self) -> None:
        original_least_squares = solvers.least_squares
        with patch.object(
            solvers,
            "least_squares",
            wraps=original_least_squares,
        ) as least_squares_mock:
            _position, info = self._solve()

        self.assertEqual(least_squares_mock.call_count, 2)
        self.assertEqual(info["tdoa_loss"], "huber")

    def test_jacobian_is_the_derivative_of_the_residuals(self) -> None:
        anchors = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 1.0], [0.0, 10.0, 2.0], [10.0, 10.0, 3.0]])
        position = np.array([3.0, 4.0, 5.0])

        def modeled_residuals(p: np.ndarray) -> np.ndarray:
            return -(np.linalg.norm(anchors[1:] - p, axis=1) - np.linalg.norm(p - anchors[0]))

        step = 1e-6
        numeric = np.zeros((3, 3))
        for axis in range(3):
            shift = np.zeros(3)
            shift[axis] = step
            numeric[:, axis] = (modeled_residuals(position + shift) - modeled_residuals(position - shift)) / (2 * step)
        tdoa_sets = [(anchors, np.zeros(3))]

        np.testing.assert_allclose(solvers._tdoa_jacobian(tdoa_sets, position, 3), numeric, atol=1e-8)
        np.testing.assert_allclose(solvers._tdoa_jacobian(tdoa_sets, position, 2), numeric[:, :2], atol=1e-8)

    def test_result_holds_the_estimate_and_its_diagnostics(self) -> None:
        position, info = self._solve()
        result = localization._tdoa_result(
            np.array([3.0, 4.0, 0.0]), position, info, [(1, 2), (1, 3), (1, 4)], stitched=False
        )

        self.assertEqual(result["localization_method"], "self_loc")
        self.assertAlmostEqual(result["est_position_error"], 0.0)
        self.assertEqual(result["tdoa_dimension"], 2)

    def test_cli_rejects_the_removed_solver_profile(self) -> None:
        with patch(
            "sys.argv",
            [*REQUIRED_CLI_ARGUMENTS, "--tdoa_solver_profile", "robust"],
        ):
            with self.assertRaises(SystemExit):
                post_processing.parse_arguments()

    def test_cli_rejects_unknown_options(self) -> None:
        with patch(
            "sys.argv",
            [*REQUIRED_CLI_ARGUMENTS, "--tdoa-loss", "linear"],
        ):
            with self.assertRaises(SystemExit):
                post_processing.parse_arguments()


if __name__ == "__main__":
    unittest.main()
