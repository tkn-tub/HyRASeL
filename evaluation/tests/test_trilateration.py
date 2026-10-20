import unittest

import numpy as np

import sys
from pathlib import Path

# The modules of the evaluation chain live in src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import post_processing

# Anchor layout of a 3D test scenario (X, Y, Z).
ANCHORS = np.array([
    [150.0, 348.562, 453.469],
    [310.0, 268.562, 477.48],
    [220.0, 488.562, 381.63],
    [380.0, 418.562, 370.28],
])
SPEED_OF_LIGHT = 299792458.0


def _tofs(anchors: np.ndarray, position: np.ndarray, noise=None) -> list[float]:
    distances = np.linalg.norm(anchors - position, axis=1)
    if noise is not None:
        distances = distances + noise
    return list(distances / SPEED_OF_LIGHT)


class LinearStartTest(unittest.TestCase):
    def test_worked_example_from_the_documentation(self) -> None:
        anchors = np.array([
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0],
        ])
        distances = np.sqrt(np.array([50.0, 90.0, 70.0, 50.0]))

        start = post_processing._linear_trilateration_start(anchors, distances)

        np.testing.assert_allclose(start, [3.0, 4.0, 5.0], atol=1e-9)

    def test_coplanar_anchors_have_no_linear_start(self) -> None:
        anchors = ANCHORS.copy()
        anchors[:, 2] = 400.0
        distances = np.linalg.norm(anchors - np.array([250.0, 400.0, 400.0]), axis=1)

        self.assertIsNone(
            post_processing._linear_trilateration_start(anchors, distances)
        )

    def test_three_anchors_have_no_linear_start(self) -> None:
        self.assertIsNone(
            post_processing._linear_trilateration_start(
                ANCHORS[:3], np.array([100.0, 100.0, 100.0])
            )
        )


class TrilaterationTest(unittest.TestCase):
    def test_noise_free_ranges_are_recovered_for_nodes_near_the_anchor_mean_height(
        self,
    ) -> None:
        # A single start at the anchor centroid fails for about 19 % of these.
        rng = np.random.default_rng(0)
        mean_z = ANCHORS[:, 2].mean()
        for _ in range(100):
            truth = np.array([
                rng.uniform(150, 380),
                rng.uniform(268, 488),
                mean_z + rng.uniform(-40, 40),
            ])

            estimate = post_processing._trilaterate_position_from_tof(
                ANCHORS, _tofs(ANCHORS, truth)
            )

            self.assertLess(np.linalg.norm(estimate - truth), 0.1, msg=str(truth))

    def test_noisy_ranges_have_no_gross_errors(self) -> None:
        # Noise is amplified along Z for nodes near the anchor plane (largest
        # observed error about 3 m).  A wrong local minimum is off by tens of m.
        rng = np.random.default_rng(1)
        mean_z = ANCHORS[:, 2].mean()
        for _ in range(100):
            truth = np.array([
                rng.uniform(150, 380),
                rng.uniform(268, 488),
                mean_z + rng.uniform(-40, 40),
            ])

            estimate = post_processing._trilaterate_position_from_tof(
                ANCHORS, _tofs(ANCHORS, truth, rng.normal(0, 0.1, 4))
            )

            self.assertLess(np.linalg.norm(estimate - truth), 6.0, msg=str(truth))

    def test_coplanar_anchors_fall_back_to_centroid_start(self) -> None:
        anchors = ANCHORS.copy()
        anchors[:, 2] = 400.0
        truth = np.array([250.0, 380.0, 400.0])

        estimate = post_processing._trilaterate_position_from_tof(
            anchors, _tofs(anchors, truth)
        )

        np.testing.assert_allclose(estimate[:2], truth[:2], atol=1e-3)


if __name__ == "__main__":
    unittest.main()
