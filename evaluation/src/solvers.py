"""Position solvers: trilateration from TWR times of flight and TDoA least squares.

Pure numerics without trace handling; localization.py collects the
measurements.  Positions are 3D (X, Y, Z) in metres, times in seconds.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

SPEED_OF_LIGHT = 299792458.0

# 1.4826 MAD estimates the standard deviation of normally distributed values.
MAD_TO_SIGMA = 1.4826

# Below this ratio of smallest to largest singular value, the anchors count as
# (nearly) coplanar and the closed-form start is not used.
TRILATERATION_MIN_REL_SINGULAR_VALUE = 1e-3

# Relative Z span below which the anchors of a TDoA solve count as planar.
TDOA_PLANAR_REL_TOLERANCE = 1e-6

# Without a fixed f_scale (--tdoa_f_scale), it is estimated locally from a
# preliminary least-squares solution.  The fallback and limits keep that
# estimate usable when only a minimally determined system or a heavily
# corrupted event is available.
TDOA_HUBER_SCALE_FACTOR = 1.345
TDOA_HUBER_F_SCALE_FALLBACK_M = 1.0
TDOA_HUBER_F_SCALE_MIN_M = 0.05
TDOA_HUBER_F_SCALE_MAX_M = 10.0

# Bounds are derived from physical (not virtual) anchors.  Z deliberately gets
# a larger margin because a future 3D deployment can extend above/below them.
LOCALIZATION_XY_BOUND_MARGIN_RATIO = 0.30
LOCALIZATION_Z_BOUND_MARGIN_RATIO = 0.50
LOCALIZATION_MIN_BOUND_REFERENCE_SPAN_M = 1.0


def twr_time_of_flight(R_a: float, R_b: float, D_a: float, D_b: float) -> float:
    """Time of flight of a double-sided TWR exchange from its round (R) and reply (D) times."""
    return (R_a * R_b - D_a * D_b) / (R_a + D_a + R_b + D_b)


def linear_trilateration_start(anchors: np.ndarray, distances: np.ndarray) -> np.ndarray | None:
    """Closed-form position estimate from ranges (no start value required).

    Subtracting the sphere equation |p - a_1|^2 = d_1^2 from |p - a_i|^2 = d_i^2
    cancels the quadratic term in p and leaves one linear equation per anchor:

        2 (a_i - a_1) . p = |a_i|^2 - |a_1|^2 - d_i^2 + d_1^2

    The system is solved in least squares.  It is noise sensitive, so the result
    is only meant as the start of the nonlinear fit.  Coordinates are shifted to
    the anchor centroid for numerical conditioning.

    Returns:
        np.ndarray | None: The estimate, or None if it is not determined, i.e.
        fewer than 4 anchors or (nearly) coplanar anchors.
    """
    if len(anchors) < 4:
        return None

    centroid = anchors.mean(axis=0)
    shifted = anchors - centroid
    # One row of the linear equation above per anchor i >= 2, in shifted coordinates.
    system_matrix = 2.0 * (shifted[1:] - shifted[0])
    right_hand_side = (
        np.sum(shifted[1:] ** 2, axis=1)
        - np.sum(shifted[0] ** 2)
        - distances[1:] ** 2
        + distances[0] ** 2
    )

    # Singular values in descending order; a small last one means (nearly) coplanar anchors.
    singular_values = np.linalg.svd(system_matrix, compute_uv=False)
    if (
        singular_values.size < 3
        or singular_values[0] <= 0
        or singular_values[2] / singular_values[0] < TRILATERATION_MIN_REL_SINGULAR_VALUE
    ):
        return None

    solution = np.linalg.lstsq(system_matrix, right_hand_side, rcond=None)[0]
    if not np.all(np.isfinite(solution)):
        return None
    # Back to the original coordinates.
    return solution + centroid


def trilaterate(anchor_positions, times_of_flight) -> np.ndarray:
    """3D position from the times of flight to the anchors.

    The range equations are non-convex, so a least-squares run started at the
    anchor centroid can converge to a wrong local minimum (typically on the
    wrong side of the anchor geometry in Z), even for noise-free ranges.  If the
    anchors are not coplanar, the fit therefore starts at a closed-form linear
    estimate (linear_trilateration_start) that lies in the correct basin.  For
    coplanar anchors, e.g. in 2D experiments, the anchor centroid is the start.
    """
    anchors = np.array(anchor_positions, dtype=float)
    distances = np.array(times_of_flight, dtype=float) * SPEED_OF_LIGHT

    def residual(p, anchors, d):
        """Modeled minus measured distances."""
        return np.sqrt(((anchors - p) ** 2).sum(axis=1)) - d

    start = linear_trilateration_start(anchors, distances)
    if start is None:
        start = anchors.mean(axis=0)
    return least_squares(residual, start, args=(anchors, distances)).x


def localization_bounds(
    physical_anchor_positions: np.ndarray, dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solver bounds (lower, upper) around the bounding box of the physical anchors."""
    anchor_min = np.min(physical_anchor_positions, axis=0)
    anchor_max = np.max(physical_anchor_positions, axis=0)
    # One XY margin from the largest horizontal extent, so that a narrow or
    # zero span on one axis does not also produce a zero margin.
    largest_xy_span = float(np.max((anchor_max - anchor_min)[:2]))
    reference_span = max(largest_xy_span, LOCALIZATION_MIN_BOUND_REFERENCE_SPAN_M)

    xy_margin = LOCALIZATION_XY_BOUND_MARGIN_RATIO * reference_span
    lower = anchor_min[:2] - xy_margin
    upper = anchor_max[:2] + xy_margin
    if dimension == 3:
        z_margin = LOCALIZATION_Z_BOUND_MARGIN_RATIO * reference_span
        lower = np.append(lower, anchor_min[2] - z_margin)
        upper = np.append(upper, anchor_max[2] + z_margin)
    return lower.astype(float), upper.astype(float)


def localization_bounds_by_dimension(
    physical_anchor_positions: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """2D and 3D solver bounds of one run."""
    return {dimension: localization_bounds(physical_anchor_positions, dimension) for dimension in (2, 3)}


def _tdoa_dimension(all_anchor_positions: np.ndarray, dimension_mode: str) -> tuple[int, float | None]:
    """Solve dimension (2 or 3) and, in 2D, the fixed height (median anchor Z)."""
    spans = np.ptp(all_anchor_positions, axis=0)
    # Largest XY span, at least 1 m, so that the tolerance below is never zero.
    reference_span = max(float(np.max(spans[:2])), 1.0)
    # Planar anchors constrain the target to their plane; the relative
    # tolerance allows for small coordinate noise.
    anchors_are_planar = float(spans[2]) <= TDOA_PLANAR_REL_TOLERANCE * reference_span
    if dimension_mode == "3d" or (dimension_mode == "auto" and not anchors_are_planar):
        return 3, None
    return 2, float(np.median(all_anchor_positions[:, 2]))


def _tdoa_jacobian(
    tdoa_sets: list[tuple[np.ndarray, np.ndarray]], position_3d: np.ndarray, dimension: int,
) -> np.ndarray:
    """Jacobian of all TDoA residuals with respect to the first dimension coordinates of the position.

    The residual measured - (|p - a_i| - |p - a_ref|) has the gradient
    (p - a_ref) / |p - a_ref| - (p - a_i) / |p - a_i|, the difference of the unit
    vectors from the anchors to p.  It is NaN at an anchor, where it is undefined.
    """
    rows = []
    for anchor_positions, _time_differences in tdoa_sets:
        ref_delta = position_3d - anchor_positions[0]
        other_deltas = position_3d - anchor_positions[1:]
        # At an anchor, 0 / 0 gives NaN without a warning.
        with np.errstate(divide="ignore", invalid="ignore"):
            ref_direction = ref_delta / np.linalg.norm(ref_delta)
            other_directions = other_deltas / np.linalg.norm(other_deltas, axis=1)[:, np.newaxis]
        rows.append(ref_direction - other_directions)
    # In 2D, Z is fixed: only the columns of X and Y.
    return np.vstack(rows)[:, :dimension]


def _tdoa_geometry(
    tdoa_sets: list[tuple[np.ndarray, np.ndarray]], position_3d: np.ndarray, dimension: int,
) -> dict:
    """Rank, condition number, PDOP and position covariance (unit range variance) at a position.

    Singular geometries get cond = PDOP = inf and no covariance.
    """
    singular = {"rank": 0, "cond": math.inf, "PDOP": math.inf, "Q_pos": None}
    jacobian = _tdoa_jacobian(tdoa_sets, position_3d, dimension)
    if jacobian.size == 0 or not np.all(np.isfinite(jacobian)):
        return singular

    rank = int(np.linalg.matrix_rank(jacobian))
    cond = float(np.linalg.cond(jacobian))
    # The rank is also reported for singular geometries.
    singular["rank"] = rank
    # No pinv: it would report a finite PDOP for an unobservable direction.
    if rank < dimension:
        return singular
    try:
        covariance = np.linalg.inv(jacobian.T @ jacobian)
    except np.linalg.LinAlgError:
        return singular
    # Rounding can yield non-positive variances near the rank tolerance.
    if not np.all(np.isfinite(covariance)) or np.any(np.diag(covariance) <= 0):
        return singular
    # PDOP: square root of the summed coordinate variances.
    return {"rank": rank, "cond": cond, "PDOP": float(np.sqrt(np.trace(covariance))), "Q_pos": covariance}


def _huber_f_scale(residuals: np.ndarray, dimension: int) -> float:
    """Bounded Huber transition scale from the residual MAD of one event."""
    finite_residuals = np.asarray(residuals, dtype=float)
    finite_residuals = finite_residuals[np.isfinite(finite_residuals)]
    # A minimally determined system fits its measurements almost exactly.
    if finite_residuals.size < dimension + 2:
        return TDOA_HUBER_F_SCALE_FALLBACK_M

    residual_median = np.median(finite_residuals)
    robust_sigma = MAD_TO_SIGMA * np.median(np.abs(finite_residuals - residual_median))
    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        return TDOA_HUBER_F_SCALE_FALLBACK_M
    # 1.345 sigma: the usual Huber constant (95 % efficiency for normally distributed residuals).
    f_scale = TDOA_HUBER_SCALE_FACTOR * robust_sigma
    return float(np.clip(f_scale, TDOA_HUBER_F_SCALE_MIN_M, TDOA_HUBER_F_SCALE_MAX_M))


def solve_tdoa_least_squares(
    tdoa_sets: list[tuple[np.ndarray, np.ndarray]],
    *,
    localization_bounds_by_dimension: dict[int, tuple[np.ndarray, np.ndarray]],
    dimension_mode: str,
    loss: str,
    huber_f_scale_m: float | None,
) -> tuple[np.ndarray, dict]:
    """Estimate a position from TDoA measurements with bounded (robust) nonlinear least squares.

    Args:
        tdoa_sets: (anchor positions, time differences) per reference anchor; row 0
            of the positions is the reference, time difference i belongs to row i + 1.
        localization_bounds_by_dimension: Solver bounds for 2D and 3D (see
            localization_bounds_by_dimension).
        dimension_mode: "auto" (2D for planar anchors, else 3D), "2d" or "3d".
        loss: SciPy loss of the fit; a non-linear loss starts from a bounded
            linear fit, whose residual MAD also sets Huber's f_scale if
            huber_f_scale_m is None.

    Returns:
        The estimated 3D position and its diagnostics, named as the exported
        columns (residuals, PDOP, HDOP, VDOP, cond, tdoa_*, localization_bounds).
    """
    # The anchors of all sets decide the dimension and the start.
    all_anchors = np.vstack([anchors for anchors, _ in tdoa_sets]).astype(float)
    dimension, fixed_z = _tdoa_dimension(all_anchors, dimension_mode)
    lower_bounds, upper_bounds = localization_bounds_by_dimension[dimension]

    # Start at the anchor centroid, strictly inside the bounds (required by TRF);
    # in 2D, Z is fixed and only X/Y are optimized.
    initial_position_3d = np.mean(all_anchors, axis=0)
    if dimension == 2:
        initial_position_3d[2] = fixed_z
    bound_epsilon = 1e-9 * np.maximum(upper_bounds - lower_bounds, 1.0)
    parameter_guess = np.clip(
        initial_position_3d[:dimension], lower_bounds + bound_epsilon, upper_bounds - bound_epsilon
    )

    def expand_position(parameters: np.ndarray) -> np.ndarray:
        """3D position of the optimizer parameters."""
        if dimension == 3:
            return parameters
        return np.array([parameters[0], parameters[1], fixed_z])

    def residuals(parameters: np.ndarray) -> np.ndarray:
        """Measured minus modeled range differences of all TDoA sets."""
        position_3d = expand_position(parameters)
        all_residuals = []
        for anchor_positions, time_differences in tdoa_sets:
            range_differences = time_differences * SPEED_OF_LIGHT
            dist_to_ref = np.linalg.norm(position_3d - anchor_positions[0])
            dists_to_others = np.linalg.norm(anchor_positions[1:] - position_3d, axis=1)
            # Modeled: distance to the other anchor minus distance to the reference.
            all_residuals.append(range_differences - (dists_to_others - dist_to_ref))
        return np.concatenate(all_residuals)

    def jacobian(parameters: np.ndarray) -> np.ndarray:
        """Derivatives of the residuals with respect to the optimizer parameters."""
        return _tdoa_jacobian(tdoa_sets, expand_position(parameters), dimension)

    bounds = (lower_bounds, upper_bounds)
    f_scale = huber_f_scale_m
    start = parameter_guess
    if loss != "linear":
        # Huber's cost is flat for large residuals, so the robust fit starts
        # from the bounded linear solution instead of the centroid.
        linear_result = least_squares(residuals, parameter_guess, jac=jacobian, method="trf", bounds=bounds)
        if f_scale is None:
            f_scale = _huber_f_scale(linear_result.fun, dimension)
        start = np.clip(linear_result.x, lower_bounds + bound_epsilon, upper_bounds - bound_epsilon)
    elif f_scale is None:
        f_scale = 1.0  # Ignored by SciPy for loss='linear'.
    result = least_squares(
        residuals, start, jac=jacobian, method="trf", loss=loss, f_scale=f_scale, bounds=bounds
    )
    estimated_position_3d = expand_position(result.x)

    # Geometry from the geometric Jacobian, not result.jac, which SciPy scales for robust losses.
    initial_geometry = _tdoa_geometry(tdoa_sets, expand_position(parameter_guess), dimension)
    geometry = _tdoa_geometry(tdoa_sets, estimated_position_3d, dimension)
    covariance = geometry["Q_pos"]
    # In 2D, PDOP only covers X and Y, so it is the HDOP; there is no VDOP.
    if dimension == 2:
        hdop, vdop = geometry["PDOP"], None
    elif covariance is None:
        hdop, vdop = math.inf, math.inf
    else:
        # HDOP from the X and Y variances, VDOP from the Z variance.
        hdop = float(np.sqrt(covariance[0, 0] + covariance[1, 1]))
        vdop = float(np.sqrt(covariance[2, 2]))
    return estimated_position_3d, {
        "residuals": result.fun,
        "PDOP": geometry["PDOP"],
        "HDOP": hdop,
        "VDOP": vdop,
        "cond": geometry["cond"],
        "tdoa_dimension": dimension,
        "tdoa_fixed_z": fixed_z,
        "tdoa_loss": loss,
        "tdoa_f_scale": float(f_scale),
        "tdoa_initial_PDOP": initial_geometry["PDOP"],
        "tdoa_initial_cond": initial_geometry["cond"],
        "tdoa_initial_rank": initial_geometry["rank"],
        "tdoa_rank": geometry["rank"],
        "localization_bounds": (lower_bounds.tolist(), upper_bounds.tolist()),
    }
