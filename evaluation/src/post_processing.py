"""Localization events of the simulated runs, reconstructed from the raw traces.

Reads the OMNeT++ traces of one config and writes, per run, the TWR, v-TWR and
TDoA position estimates and their errors.
"""

import argparse
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import re
from itertools import product
from pathlib import Path
import math
from scipy.optimize import least_squares
from datetime import datetime


# GLOBAL CONSTANTS
POLL = 0
RESP = 1
FINAL = 2
SELF_LOC = 3
TWR = "TWR"
HYP_SELF_LOC = "HYPERBOLIC_SELF_LOCALIZATION"
CENTRALIZED_SELF_LOC = "CENTRALIZED"
DISTRIBUTED_SELF_LOC = "DISTRIBUTED"

SPEED_OF_LIGHT = 299792458.0
MIN_ANCHORS_FOR_TWR = 4
MIN_VALID_ANCHOR_PAIRS = 3
MIN_TIME_DIFFS_FOR_SELF_LOCALIZATION = 3
MIN_UNIQUE_ANCHORS_FOR_3D_TDOA = 4

# TDoA solver constants.  Profile, dimension, loss and f_scale are options
# (configs/defaults.yml).
TDOA_FIXED_Z = None  # In 2D, None uses the median participating-anchor height.
TDOA_PLANAR_REL_TOLERANCE = 1e-6
TDOA_OPTIMIZER_METHOD = "trf"

# Without a fixed f_scale (--tdoa_f_scale), it is estimated locally from a
# preliminary least-squares solution.  The fallback and limits keep that
# estimate usable when only a minimally determined system or a heavily
# corrupted event is available.
TDOA_HUBER_SCALE_FACTOR = 1.345
TDOA_HUBER_F_SCALE_FALLBACK_M = 1.0
TDOA_HUBER_F_SCALE_MIN_M = 0.05
TDOA_HUBER_F_SCALE_MAX_M = 10.0

# Start the robust (non-linear-loss) fit from the bounded linear least-squares
# solution instead of from the prior/centroid guess.  Huber's cost is flat for
# large residuals, so from a poor start it can settle in a wrong minimum.
TDOA_HUBER_START_FROM_LINEAR = True

# Bounds are derived from physical (not virtual) anchors.  Z deliberately gets
# a larger margin because a future 3D deployment can extend above/below them.
LOCALIZATION_XY_BOUND_MARGIN_RATIO = 0.30
LOCALIZATION_Z_BOUND_MARGIN_RATIO = 0.50
LOCALIZATION_MIN_BOUND_REFERENCE_SPAN_M = 1.0

CLOCK = "clock_timestamp"
SIM = "sim_timestamp"
NOISY = "noisy_timestamp"


def _build_event_lookup(df: pd.DataFrame) -> dict[tuple[int, int, int, int], pd.Series |None]:
    lookup: dict[tuple[int, int, int, int], dict | None] = {}
    duplicate_keys: set[tuple[int, int, int, int]] = set()

    for row in df.to_dict("records"):
        event_key = (
            int(row["NodeId"]),
            int(row["srcId"]),
            int(row["seqNo"]),
            int(row["interfaceIdx"]),
        )

        if event_key in lookup:
            duplicate_keys.add(event_key)
            lookup[event_key] = None
        elif event_key not in duplicate_keys:
            lookup[event_key] = row

    if duplicate_keys:
        print(f"{len(duplicate_keys)=}")

    return lookup

def _get_position_from_record(row: dict | pd.Series) -> np.ndarray:
    """ Extract a 3D position vector from a row-like object.

    Args:
        row (dict | pd.Series): Row containing X, Y, and Z coordinate fields.

    Returns:
        np.ndarray: Position vector with x, y, and z coordinates.
    """
    return np.array([row["X"], row["Y"], row["Z"]], dtype=float)

def _resolve_tdoa_dimension(
    all_anchor_positions: np.ndarray,
    dimension_mode: str,
    fixed_z: float | None,
) -> tuple[int, float | None]:
    """Resolve manual/automatic 2D or 3D solving and the fixed 2D height."""
    normalized_mode = dimension_mode.strip().lower()
    if normalized_mode not in {"auto", "2d", "3d"}:
        raise ValueError("dimension_mode must be 'auto', '2d', or '3d'")

    spans = np.ptp(all_anchor_positions, axis=0)
    reference_span = max(float(np.max(spans[:2])), 1.0)

    # In this project a planar deployment means the target is constrained to
    # the same plane.  The relative tolerance also tolerates small real-world
    # coordinate noise instead of requiring all Z values to be exactly equal.
    anchors_are_planar = (
        float(spans[2]) <= TDOA_PLANAR_REL_TOLERANCE * reference_span
    )
    use_2d = normalized_mode == "2d" or (
        normalized_mode == "auto" and anchors_are_planar
    )

    if not use_2d:
        return 3, None

    resolved_fixed_z = (
        float(np.median(all_anchor_positions[:, 2]))
        if fixed_z is None
        else float(fixed_z)
    )
    return 2, resolved_fixed_z


def _build_localization_bounds(
    physical_anchor_positions: np.ndarray,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build finite solver bounds from the physical-anchor bounding box."""
    anchor_min = np.min(physical_anchor_positions, axis=0)
    anchor_max = np.max(physical_anchor_positions, axis=0)
    reference_span = max(
        # The requested safety margin is tied to the largest horizontal
        # deployment extent; vertical anchor spread does not enlarge XY bounds.
        float(np.max((anchor_max - anchor_min)[:2])),
        LOCALIZATION_MIN_BOUND_REFERENCE_SPAN_M,
    )

    # Use one XY margin derived from the largest deployment extent so that a
    # narrow or even zero span on one axis does not also produce a zero margin.
    xy_margin = LOCALIZATION_XY_BOUND_MARGIN_RATIO * reference_span
    lower = anchor_min[:2] - xy_margin
    upper = anchor_max[:2] + xy_margin

    if dimension == 3:
        z_margin = LOCALIZATION_Z_BOUND_MARGIN_RATIO * reference_span
        lower = np.append(lower, anchor_min[2] - z_margin)
        upper = np.append(upper, anchor_max[2] + z_margin)

    return lower.astype(float), upper.astype(float)


def _build_localization_bounds_by_dimension(
    physical_anchor_positions: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Precompute the immutable 2D and 3D bounds once for one run."""
    return {
        dimension: _build_localization_bounds(
            physical_anchor_positions, dimension
        )
        for dimension in (2, 3)
    }


def _tdoa_jacobian(
    tdoa_sets: list[tuple[np.ndarray, np.ndarray]],
    position_3d: np.ndarray,
    dimension: int,
) -> np.ndarray:
    """Return the unweighted geometric Jacobian of all TDoA residuals."""
    rows = []
    for anchor_positions, _time_differences in tdoa_sets:
        ref_delta = position_3d - anchor_positions[0]
        ref_distance = np.linalg.norm(ref_delta)
        for other_anchor in anchor_positions[1:]:
            other_delta = position_3d - other_anchor
            other_distance = np.linalg.norm(other_delta)

            # A source exactly on an anchor has an undefined distance gradient.
            # NaNs make the corresponding geometry fail safely below.
            if ref_distance == 0 or other_distance == 0:
                rows.append(np.full(dimension, np.nan))
                continue

            # residual = measured_difference - (distance_other - distance_ref)
            row_3d = ref_delta / ref_distance - other_delta / other_distance
            rows.append(row_3d[:dimension])

    return np.asarray(rows, dtype=float)


def _tdoa_geometry_metrics(
    tdoa_sets: list[tuple[np.ndarray, np.ndarray]],
    position_3d: np.ndarray,
    dimension: int,
    sigma_r: float,
) -> dict:
    """Compute rank, condition number and DOP without hiding singularity."""
    jacobian = _tdoa_jacobian(tdoa_sets, position_3d, dimension)
    if jacobian.size == 0 or not np.all(np.isfinite(jacobian)):
        return {"rank": 0, "cond": math.inf, "PDOP": math.inf, "Q_pos": None}

    rank = int(np.linalg.matrix_rank(jacobian))
    cond = float(np.linalg.cond(jacobian))
    if rank < dimension:
        # Do not use pinv here: it would put zero variance into an unobservable
        # direction and could therefore report a deceptively finite PDOP.
        return {"rank": rank, "cond": math.inf, "PDOP": math.inf, "Q_pos": None}

    information = jacobian.T @ jacobian
    try:
        covariance = np.linalg.inv(information) * sigma_r**2
    except np.linalg.LinAlgError:
        # Numerical singularity can still occur near the rank tolerance.
        return {"rank": rank, "cond": math.inf, "PDOP": math.inf, "Q_pos": None}

    # A nearly singular information matrix can pass matrix_rank and inv() but
    # still yield negative variances from rounding.  The square roots below
    # would then return NaN, which the statistics filters treat as "missing"
    # and therefore keep.  Report such geometries as singular instead.
    if (
        not np.all(np.isfinite(covariance))
        or np.any(np.diag(covariance) <= 0)
    ):
        return {"rank": rank, "cond": math.inf, "PDOP": math.inf, "Q_pos": None}
    pdop = float(np.sqrt(np.trace(covariance)))
    return {"rank": rank, "cond": cond, "PDOP": pdop, "Q_pos": covariance}


def _estimate_local_huber_f_scale(
    residuals: np.ndarray,
    dimension: int,
) -> float:
    """Estimate a bounded Huber transition scale from one event's residuals."""
    finite_residuals = np.asarray(residuals, dtype=float)
    finite_residuals = finite_residuals[np.isfinite(finite_residuals)]

    # A minimally determined system can fit its own measurements almost
    # exactly and contains too little redundancy for a meaningful scale.  Use
    # the explicit meter-valued fallback in that case.
    if finite_residuals.size < dimension + 2:
        return TDOA_HUBER_F_SCALE_FALLBACK_M

    residual_median = np.median(finite_residuals)
    mad = np.median(np.abs(finite_residuals - residual_median))
    robust_sigma = 1.4826 * mad
    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        return TDOA_HUBER_F_SCALE_FALLBACK_M

    f_scale = TDOA_HUBER_SCALE_FACTOR * robust_sigma
    return float(
        np.clip(
            f_scale,
            TDOA_HUBER_F_SCALE_MIN_M,
            TDOA_HUBER_F_SCALE_MAX_M,
        )
    )


def _get_robust_dop_info(
    tdoa_sets: list[tuple[np.ndarray, np.ndarray]],
    estimated_position_3d: np.ndarray,
    dimension: int,
    sigma_r: float,
) -> dict:
    """Return geometry diagnostics for a bounded TRF estimate."""
    geometry = _tdoa_geometry_metrics(
        tdoa_sets, estimated_position_3d, dimension, sigma_r
    )
    covariance = geometry["Q_pos"]
    hdop = geometry["PDOP"] if dimension == 2 else (
        math.inf
        if covariance is None
        else float(np.sqrt(covariance[0, 0] + covariance[1, 1]))
    )
    vdop = None if dimension == 2 else (
        math.inf
        if covariance is None
        else float(np.sqrt(covariance[2, 2]))
    )
    return {
        "PDOP": geometry["PDOP"],
        "HDOP": hdop,
        "VDOP": vdop,
        "Q_pos": covariance,
        "cond": geometry["cond"],
        "rank": geometry["rank"],
    }


def solve_tdoa_least_squares(
    tdoa_sets: list[tuple[np.ndarray, np.ndarray]],
    speed_of_signal: float = 299792458.0,
    initial_guess: np.ndarray | None = None,
    sigma_r: float = 1.0,  # optional standard deviation of the range measurement in meters
    compute_dop: bool = True,
    max_nfev: int | None = None,
    *,
    solver_profile: str,
    dimension_mode: str,
    fixed_z: float | None = TDOA_FIXED_Z,
    localization_bounds_source_positions: np.ndarray | None = None,
    localization_bounds_by_dimension: dict[
        int, tuple[np.ndarray, np.ndarray]
    ] | None = None,
    optimizer_method: str = TDOA_OPTIMIZER_METHOD,
    loss: str,
    huber_f_scale_m: float | None,
    huber_start_from_linear: bool = TDOA_HUBER_START_FROM_LINEAR,
) -> tuple[np.ndarray, dict]:
    """Estimate a bounded 2D/3D position with robust nonlinear least squares.

    Args:
        tdoa_sets (list[tuple[np.ndarray, np.ndarray]]): List of tuples containing anchor positions and time differences.
        speed_of_signal (float): Propagation speed used to convert time differences into range differences.
        initial_guess (np.ndarray | None): Optional initial position estimate for the optimizer.
        sigma_r (float): Standard deviation of the range measurement in meters for DOP estimation.
        compute_dop (bool): Whether to compute PDOP, HDOP, VDOP, covariance, and condition number.
        max_nfev (int | None): Optional maximum number of function evaluations for the optimizer.
        solver_profile (str): ``robust`` or ``legacy``. Legacy reproduces the previous LM solve.
        dimension_mode (str): ``auto``, ``2d``, or ``3d``. Auto uses 2D for planar anchors.
        fixed_z (float | None): Fixed target height in 2D; default is median anchor Z.
        localization_bounds_source_positions (np.ndarray | None): Physical anchors used to derive spatial bounds for isolated solver calls.
        localization_bounds_by_dimension (dict | None): Immutable run-level localization bounds keyed by dimension.
        optimizer_method (str): SciPy least_squares method; robust losses require ``trf``/``dogbox``.
        loss (str): SciPy loss function, e.g. ``huber``.
        huber_f_scale_m (float | None): Explicit robust scale in metres; None estimates it per event.
        huber_start_from_linear (bool): Start a robust-loss fit from the bounded linear solution.

    Returns:
        tuple[np.ndarray, dict]: Estimated position and diagnostic information from the optimizer.
    """

    normalized_profile = solver_profile.strip().lower()
    if normalized_profile not in {"robust", "legacy"}:
        raise ValueError("solver_profile must be 'robust' or 'legacy'")

    if normalized_profile == "legacy":
        # One switch restores the numerical behavior before the solver update:
        # unconstrained 3D Levenberg-Marquardt with ordinary least squares.
        # Fine-grained dimension/loss options are intentionally overridden.
        dimension_mode = "3d"
        optimizer_method = "lm"
        loss = "linear"

    all_anchors = np.vstack([anchors for anchors, _ in tdoa_sets]).astype(float)
    dimension, resolved_fixed_z = _resolve_tdoa_dimension(
        all_anchors, dimension_mode, fixed_z
    )

    # If no prior is available, the anchor centroid is a transparent fallback.
    # The 2D solver keeps Z fixed and optimizes only the observable X/Y values.
    if initial_guess is None:
        initial_position_3d = np.mean(all_anchors, axis=0)
    else:
        initial_position_3d = np.asarray(initial_guess, dtype=float).copy()
    if initial_position_3d.shape != (3,):
        raise ValueError("initial_guess must contain X, Y, and Z")
    if dimension == 2:
        initial_position_3d[2] = resolved_fixed_z

    if localization_bounds_by_dimension is not None:
        # Normal post-processing enters here: physical anchors and both bound
        # variants were validated and calculated once during run setup.
        try:
            precomputed_lower, precomputed_upper = localization_bounds_by_dimension[
                dimension
            ]
        except KeyError as error:
            raise ValueError(
                f"No localization bounds available for {dimension}D"
            ) from error
        lower_bounds = np.asarray(precomputed_lower, dtype=float).copy()
        upper_bounds = np.asarray(precomputed_upper, dtype=float).copy()
    else:
        # Retain a convenient fallback for isolated calls to this low-level
        # solver.  The post-processing path never relies on event anchors.
        bounds_source = (
            all_anchors
            if localization_bounds_source_positions is None
            else np.asarray(localization_bounds_source_positions, dtype=float)
        )
        if bounds_source.ndim != 2 or bounds_source.shape[1] != 3:
            raise ValueError(
                "localization_bounds_source_positions must have shape (n, 3)"
            )
        lower_bounds, upper_bounds = _build_localization_bounds(
            bounds_source, dimension
        )

    expected_bounds_shape = (dimension,)
    if (
        lower_bounds.shape != expected_bounds_shape
        or upper_bounds.shape != expected_bounds_shape
        or not np.all(np.isfinite(lower_bounds))
        or not np.all(np.isfinite(upper_bounds))
        or np.any(lower_bounds >= upper_bounds)
    ):
        raise ValueError(f"Invalid localization bounds for {dimension}D")

    parameter_guess = initial_position_3d[:dimension]
    if normalized_profile != "legacy":
        # TRF requires a feasible starting point.  Keep it just inside bounds
        # to avoid starting exactly on a reflective boundary.  Legacy LM keeps
        # the original unbounded centroid/explicit initial guess unchanged.
        bound_epsilon = 1e-9 * np.maximum(upper_bounds - lower_bounds, 1.0)
        parameter_guess = np.clip(
            parameter_guess,
            lower_bounds + bound_epsilon,
            upper_bounds - bound_epsilon,
        )

    def expand_position(parameters: np.ndarray) -> np.ndarray:
        if dimension == 3:
            return parameters
        return np.array([parameters[0], parameters[1], resolved_fixed_z])

    def residuals(parameters):
        """ Compute stacked TDoA residuals for the current optimizer position.

        Args:
            parameters (np.ndarray): Current 2D/3D optimizer parameters.

        Returns:
            np.ndarray: Concatenated residual vector over all TDoA anchor sets.
        """
        position_3d = expand_position(parameters)
        all_residuals = []

        for anchor_positions, time_differences in tdoa_sets:

            range_differences = time_differences * speed_of_signal

            ref_anchor = anchor_positions[0]
            other_anchors = anchor_positions[1:]

            dist_to_ref = np.linalg.norm(position_3d - ref_anchor)

            dists_to_others = np.linalg.norm(other_anchors - position_3d, axis=1)

            calculated_diff = dists_to_others - dist_to_ref

            res = range_differences - calculated_diff

            all_residuals.append(res)

        return np.concatenate(all_residuals)

    if optimizer_method == "lm" and loss != "linear":
        raise ValueError("SciPy's lm method only supports loss='linear'")

    initial_geometry = _tdoa_geometry_metrics(
        tdoa_sets, expand_position(parameter_guess), dimension, sigma_r
    )

    resolved_f_scale = huber_f_scale_m
    linear_result = None
    if loss != "linear" and (
        resolved_f_scale is None
        or huber_start_from_linear
    ):
        # The preliminary fit is performed independently for each localization,
        # never globally across speeds/scenarios.  Its residual MAD only sets
        # Huber's scale; the final stored position comes from the robust fit.
        linear_result = least_squares(
            residuals,
            parameter_guess,
            method="trf",
            loss="linear",
            bounds=(lower_bounds, upper_bounds),
            max_nfev=max_nfev,
        )
        if resolved_f_scale is None:
            resolved_f_scale = _estimate_local_huber_f_scale(
                linear_result.fun, dimension
            )
    elif resolved_f_scale is None:
        resolved_f_scale = 1.0  # Ignored by SciPy for loss='linear'.

    if not np.isfinite(resolved_f_scale) or resolved_f_scale <= 0:
        raise ValueError("huber_f_scale_m must be finite and greater than zero")

    solver_bounds = (
        (-np.inf, np.inf)
        if optimizer_method == "lm"
        else (lower_bounds, upper_bounds)
    )
    effective_lower_bounds = (
        np.full(dimension, -np.inf)
        if optimizer_method == "lm"
        else lower_bounds
    )
    effective_upper_bounds = (
        np.full(dimension, np.inf)
        if optimizer_method == "lm"
        else upper_bounds
    )
    robust_start = parameter_guess
    if huber_start_from_linear and linear_result is not None:
        # TRF needs a strictly feasible start; the linear solution may lie on
        # a bound.
        robust_start = np.clip(
            linear_result.x,
            lower_bounds + bound_epsilon,
            upper_bounds - bound_epsilon,
        )
    result = least_squares(
        residuals,
        robust_start,
        method=optimizer_method,
        loss=loss,
        f_scale=resolved_f_scale,
        bounds=solver_bounds,
        max_nfev=max_nfev,
    )
    estimated_position_3d = expand_position(result.x)

    info = {
        "success": result.success,
        "cost": float(result.cost),
        "iterations": result.nfev,
        "residuals": result.fun,
        "solver_profile": normalized_profile,
        "optimizer_method": optimizer_method,
        "dimension": dimension,
        "fixed_z": resolved_fixed_z,
        "loss": loss,
        "f_scale": float(resolved_f_scale),
        "bounds_lower": effective_lower_bounds,
        "bounds_upper": effective_upper_bounds,
        "initial_PDOP": initial_geometry["PDOP"],
        "initial_cond": initial_geometry["cond"],
        "initial_rank": initial_geometry["rank"],
    }

    if compute_dop:
        if normalized_profile == "legacy":
            # Preserve the old diagnostics as well as the old optimizer.  The
            # pseudoinverse can hide an unobservable direction, and cond(JTJ)
            # squares cond(J), but both are required for historical comparison.
            jacobian = result.jac
            information = jacobian.T @ jacobian
            covariance = np.linalg.pinv(information) * sigma_r**2
            pdop = float(np.sqrt(np.trace(covariance)))
            info.update(
                {
                    "PDOP": pdop,
                    "HDOP": float(np.sqrt(covariance[0, 0] + covariance[1, 1])),
                    "VDOP": float(np.sqrt(covariance[2, 2])),
                    "Q_pos": covariance,
                    "cond": float(np.linalg.cond(information)),
                    "rank": int(np.linalg.matrix_rank(jacobian)),
                }
            )
        else:
            # Recompute the unmodified geometric Jacobian.  SciPy scales
            # result.jac for robust losses, which is useful for optimization
            # but is not a pure geometry/DOP diagnostic.
            info.update(
                _get_robust_dop_info(
                    tdoa_sets,
                    estimated_position_3d,
                    dimension,
                    sigma_r,
                )
            )
    else:
        info.update({"PDOP": None, "HDOP": None, "VDOP": None, "Q_pos": None, "cond": None, "rank": None})

    return estimated_position_3d, info

def _localize_nodes_vtwr(df: pd.DataFrame, event_lookup: dict, mobile_node_ids: list,  timestamp_format: str):
    # Compute virtual TWR statistics per mobile node
    for node_id in mobile_node_ids:
        print(f"Localize v-TWR Position for Node {node_id} @ {datetime.now()}")

        # Get all entries where the node transmits.
        node_transmissions = _get_node_transmissions(df, node_id)
        
        # Get all messages that were received by the node
        node_receptions = _get_node_reception_data(df, node_id) # pd.Series

        # Iterate from last to first entry
        for _, v_final_tx_record in node_transmissions.iloc[::-1].iterrows(): 
            # Get the sequence number of the current event/message
            v_final_seq_no = int(v_final_tx_record["seqNo"]) 

            # Get the interface Id over which the message was received
            v_final_interface_id = int(v_final_tx_record["interfaceIdx"])

            # Get all the transmissions by this node that were done since the last transmission on the same interface or later
            v_poll_tx_candidates = _get_candidates_since_last_round_from_transmissions(
                node_transmissions,
                v_final_seq_no,
                v_final_interface_id,
            )

            # If empty, there is no suitable poll candidate with which we could try to perform v-TWR
            if v_poll_tx_candidates.empty:
                continue

            # Get "oldest" entry in v_poll_tx
            v_poll_tx = v_poll_tx_candidates.iloc[0]
            
            # Check what messages have been received between the v-Final and v-Poll
            v_poll_tx_idx = int(v_poll_tx["event_idx"]) # Lookup dictionary row index.
            v_final_tx_idx = int(v_final_tx_record["event_idx"])
            node_rx_between_poll_and_final = node_receptions.loc[v_poll_tx_idx:v_final_tx_idx]

            # Only anchors are used for localization; What anchors were overheard in the last round?
            anchor_ids = node_rx_between_poll_and_final.loc[node_rx_between_poll_and_final["src_is_anchor"] == True, "srcId"].unique()

            # FINAL is fixed for all of them the same
            anchor_resp_rx_dict = {} 
            anchor_resp_tx_dict = {}
            anchor_poll_rx_dict = {}
            anchor_poll_tx_dict = {}
            anchor_final_rx_dict = {}

            # Contains the youngest responses received by the node
            # Get all the response messages from all the anchors
            for anchor_id in anchor_ids:

                # Check whether anchor received final message
                anchor_final_rx_record = _get_transmission_for_receiver(
                    event_lookup, 
                    node_id, 
                    anchor_id, 
                    v_final_seq_no,
                    v_final_interface_id
                )
                if anchor_final_rx_record is None: 
                    continue

                anchor_final_rx_dict[int(anchor_id)] = anchor_final_rx_record

                # Filter all messages received from a certain anchor
                anchor_rx_records_at_node = node_rx_between_poll_and_final.loc[node_rx_between_poll_and_final["srcId"]==anchor_id]
                
                # Get the youngest RESP message that was received within the last round
                youngest_resp_rx_entry = anchor_rx_records_at_node.iloc[-1]
                assert(not youngest_resp_rx_entry.empty)

                # Save the current anchor id with the rx entry of the youngest RESP msg at the current node
                anchor_resp_rx_dict[int(anchor_id)] = youngest_resp_rx_entry

                # Save the tx from the current anchor 
                anchor_resp_tx_record = _get_transmission_for_receiver(
                    event_lookup, 
                    anchor_id, 
                    anchor_id, 
                    youngest_resp_rx_entry["seqNo"],
                    youngest_resp_rx_entry["interfaceIdx"]
                )

                if anchor_resp_tx_record is None:
                    continue        

                anchor_resp_tx_dict[int(anchor_id)] = anchor_resp_tx_record

                # Check which v_poll_tx_entry is the first one to precede the resp entry
                # Find youngest poll transmission that was send before the youngest response was received
                for _, v_poll_tx_entry in v_poll_tx_candidates.iloc[::-1].iterrows(): # From youngest to oldest
                    # The resp timestamp must be younger than the poll timestamp
                    if v_poll_tx_entry[timestamp_format] > youngest_resp_rx_entry[timestamp_format]:
                        # Continue with the next one
                        continue
                    # Check whether this was received by the anchor
                    youngest_v_poll_anchor_rx = _get_transmission_for_receiver(
                        event_lookup, 
                        node_id, 
                        anchor_id,
                        v_poll_tx_entry["seqNo"],
                        v_poll_tx_entry["interfaceIdx"]
                    )
                    if youngest_v_poll_anchor_rx is None:
                        # The anchor seems to not have received this virtual poll message
                        continue

                    # If it was received, we have a valid entry; Poll was received by anchor and resp was received by Node
                    anchor_poll_rx_dict[int(anchor_id)] = youngest_v_poll_anchor_rx
                    anchor_poll_tx_dict[int(anchor_id)] = v_poll_tx_entry
                    # If we have found something, we do not need to continue (we used the the youngest entry anyways)
                    break
                        # If the youngest entry does not match, we have to go on with the next oldest

            confirmed_anchor_ids = sorted(
                set(anchor_poll_rx_dict)
                & set(anchor_poll_tx_dict)
                & set(anchor_resp_rx_dict)
                & set(anchor_resp_tx_dict)
                & set(anchor_final_rx_dict)
            )

            # Check whether I have sufficient anchors for TWR
            if (len(confirmed_anchor_ids)< MIN_ANCHORS_FOR_TWR):
                continue

            # Get the correct timestamp for the poll and final message
            _, anchor_positions, tofs = _compute_vtwr_measurements(
                                        node_id, 
                                        confirmed_anchor_ids, 
                                        anchor_resp_rx_dict,
                                        anchor_resp_tx_dict,
                                        anchor_poll_rx_dict,
                                        anchor_poll_tx_dict,
                                        anchor_final_rx_dict,
                                        v_final_tx=v_final_tx_record, # current_msg record is equivalent to the virtual final message
                                        timestamp_format=timestamp_format
                                        )
            
            node_true_position = _get_position_from_record(v_final_tx_record)
            estimated_pos = _trilaterate_position_from_tof(anchor_positions, tofs)

            pos_error = math.dist(node_true_position, estimated_pos)

            df.at[v_final_tx_idx, "used_anchors_or_pairs"] = confirmed_anchor_ids
            df.at[v_final_tx_idx, "v_twr_x"] = estimated_pos[0]
            df.at[v_final_tx_idx, "v_twr_y"] = estimated_pos[1]
            df.at[v_final_tx_idx, "v_twr_z"] = estimated_pos[2]
            df.at[v_final_tx_idx, "v_twr_pos"] = estimated_pos.tolist()
            df.at[v_final_tx_idx, "v_twr_error"] = pos_error


def _compute_vtwr_measurements(
    node_id: int,
    confirmed_anchor_ids: list[int],
    anchor_resp_rx_dict: dict[int, pd.Series],
    anchor_resp_tx_dict: dict[int, pd.Series],
    anchor_poll_rx_dict: dict[int, pd.Series],
    anchor_poll_tx_dict: dict[int, pd.Series],
    anchor_final_rx_dict: dict[int, pd.Series],
    v_final_tx: pd.Series,
    timestamp_format: str,
) -> tuple[list, list, list]:
    """Times of flight between node_id and its confirmed anchors for one v-TWR exchange.

    Returns the anchor IDs, the anchor positions and the times of flight.
    """
    v_final_tx_ts = float(v_final_tx[timestamp_format])

    anchor_ids_for_tof = []
    anchor_positions = []
    tofs = []

    for anchor_id in confirmed_anchor_ids:
        v_resp_rx = anchor_resp_rx_dict[anchor_id]
        v_resp_tx = anchor_resp_tx_dict[anchor_id]
        v_poll_rx = anchor_poll_rx_dict[anchor_id]
        v_poll_tx = anchor_poll_tx_dict[anchor_id]
        v_final_rx = anchor_final_rx_dict[anchor_id]

        assert anchor_id != node_id

        v_poll_tx_ts = float(v_poll_tx[timestamp_format])
        v_response_rx_ts = float(v_resp_rx[timestamp_format])
        v_response_tx_ts = float(v_resp_tx[timestamp_format])
        v_poll_rx_ts = float(v_poll_rx[timestamp_format])
        v_final_rx_ts = float(v_final_rx[timestamp_format])

        tof = calc_twr_tof(
            R_a=v_response_rx_ts - v_poll_tx_ts,
            R_b=v_final_rx_ts - v_response_tx_ts,
            D_a=v_final_tx_ts - v_response_rx_ts,
            D_b=v_response_tx_ts - v_poll_rx_ts,
        )
        # Position when receiving the final message
        anchor_ids_for_tof.append(anchor_id)
        anchor_positions.append(_get_position_from_record(v_final_rx))
        tofs.append(float(tof))

    return anchor_ids_for_tof, anchor_positions, tofs


def _get_node_transmissions(df: pd.DataFrame, node_id: int) -> pd.DataFrame:
    """Get all records of transmissions of a specified

    Args:
        df (pd.DataFrame): Overall dataframe with all transmission entries of an individual run
        node_id (int): The node Id for which the transmissions are searched

    Returns:
        pd.DataFrame: All transmission records of the specified node
    """
    return df.loc[(df["srcId"] == node_id) & (df["NodeId"] == node_id)]

def _build_tdoa_info(
    anchor_ids: list,
    time_diffs: list,
    anchor_positions: list,
    current_index: int,
    sender_id: int,
) -> dict:
    """Map each (sender, anchor) pair to its TDoA metadata."""
    sender_position = anchor_positions[0]
    return {
        (int(sender_id), int(anchor_id)): [current_index, time_diff, sender_position, anchor_pos]
        for anchor_id, time_diff, anchor_pos in zip(anchor_ids, time_diffs, anchor_positions[1:])
    }

def _collect_latest_stitched_pairs(between_df: pd.DataFrame) -> list[tuple]:
    """
    Walk rows in *between_df* and return the most recent TDoA entry per unique anchor pair.

    Returns a list of (anchor_pair, tdoa_info) tuples.

    Note: pairs (a, b) and (b, a) are treated as identical; the orientation
    of whichever entry was kept is preserved in the returned tuple.
    """
    latest: dict[tuple, tuple] = {}  # sorted_pair -> (original_pair, tdoa_info)

    for row in between_df.itertuples(index=False):
        if row.tdoa_info is None:
            continue

        for anchor_pair, tdoa_info in row.tdoa_info.items():
            sorted_pair = tuple(sorted(anchor_pair))
            candidate_index = tdoa_info[0]

            if sorted_pair not in latest:
                latest[sorted_pair] = (anchor_pair, tdoa_info)
                continue

            existing_index = latest[sorted_pair][1][0]
            if candidate_index > existing_index:
                latest[sorted_pair] = (anchor_pair, tdoa_info)

    return list(latest.values())

def _build_stitched_tdoa_groups(
    latest_pairs: list[tuple],
) -> tuple[list, list]:
    """
    Group TDoA pairs by reference anchor and build the arrays expected by
    *solve_tdoa_least_squares*.
 
    Returns:
        solver_inputs:     list of (positions_array, time_diffs_array), one per
                           reference anchor, ready for solve_tdoa_least_squares.
        used_anchor_pairs: flat list of all (sender, anchor) pairs consumed.
    """
    # Named indices for the fields inside each pair_info entry.
    # pair_info layout (from _build_tdoa_info): [event_index, time_diff, ref_pos, other_pos]
    EVENT_INDEX, TIME_DIFF, REF_POS, OTHER_POS = range(4)
 
    ref_anchor_groups: dict[int, list] = {}
    used_anchor_pairs: list = []
 
    for anchor_pair, pair_info in latest_pairs:
        used_anchor_pairs.append(anchor_pair)
        ref_anchor = anchor_pair[0]
        # setdefault avoids an explicit "if ref_anchor not in dict" check:
        # it inserts an empty list on first access and returns the list either way.
        ref_anchor_groups.setdefault(ref_anchor, []).append(pair_info)
 
    solver_inputs = []
    for ref_anchor_pairs in ref_anchor_groups.values():
        # Sort newest-first so that entries_sorted[0] is the most recent entry.
        entries_sorted = sorted(ref_anchor_pairs, key=lambda e: e[EVENT_INDEX], reverse=True)
        # Use the ref-anchor position from the most recent entry only —
        # its position estimate should be the most accurate one available.
        ref_pos_latest = entries_sorted[0][REF_POS]
 
        positions = [ref_pos_latest]
        time_diffs = []
        for entry in entries_sorted:
            positions.append(entry[OTHER_POS])
            time_diffs.append(entry[TIME_DIFF])
 
        solver_inputs.append((np.asarray(positions), np.asarray(time_diffs)))
 
    return solver_inputs, used_anchor_pairs

def _store_localization_result(
    run_df: pd.DataFrame,
    current_index: int,
    estimated_pos: tuple,
    true_pos: tuple,
    info: dict,
    used_anchors_or_pairs: list,
    *,
    using_anchor_stitching: bool = False,
) -> None:
    """Write a localization result and its diagnostics into *run_df*."""
    run_df.at[current_index, "used_anchors_or_pairs"] = used_anchors_or_pairs
    run_df.at[current_index, "self_loc_x"] = estimated_pos[0]
    run_df.at[current_index, "self_loc_y"] = estimated_pos[1]
    run_df.at[current_index, "self_loc_z"] = estimated_pos[2]
    run_df.at[current_index, "self_loc_error"] = math.dist(true_pos, estimated_pos)
    run_df.at[current_index, "residuals"] = info["residuals"]
    run_df.at[current_index, "PDOP"] = info["PDOP"]
    run_df.at[current_index, "HDOP"] = info["HDOP"]
    run_df.at[current_index, "VDOP"] = info["VDOP"]
    run_df.at[current_index, "cond"] = info["cond"]
    run_df.at[current_index, "tdoa_solver_profile"] = info["solver_profile"]
    run_df.at[current_index, "tdoa_optimizer_method"] = info["optimizer_method"]
    run_df.at[current_index, "tdoa_dimension"] = info["dimension"]
    run_df.at[current_index, "tdoa_fixed_z"] = info["fixed_z"]
    run_df.at[current_index, "tdoa_loss"] = info["loss"]
    run_df.at[current_index, "tdoa_f_scale"] = info["f_scale"]
    run_df.at[current_index, "tdoa_initial_PDOP"] = info["initial_PDOP"]
    run_df.at[current_index, "tdoa_initial_cond"] = info["initial_cond"]
    run_df.at[current_index, "tdoa_initial_rank"] = info["initial_rank"]
    run_df.at[current_index, "tdoa_rank"] = info["rank"]
    run_df.at[current_index, "localization_bounds"] = (
        info["bounds_lower"].tolist(),
        info["bounds_upper"].tolist(),
    )

    if using_anchor_stitching:
        run_df.at[current_index, "using_anchor_stitching"] = True


def _compute_hyperbola_for_processed_message(self_loc_RTX_info: dict[int, dict[str, pd.Series]], refAnchor_current_TX: pd.Series, timestamp_format: str, use_v_twr_pos: bool):

    # Position of the reference Anchor, transmitting the prev and current message
    refAnchor_position = (  float(refAnchor_current_TX["X"]),
                            float(refAnchor_current_TX["Y"]),
                            float(refAnchor_current_TX["Z"]))
    # if v-Anchor, we require v-TWR position of the transmitting anchor  
    if (refAnchor_current_TX["src_is_anchor"] == False and use_v_twr_pos):
        refAnchor_position = refAnchor_current_TX["v_twr_pos"]

    anchor_positions = [refAnchor_position]
    time_differences = []
    for peerAnchor_id, RTX_records in self_loc_RTX_info.items():
        
        # Unpack
        # NOTE: I do not require all entries as I only use one anchor as reference (namely the peer Anchor - for some reason; I guess I followed the formula from the original paper)
        peerAnchor_prev_RX      = RTX_records["peerAnchor_prev_RX"]
        peerAnchor_mid_TX       = RTX_records["peerAnchor_mid_TX"]
        peerAnchor_current_RX   = RTX_records["peerAnchor_current_RX"]

        node_prev_RX    = RTX_records["node_prev_RX"]
        node_mid_RX     = RTX_records["node_mid_RX"]
        node_current_RX = RTX_records["node_current_RX"]

        # Position of the peer Anchor transmitting the mid message
        peerAnchor_position = ( float(peerAnchor_mid_TX["X"]),
                                float(peerAnchor_mid_TX["Y"]),
                                float(peerAnchor_mid_TX["Z"]))
        if (peerAnchor_mid_TX["src_is_anchor"] == False and use_v_twr_pos):
            peerAnchor_position = peerAnchor_mid_TX["v_twr_pos"]

        dist_ab = math.dist(refAnchor_position, peerAnchor_position)
        t_ab = dist_ab / SPEED_OF_LIGHT

        delta_t = _calc_self_loc_delta_t(
            s_rx1_a=node_prev_RX[timestamp_format],
            s_rx1_b=node_mid_RX[timestamp_format],
            s_rx2_a=node_current_RX[timestamp_format],
            b_rx1_a=peerAnchor_prev_RX[timestamp_format],
            b_tx1=peerAnchor_mid_TX[timestamp_format],
            b_rx2_a=peerAnchor_current_RX[timestamp_format],
            t_ab=t_ab,
        )
        anchor_positions.append(peerAnchor_position)
        time_differences.append(delta_t)
    anchor_positions = np.array(anchor_positions)
    time_differences = np.array(time_differences)
    return anchor_positions, time_differences


def _calc_self_loc_delta_t(s_rx1_a: float, 
                           s_rx1_b: float, 
                           s_rx2_a: float,
                           b_rx1_a: float, 
                           b_tx1: float, 
                           b_rx2_a: float, 
                           t_ab: float) -> float:
    """ Compute the self-localization time difference from sender, anchor, and receiver timestamps.

    Args:
        s_rx1_a (float): Receiver timestamp for the previous sender message.
        s_rx1_b (float): Receiver timestamp for the anchor message.
        s_rx2_a (float): Receiver timestamp for the current sender message.
        b_rx1_a (float): Anchor timestamp for the previous sender message.
        b_tx1 (float): Anchor transmit timestamp.
        b_rx2_a (float): Anchor timestamp for the current sender message.
        t_ab (float): Propagation time between sender and anchor.

    Returns:
        float: Estimated time difference for hyperbolic localization.
    """
    R_ba = b_rx2_a - b_tx1
    D_ba = b_tx1 - b_rx1_a
    S_ab = s_rx1_b - s_rx1_a
    S_ba = s_rx2_a - s_rx1_b
    numerator = (R_ba * S_ab) - (S_ba * D_ba) - (t_ab * R_ba) - (t_ab * S_ab) + (t_ab**2)
    denominator = R_ba + D_ba - t_ab
    delta_t = numerator / denominator
    return delta_t

def _get_transmission_for_receiver(df_lookup: dict | pd.DataFrame, sender_id: int, receiver_id: int, seq_no: int, interface_id: int ) -> dict | pd.Series | None:
    """ Retrieve one transmission/reception row for a sender-receiver-sequence-interfaceIdx tuple.

    Args:
        df_lookup (dict | pd.DataFrame): Precomputed dictionary lookup or indexed dataframe.
        sender_id (int): ID of the transmitting node.
        receiver_id (int): ID of the receiving node.
        seq_no (int): Sequence number of the message.
        interfaceIdx (int): interfaceIdx over which the message was transmitted

    Returns:
        dict | pd.Series | None: Matching row or None if no unique row exists.
    """
    key = (int(receiver_id), int(sender_id), int(seq_no), int(interface_id))

    if isinstance(df_lookup, dict):
        return df_lookup.get(key)

    try:
        row = df_lookup.loc[key]
        if isinstance(row, pd.DataFrame):
            return row.iloc[0] if len(row) == 1 else None
        return row
    except KeyError:
        return None


def _get_node_reception_data(df: pd.DataFrame, node_id: int)->pd.DataFrame:
    """ Get the reception data from all nodes that the queried node receives

    Args:
        df (pd.DataFrame): Input dataframe to be filtered
        node_id (int): Query ID for which the reception data is going to be extracted

    Returns:
        pd.DataFrame: filtered dataframe containing reception data
    """
    mask = (
        (df["NodeId"] == node_id) &
        (df["srcId"] != node_id)  # Only received, no transmitted messages
    )
    return df[mask]


def _localize_nodes_twr(run_df: pd.DataFrame, mobile_node_ids: list[int] , timestamp_format: str):
    # Go through all all nodes
    for (node_id, seq_no), df_subset_raw in run_df.groupby(["NodeId", "seqNo"]):
        # Only look at mobile nodes
        if node_id not in mobile_node_ids: # Only mobile nodes
            continue
        
        # Get all entries where the transmitter is either the current node or an anchor 
        df_subset = df_subset_raw[(df_subset_raw["src_is_anchor"]) | (df_subset_raw["srcId"] == node_id)]

        # Get all the transmitter for the current sequence number, I guess
        src_list = df_subset.loc[df_subset["msgType"] == RESP, "srcId"].tolist()

        # Only anchors are considered that have sent a RESP message, thus they must have overheard the POLL
        poll_rx_df, resp_tx_df, final_rx_df = _get_twr_anchor_timestamps(run_df, src_list, node_id, seq_no)
        confirmed_anchor_ids = final_rx_df["NodeId"].tolist()

        # Must be the node, as it is the only node in the subset that transmits a FINAL message
        final_tx_row = df_subset.loc[
            df_subset["msgType"] == FINAL
        ]

        # There is either one single message available or none at all (right in the end)
        assert(len(final_tx_row)==1 or len(final_tx_row) == 0)
        if len(final_tx_row) != 1:
            continue

        # We require at least MIN_ANCHORS_FOR_TWR, which consists of the confirmed anchors that have received the final.
        num_confirmed_responses = len(confirmed_anchor_ids)
        if (num_confirmed_responses < MIN_ANCHORS_FOR_TWR):
            continue
        
        estimated_pos, pos_error = _calc_twr_position(
                                        df_subset, 
                                        node_id, 
                                        poll_rx_df, 
                                        resp_tx_df, 
                                        final_rx_df, 
                                        timestamp_format=timestamp_format
                                        )
        estimated_x = estimated_pos[0]
        estimated_y = estimated_pos[1]
        estimated_z = estimated_pos[2]
        run_df.at[final_tx_row.index.item(), "twr_x"] = estimated_x
        run_df.at[final_tx_row.index.item(), "twr_y"] = estimated_y
        run_df.at[final_tx_row.index.item(), "twr_z"] = estimated_z
        run_df.at[final_tx_row.index.item(), "twr_error"] = pos_error
        run_df.at[final_tx_row.index.item(), "used_anchors_or_pairs"] = confirmed_anchor_ids

    return run_df


def _build_node_views(run_df: pd.DataFrame) -> tuple[
    dict[int, pd.DataFrame],
    dict[int, pd.DataFrame],
    dict[int, pd.DataFrame],
]:
    # Get per node all the records that were recorded by the given node (Rx and Tx)
    records_by_node = {
        int(node_id): node_df
        for node_id, node_df in run_df.groupby("NodeId", sort=False)
    }

    # From all the records by a node, get all those entries, where the node is only the receiver
    receptions_by_node = {
        node_id: node_df.loc[node_df["srcId"] != node_id].copy()
        for node_id, node_df in records_by_node.items()
    }

    # From all the records by a node, get all those entries, where the node is the tranmsitter
    self_transmissions_by_node = {
        node_id: node_df.loc[node_df["srcId"] == node_id]
        for node_id, node_df in records_by_node.items()
    }

    return records_by_node, receptions_by_node, self_transmissions_by_node

def _get_physical_anchor_positions(run_df: pd.DataFrame) -> np.ndarray:
    """Return one representative true position for every physical anchor."""
    required_columns = {"node_is_anchor", "NodeId", "X", "Y", "Z"}
    missing_columns = required_columns.difference(run_df.columns)
    if missing_columns:
        missing_columns_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            "Cannot initialize this run: missing physical-anchor columns: "
            f"{missing_columns_text}"
        )

    physical_anchor_rows = run_df.loc[
        run_df["node_is_anchor"], ["NodeId", "X", "Y", "Z"]
    ]
    if physical_anchor_rows.empty:
        raise ValueError(
            "Cannot initialize this run: no rows with node_is_anchor == True "
            "were found"
        )

    # Physical anchors are expected to be fixed.  The median is nevertheless
    # robust against duplicate records and small coordinate logging noise.
    physical_anchor_positions = (
        physical_anchor_rows.groupby("NodeId")[["X", "Y", "Z"]]
        .median()
        .to_numpy(dtype=float)
    )
    if (
        physical_anchor_positions.shape[0] == 0
        or not np.all(np.isfinite(physical_anchor_positions))
    ):
        raise ValueError(
            "Cannot initialize this run: physical anchor positions contain "
            "missing or non-finite coordinates"
        )
    return physical_anchor_positions


def _localize_nodes_self_loc(
    run_df: pd.DataFrame,
    mobile_node_ids: list[int],
    timestamp_format: str,
    event_lookup: dict[tuple[int, int, int], pd.Series | None],
    use_v_twr_pos: bool,
    *,
    localization_bounds_by_dimension: dict[int, tuple[np.ndarray, np.ndarray]],
    tdoa_solver_profile: str,
    tdoa_dimension_mode: str,
    tdoa_loss: str,
    tdoa_f_scale_m: float | None,
) -> pd.DataFrame:

    # A TDoA based localization consists always of 3 messages: 
        # 2 Send by Reference Anchor: Current self-loc msg & previous self-loc message 
        # 1 send by other anchor: "Mid" self-loc message

    # Naming: NODE that holds record _ msgType _ RX/TX _ Additional Information/description

    # Build dicts with {nodeId: RTX_df} to save computation time
    records_by_node, receptions_by_node, self_transmissions_by_node = _build_node_views(run_df)

    # iterate through all nodes
    for node_id in mobile_node_ids:
        print(f"Localize Self-Localization Position for Node {node_id} @ {datetime.now()}")
        # Get all messages, that were received by the current mobile_node 
        node_reception_df = receptions_by_node.get(node_id)


        # Iterate through all messages that were received by this node
        for current_index, node_current_RX in node_reception_df.iterrows(): 
            # We got the current receive record from which we can get some information
            refAnchor_id        = int(node_current_RX["srcId"])
            current_interface   = int(node_current_RX["interfaceIdx"])
            current_seq_no      = int(node_current_RX["seqNo"])

            # Get CANDIDATES for previous transmissions
                # Either on same interface with a reduced sequence number OR
                # on same sequence number but with a smaller interface
            refAnchor_transmissions = self_transmissions_by_node.get(refAnchor_id)
            if refAnchor_transmissions is None or refAnchor_transmissions.empty:
                continue

            refAnchor_prev_TX_candidates = _get_candidates_since_last_round_from_transmissions(
                refAnchor_transmissions,
                current_seq_no,
                current_interface,
            )
            if refAnchor_prev_TX_candidates.empty:
                continue

            # Get the records for all the previous candidates transmitted by the current reference anchor that were actually received by the current node
            node_prev_RX_confirmed_candidates = _get_confirmed_previous_rx_candidates(event_lookup, refAnchor_id, node_id, refAnchor_prev_TX_candidates)
            if node_prev_RX_confirmed_candidates.empty:
                continue

            # Check which other messages were received by the node between the current message and the oldest previous candidate (maximum span)
            ## Get the oldest valid prev candidate
            node_prev_RX_oldest_candidate = node_prev_RX_confirmed_candidates.iloc[0] 
            node_between_prev_and_current_RX = _get_records_between_prev_and_current(node_reception_df, node_prev_RX_oldest_candidate, node_current_RX)


            # Filter out those messages where the srcId != sender_id
            node_mid_between_prev_and_current_RX = node_between_prev_and_current_RX.loc[node_between_prev_and_current_RX["srcId"] != refAnchor_id]

            # If we have not received anything else between the two messages, we continue
            if node_mid_between_prev_and_current_RX.empty:
                continue

            # Get the between DF of the reference Anchor
            refAnchor_df = records_by_node.get(refAnchor_id)
            if refAnchor_df is None or refAnchor_df.empty:
                continue

            oldest_prev_seqNo       = node_prev_RX_oldest_candidate["seqNo"]
            oldest_prev_interfaceId = node_prev_RX_oldest_candidate["interfaceIdx"]
            refAnchor_current_TX   = _get_transmission_for_receiver(event_lookup, refAnchor_id, refAnchor_id, current_seq_no, current_interface)
            refAnchor_oldest_prev_TX= _get_transmission_for_receiver(event_lookup, refAnchor_id, refAnchor_id, oldest_prev_seqNo, oldest_prev_interfaceId)
            
            if refAnchor_current_TX is None or refAnchor_oldest_prev_TX is None:
                continue
            
            refAnchor_between_prev_and_current_RTX = _get_records_between_prev_and_current(refAnchor_df, refAnchor_oldest_prev_TX, refAnchor_current_TX)

            # The current message must have a location if it is a v-Anchor
            if not _has_required_self_loc_position(refAnchor_current_TX, use_v_twr_pos):
                continue
            
            # Did the reference anchor receive the messages by the other anchors
            refAnchor_mid_RX_confirmed = _get_refAnchor_confirmed_mid_rx_records(event_lookup, refAnchor_id, node_mid_between_prev_and_current_RX)
            
            # If the reference anchor did not receive any of the other anchors transmissions, we do not need to continue with this message
            if refAnchor_mid_RX_confirmed.empty:
                continue
            
            # Get all information required from the peer anchor
            self_loc_RTX_info = _build_self_loc_peer_anchor_info(
                event_lookup,
                records_by_node,
                refAnchor_mid_RX_confirmed,
                refAnchor_id,
                node_id,
                current_seq_no,
                current_interface,
                node_current_RX,
                node_prev_RX_confirmed_candidates,
                node_between_prev_and_current_RX,
                refAnchor_between_prev_and_current_RTX,
                refAnchor_current_TX,
            )

            if not self_loc_RTX_info:
                continue

            anchor_positions, time_differences, node_true_position, valid_anchor_pairs = (
                _prepare_self_loc_solver_inputs(
                    run_df=run_df,
                    node_reception_df=node_reception_df,
                    current_index=current_index,
                    node_current_RX=node_current_RX,
                    refAnchor_id=refAnchor_id,
                    self_loc_RTX_info=self_loc_RTX_info,
                    refAnchor_current_TX=refAnchor_current_TX,
                    timestamp_format=timestamp_format,
                    use_v_twr_pos=use_v_twr_pos,
                )
            )

            # Perform anchor stitching, when this round alone lacks enough anchor pairs.
            if len(valid_anchor_pairs) < MIN_VALID_ANCHOR_PAIRS:
                _try_self_loc_anchor_stitching(
                    run_df=run_df,
                    current_index=current_index,
                    node_true_position=node_true_position,
                    node_between_prev_and_current_RX=node_between_prev_and_current_RX,
                    localization_bounds_by_dimension=localization_bounds_by_dimension,
                    solver_profile=tdoa_solver_profile,
                    dimension_mode=tdoa_dimension_mode,
                    loss=tdoa_loss,
                    huber_f_scale_m=tdoa_f_scale_m,
                )
                continue

            # If fewer than three time differences are available, 3D hyperbolic localization is not possible.
            _try_direct_self_loc_solver(
                run_df=run_df,
                current_index=current_index,
                anchor_positions=anchor_positions,
                time_differences=time_differences,
                node_true_position=node_true_position,
                valid_anchor_pairs=valid_anchor_pairs,
                localization_bounds_by_dimension=localization_bounds_by_dimension,
                solver_profile=tdoa_solver_profile,
                dimension_mode=tdoa_dimension_mode,
                loss=tdoa_loss,
                huber_f_scale_m=tdoa_f_scale_m,
            )
            continue
    
    return run_df


def _has_minimum_unique_anchors_for_3d_tdoa(
    anchor_pairs: list[tuple[int, int]],
) -> bool:
    """Return whether the pairs contain enough distinct anchors for 3D TDoA."""
    unique_anchor_ids = {
        anchor_id
        for anchor_pair in anchor_pairs
        for anchor_id in anchor_pair
    }
    return len(unique_anchor_ids) >= MIN_UNIQUE_ANCHORS_FOR_3D_TDOA


def _try_direct_self_loc_solver(
    run_df: pd.DataFrame,
    current_index: int,
    anchor_positions: np.ndarray,
    time_differences: np.ndarray,
    node_true_position: np.ndarray,
    valid_anchor_pairs: list[tuple[int, int]],
    localization_bounds_by_dimension: dict[int, tuple[np.ndarray, np.ndarray]],
    solver_profile: str,
    dimension_mode: str,
    loss: str,
    huber_f_scale_m: float | None,
) -> bool:
    """
    Try to estimate and store a self-localization result from the current round only.

    Returns:
        bool:
            True if a localization result was computed and stored.
            False if not enough time differences were available.
    """

    # If fewer than three time differences are available, 3D hyperbolic localization is not possible.
    if len(time_differences) < MIN_TIME_DIFFS_FOR_SELF_LOCALIZATION:
        return False

    # Three time differences must also involve at least four distinct anchors in 3D.
    if not _has_minimum_unique_anchors_for_3d_tdoa(valid_anchor_pairs):
        return False

    # Estimate the position from time differences and anchor positions.
    estimated_pos, info = solve_tdoa_least_squares(
        [(anchor_positions, time_differences)],
        localization_bounds_by_dimension=localization_bounds_by_dimension,
        solver_profile=solver_profile,
        dimension_mode=dimension_mode,
        loss=loss,
        huber_f_scale_m=huber_f_scale_m,
    )

    _store_localization_result(
        run_df,
        current_index,
        estimated_pos,
        node_true_position,
        info,
        valid_anchor_pairs,
    )

    return True

def _try_self_loc_anchor_stitching(
    run_df: pd.DataFrame,
    current_index: int,
    node_true_position: np.ndarray,
    node_between_prev_and_current_RX: pd.DataFrame,
    localization_bounds_by_dimension: dict[int, tuple[np.ndarray, np.ndarray]],
    solver_profile: str,
    dimension_mode: str,
    loss: str,
    huber_f_scale_m: float | None,
) -> bool:
    """
    Try to estimate and store a self-localization result using anchor stitching.

    Returns:
        bool:
            True if anchor stitching produced and stored a localization result.
            False if not enough stitched anchor pairs were available.
    """

    # Try to get the position with anchor pairs of previous rounds.
    latest_anchor_pairs = _collect_latest_stitched_pairs(
        node_between_prev_and_current_RX
    )

    # Three pairs must also involve at least four distinct anchors in 3D.
    if len(latest_anchor_pairs) < MIN_TIME_DIFFS_FOR_SELF_LOCALIZATION:
        return False

    stitched_pairs = [
        anchor_pair for anchor_pair, _ in latest_anchor_pairs
    ]
    if not _has_minimum_unique_anchors_for_3d_tdoa(stitched_pairs):
        return False

    solver_inputs, used_anchor_pairs = _build_stitched_tdoa_groups(
        latest_anchor_pairs
    )

    estimated_pos, info = solve_tdoa_least_squares(
        solver_inputs,
        localization_bounds_by_dimension=localization_bounds_by_dimension,
        solver_profile=solver_profile,
        dimension_mode=dimension_mode,
        loss=loss,
        huber_f_scale_m=huber_f_scale_m,
    )

    # Put the calculated data into the dataframe.
    _store_localization_result(
        run_df,
        current_index,
        estimated_pos,
        node_true_position,
        info,
        used_anchor_pairs,
        using_anchor_stitching=True,
    )

    return True

def _prepare_self_loc_solver_inputs(
    run_df: pd.DataFrame,
    node_reception_df: pd.DataFrame, 
    current_index: int,
    node_current_RX: pd.Series,
    refAnchor_id: int,
    self_loc_RTX_info: dict[int, dict[str, pd.Series]],
    refAnchor_current_TX: pd.Series,
    timestamp_format: str,
    use_v_twr_pos: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """
    Compute the hyperbola inputs and store TDoA information for possible anchor stitching.

    Returns:
        tuple:
            anchor_positions:
                Anchor or virtual-anchor positions used by the TDoA solver.
            time_differences:
                TDoA values used by the TDoA solver.
            node_true_position:
                True node position at the current received message.
            valid_anchor_pairs:
                Anchor pairs available from the current self-localization round.
    """

    # Compute Hyperbola:
    anchor_positions, time_differences = _compute_hyperbola_for_processed_message(
        self_loc_RTX_info,
        refAnchor_current_TX,
        timestamp_format,
        use_v_twr_pos,
    )

    # Get the true position.
    node_true_position = _get_position_from_record(node_current_RX)

    # Get all peer anchor ids.
    peerAnchor_ids = self_loc_RTX_info.keys()

    # At least three additional anchors are required because the sender is the reference anchor.
    valid_anchor_pairs = [
        (int(refAnchor_id), int(anchor_id))
        for anchor_id in peerAnchor_ids
    ]

    # Store reference-anchor and other-anchor positions for possible anchor stitching.
    tdoa_info = _build_tdoa_info(
        peerAnchor_ids,
        time_differences,
        anchor_positions,
        current_index,
        refAnchor_id,
    )

    # Store in the local node reception dataframe because anchor stitching reads from it.
    node_reception_df.at[current_index, "tdoa_info"] = tdoa_info

    run_df.at[current_index, "tdoa_info"] = tdoa_info

    return (
        anchor_positions,
        time_differences,
        node_true_position,
        valid_anchor_pairs,
    )
            
def _build_self_loc_peer_anchor_info(
    event_lookup: dict,
    records_by_node: dict[int, pd.DataFrame],
    refAnchor_mid_RX_confirmed: pd.DataFrame,
    refAnchor_id: int,
    node_id: int,
    current_seq_no: int,
    current_interface: int,
    node_current_RX: pd.Series,
    node_prev_RX_confirmed_candidates: pd.DataFrame,
    node_between_prev_and_current_RX: pd.DataFrame,
    refAnchor_between_prev_and_current_RTX: pd.DataFrame,
    refAnchor_current_TX: pd.Series,
) -> dict[int, dict]:
    """
    Build the TX/RX records required for hyperbolic self-localization.

    Returns:
        dict[int, dict]:
            Mapping from peerAnchorId to all TX/RX records required for one
            hyperbola.
    """
    self_loc_RTX_info = {}

    # Did the other anchor receive the current message and any of the previous message candidates
    refAnchor_mid_RX_confirmed_peerAnchor_ids = set(refAnchor_mid_RX_confirmed["srcId"])

    # I need a list of messages that were received by the reference anchor and the node
    for peerAnchor_id in refAnchor_mid_RX_confirmed_peerAnchor_ids:
        # Check whether the other anchors have received the current message
        peerAnchor_current_RX = _get_transmission_for_receiver(event_lookup, refAnchor_id, peerAnchor_id, current_seq_no, current_interface)
        # If the other did not even receive the current message, we do not need to continue any further 
        if peerAnchor_current_RX is None: 
            continue

        # Get the dataframe of prev candidates that were sent by the reference anchor and actually received by the peer anchor (handled with the input)
        peerAnchor_prev_RX_confirmed_candidates =_get_confirmed_previous_rx_candidates(event_lookup, refAnchor_id, peerAnchor_id, node_prev_RX_confirmed_candidates)
        # If the other anchor has not received any previous message, we cannot use it further
        if peerAnchor_prev_RX_confirmed_candidates.empty: 
            continue

        peerAnchor_df = records_by_node.get(int(peerAnchor_id))
        if peerAnchor_df is None or peerAnchor_df.empty:
            continue

        ## Get the oldest valid prev candidate
        peerAnchor_prev_RX_oldest_candidate = peerAnchor_prev_RX_confirmed_candidates.iloc[0] 
        peerAnchor_between_prev_and_current_RTX = _get_records_between_prev_and_current(peerAnchor_df, peerAnchor_prev_RX_oldest_candidate, peerAnchor_current_RX)
        
        # Get all common messages at all three nodes (peer Anchor, reference Anchor and current node)
        selected_messages = _select_common_self_loc_messages(
            node_between_prev_and_current_RX,
            refAnchor_between_prev_and_current_RTX,
            peerAnchor_between_prev_and_current_RTX,
            refAnchor_id,
            peerAnchor_id,
        )

        if selected_messages is None:
            continue

        prev_seq_no = selected_messages["prev_seq_no"]
        prev_interface = selected_messages["prev_interface"]

        mid_seq_no = selected_messages["mid_seq_no"]
        mid_interface = selected_messages["mid_interface"]

        # Get the TDoA/Hyperbola with this one Node 
        self_loc_RTX_info[peerAnchor_id] = {
            # NODE
            "node_prev_RX": _get_transmission_for_receiver(event_lookup, refAnchor_id, node_id, prev_seq_no, prev_interface),
            "node_mid_RX": _get_transmission_for_receiver(event_lookup, peerAnchor_id, node_id, mid_seq_no, mid_interface),
            "node_current_RX": node_current_RX,
            # PEER
            "peerAnchor_prev_RX": _get_transmission_for_receiver(event_lookup, refAnchor_id, peerAnchor_id, prev_seq_no, prev_interface),
            "peerAnchor_mid_TX": _get_transmission_for_receiver(event_lookup, peerAnchor_id, peerAnchor_id, mid_seq_no, mid_interface),
            "peerAnchor_current_RX": peerAnchor_current_RX,
            # RefAnchor
            "refAnchor_prev_TX": _get_transmission_for_receiver(event_lookup, refAnchor_id, refAnchor_id, prev_seq_no, prev_interface),
            "refAnchor_mid_RX": _get_transmission_for_receiver(event_lookup, peerAnchor_id, refAnchor_id, mid_seq_no, mid_interface),
            "refAnchor_current_TX": refAnchor_current_TX,
        }
    
    return self_loc_RTX_info



def _select_common_self_loc_messages(
    node_between_prev_and_current_RX: pd.DataFrame,
    refAnchor_between_prev_and_current_RTX: pd.DataFrame,
    peerAnchor_between_prev_and_current_RTX: pd.DataFrame,
    refAnchor_id: int,
    peerAnchor_id: int,
) -> dict | None:
    """
    Select the previous and mid message that were received by
    node, reference anchor and peer anchor.

    Returns:
        dict | None:
            {
                "prev_seq_no": ...,
                "prev_interface": ...,
                "mid_seq_no": ...,
                "mid_interface": ...,
            }
    """

    message_columns = ["srcId", "seqNo", "interfaceIdx"]

    # Find messages received by all three nodes.
    message_columns = ["srcId", "seqNo", "interfaceIdx"]
    required_columns = [
        "srcId",
        "seqNo",
        "interfaceIdx",
        "event_idx",
        "src_is_anchor",
        "v_twr_pos",
    ]

    # Find messages received by all three nodes.
    common_messages = (
        node_between_prev_and_current_RX[required_columns]
        .merge(
            refAnchor_between_prev_and_current_RTX[required_columns],
            on=message_columns,
            how="inner",
        )
        .merge(
            peerAnchor_between_prev_and_current_RTX[required_columns],
            on=message_columns,
            how="inner",
        )
    )

    # Common previous/current reference-anchor messages.
    common_ref_messages = common_messages.loc[
        common_messages["srcId"] == refAnchor_id
    ]

    if common_ref_messages.empty:
        return None

    all_prev = common_ref_messages.iloc[:-1]

    # Common peer-anchor mid messages.
    common_peer_messages = common_messages.loc[
        (common_messages["srcId"] == peerAnchor_id)
        & ~(
            (common_messages["src_is_anchor"] == False)
            & (common_messages["v_twr_pos"].isna())
        )
    ]

    if common_peer_messages.empty:
        return None

    # Get youngest common mid message.
    youngest_mid = common_peer_messages.iloc[-1]

    # Get youngest previous reference message before the selected mid message.
    prev_before_mid = all_prev.loc[
        all_prev.index < youngest_mid.name
    ]

    if prev_before_mid.empty:
        return None

    youngest_prev = prev_before_mid.iloc[-1]

    return {
        "prev_seq_no": youngest_prev["seqNo"],
        "prev_interface": youngest_prev["interfaceIdx"],
        "mid_seq_no": youngest_mid["seqNo"],
        "mid_interface": youngest_mid["interfaceIdx"],
    }

def _get_refAnchor_confirmed_mid_rx_records(
    event_lookup: dict,
    refAnchor_id: int,
    other_anchor_rx_between_previous_and_current: pd.DataFrame,
) -> pd.DataFrame:
    """Return records from other anchors that were also received by the reference anchor."""

    refAnchor_mid_RX_confirmed = []

    for other_anchor_rx_record in other_anchor_rx_between_previous_and_current.itertuples(index=False):
        peerAnchor_id = int(other_anchor_rx_record.srcId)
        other_anchor_seq_no = int(other_anchor_rx_record.seqNo)
        other_anchor_interface = int(other_anchor_rx_record.interfaceIdx)

        # Did the reference anchor receive these messages? 
        reference_anchor_rx_record = _get_transmission_for_receiver(
            event_lookup,
            peerAnchor_id,
            refAnchor_id,
            other_anchor_seq_no,
            other_anchor_interface,
        )
        
        # If the reference anchor did not overhear the message, we will not consider it further
        if reference_anchor_rx_record is None:
            continue

        refAnchor_mid_RX_confirmed.append(reference_anchor_rx_record)

    return pd.DataFrame(refAnchor_mid_RX_confirmed)

            
def _get_records_between_prev_and_current(node_reception_df: pd.DataFrame, previous_reference_record: pd.Series, current_reference_record: pd.Series) -> pd.DataFrame:

    # Get the required indices
    oldest_previous_idx = int(previous_reference_record["event_idx"])
    current_idx = int(current_reference_record["event_idx"])

    # Get all the messages that were received between these two indices
    between_prev_and_current_records = node_reception_df.loc[oldest_previous_idx:current_idx]

    return between_prev_and_current_records
    
def _has_required_self_loc_position(
    record: pd.Series,
    use_v_twr_pos: bool,
) -> bool:
    """Return whether a real or virtual anchor has a usable position for self-localization."""
    if bool(record["src_is_anchor"]):
        return True

    v_twr_position = record["v_twr_pos"]

    if v_twr_position is None:
        return False

    return True


def _get_confirmed_previous_rx_candidates(
    event_lookup: dict,
    sender_id: int,
    node_id: int,
    previous_candidate_records: pd.DataFrame,
) -> pd.DataFrame:
        # Get all the messages that the current reference anchor has transmitted and that were also received by the node
    confirmed_records = []

    # Go through all messages that are candidates for the previous self-loc message and check, whether the current node has received these messages
    for candidate in previous_candidate_records.itertuples(index=False):
        confirmed_record = _get_transmission_for_receiver(
            event_lookup,
            sender_id,
            node_id,
            candidate.seqNo,
            candidate.interfaceIdx,
        )

        if confirmed_record is not None:
            confirmed_records.append(confirmed_record)

    # All the prev candidates that were also received by the node
    return pd.DataFrame(confirmed_records)


def _get_candidates_since_last_round_from_transmissions(
    transmissions_df: pd.DataFrame,
    current_seq_no: int,
    current_interface: int,
) -> pd.DataFrame:
    """ Gets all the candidates for our localization since the last round.
    Candidates are only transmissions from a certain node (from the view of the node), 
    which is why we use the transmission df as an input.    


    Args:
        transmissions_df (pd.DataFrame): DataFrame containing ONLY transmission records from a certain node
        current_seq_no (int): The sequence number of the current round 
        current_interface (int): The interface id of the current record for which the localization is triggered

    Returns:
        pd.DataFrame: A dataframe with all the (tx) records that might be eligable for this round
    """
    return transmissions_df.loc[
        (
            (transmissions_df["seqNo"] == current_seq_no)
            & (transmissions_df["interfaceIdx"] < current_interface)
        )
        |
        (
            (transmissions_df["seqNo"] == current_seq_no - 1)
            & (transmissions_df["interfaceIdx"] >= current_interface)
        )
    ]


def _init_results_columns(
    run_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add required columns to the dataframe that are later filled
    """
    run_df["event_idx"] = run_df.index
    run_df["twr_x"] = None
    run_df["twr_y"] = None
    run_df["twr_z"] = None
    run_df["twr_error"] = None
    run_df["v_twr_x"] = None
    run_df["v_twr_y"] = None
    run_df["v_twr_z"] = None
    run_df["v_twr_pos"] = None
    run_df["v_twr_error"] = None
    run_df["self_loc_x"] = None
    run_df["self_loc_y"] = None
    run_df["self_loc_z"] = None
    run_df["self_loc_error"] = None
    run_df["est_position_x"] = None
    run_df["est_position_y"] = None
    run_df["est_position_z"] = None
    run_df["est_position_error"] = None
    run_df["localization_method"] = None
    run_df["residuals"] = None
    run_df["residual_mad_score"]= None 
    run_df["using_anchor_stitching"] = False
    run_df["used_anchors_or_pairs"] = None
    run_df["PDOP"] = None
    run_df["HDOP"] = None
    run_df["VDOP"] = None
    run_df["cond"] = None
    run_df["tdoa_solver_profile"] = None
    run_df["tdoa_optimizer_method"] = None
    run_df["tdoa_dimension"] = None
    run_df["tdoa_fixed_z"] = None
    run_df["tdoa_loss"] = None
    run_df["tdoa_f_scale"] = None
    run_df["tdoa_initial_PDOP"] = None
    run_df["tdoa_initial_cond"] = None
    run_df["tdoa_initial_rank"] = None
    run_df["tdoa_rank"] = None
    run_df["localization_bounds"] = None
    run_df["tdoa_info"] = None
    return run_df



def _calc_twr_position(confirmed_responses_df: pd.DataFrame, node_id: int, poll_rx_df: pd.DataFrame, resp_tx_df: pd.DataFrame, final_rx_df: pd.DataFrame, timestamp_format: str):
    """ Estimate a node position from confirmed TWR anchor responses.

    Args:
        confirmed_responses_df (pd.DataFrame): Message subset containing confirmed responses for one node and sequence.
        node_id (int): ID of the mobile node.
        poll_rx_df (pd.DataFrame): Poll reception timestamps at anchors.
        resp_tx_df (pd.DataFrame): Response transmission timestamps at anchors.
        final_rx_df (pd.DataFrame): Final message reception timestamps at anchors.
        timestamp_format (str): Timestamp column used for timing calculations.

    Returns:
        tuple[np.ndarray, float]: Estimated position and position error.
    """
    confirmed_anchor_list = final_rx_df["NodeId"].unique().tolist()

    poll_tx_ts = confirmed_responses_df.loc[confirmed_responses_df["msgType"] == POLL, timestamp_format].item()
    final_tx_row = confirmed_responses_df.loc[confirmed_responses_df["msgType"] == FINAL].iloc[0]
    final_tx_ts = final_tx_row[timestamp_format]

    poll_rx_by_anchor = poll_rx_df.set_index("NodeId", drop=False)
    resp_tx_by_anchor = resp_tx_df.set_index("NodeId", drop=False)
    final_rx_by_anchor = final_rx_df.set_index("NodeId", drop=False)
    resp_rx_by_anchor = confirmed_responses_df.loc[
        (confirmed_responses_df["NodeId"] == node_id) &
        (confirmed_responses_df["srcId"].isin(confirmed_anchor_list))
    ].set_index("srcId", drop=False)

    ToFs = []
    anchor_positions = []
    for anchor_id in confirmed_anchor_list:
        poll_rx_ts = poll_rx_by_anchor.at[anchor_id, timestamp_format]
        resp_tx_ts = resp_tx_by_anchor.at[anchor_id, timestamp_format]
        resp_rx_ts = resp_rx_by_anchor.at[anchor_id, timestamp_format]
        final_rx_ts = final_rx_by_anchor.at[anchor_id, timestamp_format]

        anchor_position = _get_position_from_record(final_rx_by_anchor.loc[anchor_id])
        anchor_positions.append(anchor_position)
        tof = calc_twr_tof(
            R_a=(resp_rx_ts - poll_tx_ts),
            R_b=(final_rx_ts - resp_tx_ts),
            D_a=(final_tx_ts - resp_rx_ts),
            D_b=(resp_tx_ts - poll_rx_ts),
        )
        ToFs.append(tof)

    node_position = _get_position_from_record(final_tx_row)
    estimated_pos = _trilaterate_position_from_tof(anchor_positions, ToFs)
    pos_error = math.dist(node_position, estimated_pos)

    return estimated_pos, pos_error

def filter_df_for_localization_method(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """ Take dataframe and extract only the required parts of it for further processing

    Args:
        df (pd.DataFrame): DataFrame to be reduced
        method (str): Name of the Method

    Returns:
        pd.DataFrame: Reduced DataFrame
    """
    df_filtered = df[df["Method"]==method].copy().reset_index(drop=True)
    return df_filtered

def _get_twr_anchor_timestamps(df: pd.DataFrame, anchor_id_list: list, node_id: int, seq_no: int):
    """ Extract anchor timestamps required for TWR time-of-flight calculation.

    Args:
        df (pd.DataFrame): Input dataframe containing all events.
        anchor_id_list (list): Anchor IDs that responded to the poll.
        node_id (int): ID of the mobile node.
        seq_no (int): Sequence number of the TWR round.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Poll reception, response transmission, and final reception dataframes.
    """

    # for anchor_id in anchor_id_list:
    poll_rx_df = df[(df["NodeId"].isin(anchor_id_list)) &
                        (df["srcId"]   == node_id) &
                        (df["msgType"] == POLL) &
                        (df["seqNo"]   == seq_no)
                        ] 
    resp_tx_df = df[(df["NodeId"].isin(anchor_id_list)) &
                        (df["srcId"]   == df["NodeId"]) &
                        (df["msgType"] == RESP) &
                        (df["seqNo"]   == seq_no) &
                        (df["destId"]  == node_id)
                        ] 

    final_rx_df = df[(df["NodeId"].isin(anchor_id_list)) &
                        (df["srcId"]   == node_id) &
                        (df["msgType"] == FINAL) &
                        (df["seqNo"]   == seq_no)
                        ] 

    return poll_rx_df, resp_tx_df, final_rx_df


def _extract_data_from_csv(
        path: str,
        run: tuple[int, str, dict[str, int]]
    )-> pd.DataFrame:
    '''
    Take all the csv files and turn them into one big dataframe for further processing
    
    Args: 
        path (str): Path to result directory with all CSV traces of a given scenario
    Return:
        pd.DataFrame: Combined dataframe with Node, Rep, and # Method columns added
    '''
    rep_filter, method_filter, parameter_filters = run

    dfs = []

    # Find all valid directories in experiment path
    for directory in Path(path).iterdir():
        if not directory.is_dir():
            continue    
        
        # Get all parameters/experiments in the directory
        dir_name = directory.name
        key_value_patterns = re.findall(r'([^=,]+)=([^,\-]+)', dir_name)
        dir_extracted_parameters = {}
        for key, value in key_value_patterns:
            # Numbers (int or decimal) are converted, everything else stays a string.
            parsed_value = _parse_number(value)
            if isinstance(parsed_value, str):
                # Remove # characters from string values.
                dir_extracted_parameters[key] = parsed_value.replace("#", "")
            else:
                dir_extracted_parameters[key] = parsed_value

        # Check whether directory matches with method filter
        if method_filter is not None: 
            if dir_extracted_parameters["Method"] != method_filter:
                continue

        # Check whether directory matches with parameter filter
        matches = True
        for parameter_name, parameter_value in parameter_filters.items():
            if (dir_extracted_parameters[parameter_name] != parameter_value):
                matches = False
                break
        if not matches:
            continue


        # Going through each file in the directory 
        for file in directory.iterdir():

            # Get node index and repetition and check whether the name makes sense
            node_index   = file.name.find("_")
            rep_index = file.name.find("rep")
            assert rep_index != -1 and node_index != -1, f"Unexpected Filename: {file.name}"

            # Get Repetition as int
            rep = int(re.search(r"rep(\d+)", file.name).group(1))
            # Rep Filter
            if rep_filter is not None and rep!= rep_filter:
                continue

            csv_file = pd.read_csv(file)
            csv_file.columns = csv_file.columns.str.replace(" ", "", regex=False)
            
            csv_file["Node_Name"] = file.name[:node_index] 
            csv_file["Rep"] = rep # Technically not needed anymore

            # Add dynamic experiment parameters.
            for key, value in dir_extracted_parameters.items():
                csv_file[key] = value

            dfs.append(csv_file)

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if df.empty: 
        print(
            f"No data found for "
            f"{rep_filter=}, "
            f"{method_filter=}, "
            f"{parameter_filters=}"
        )
        # Add derived helper columns
        return df
    
    set_node_id_in_df(df)
    set_node_and_src_is_anchor(df)
    return df


def set_node_and_src_is_anchor(df:pd.DataFrame):
    """ Add boolean columns identifying whether the source and receiver are anchors.

    Args:
        df (pd.DataFrame): Input dataframe containing NodeId, srcId, and Node_Name columns.

    Returns:
        None: The dataframe is updated in place.
    """
    physical_anchor_ids = set(
        df.loc[df["NodeType"] == "anchor", "NodeId"].unique()
    )

    df["node_is_anchor"] = df["NodeType"] == "anchor"
    df["src_is_anchor"] = df["srcId"].isin(physical_anchor_ids)

def get_sweep_parameters(sweep_parameter_list: list[str] | None)-> list[dict] :
    if sweep_parameter_list is None:
        return [{}]

    sweep_parameters = {}
    for parameter_specification in sweep_parameter_list:
        parameter_name, parameter_values = (
            _parse_sweep_parameters(parameter_specification)
        )
        sweep_parameters[parameter_name] = parameter_values

    return [
        dict(zip(sweep_parameters.keys(), values))
        for values in product(*sweep_parameters.values())
    ]
        


def set_node_id_in_df(df: pd.DataFrame) -> None:
    '''
    Get the NodeId from the nodeName for futher processing and add it as column to provided dataframe
    
    Parameters: 
        df: DataFrame retrieved from all data

    '''
    node_name_pattern = re.compile(
        r"^(anchor|persistentVAnchor|node)\[(\d+)\]$",
        re.IGNORECASE,
    )
    if "NodeId" in df.columns:
        return

    parsed_names = {}

    for node_name in df["Node_Name"].unique():
        match = node_name_pattern.fullmatch(node_name)
        if match is None:
            raise ValueError(f"Unsupported node name: {node_name!r}")

        raw_type, index = match.groups()
        normalized_type = raw_type.casefold()

        if normalized_type == "persistentvanchor":
            node_type = "persistent_v_anchor"
        else:
            node_type = normalized_type

        parsed_names[node_name] = (node_type, int(index))

    counts = {
        node_type: sum(
            1 for parsed_type, _ in parsed_names.values()
            if parsed_type == node_type
        )
        for node_type in ("anchor", "persistent_v_anchor", "node")
    }

    offsets = {
        "anchor": 0,
        "persistent_v_anchor": counts["anchor"],
        "node": counts["anchor"] + counts["persistent_v_anchor"],
    }

    id_mapping = {
        name: offsets[node_type] + index + 1
        for name, (node_type, index) in parsed_names.items()
    }
    type_mapping = {
        name: node_type
        for name, (node_type, _) in parsed_names.items()
    }

    df["NodeId"] = df["Node_Name"].map(id_mapping).astype(int)
    df["NodeType"] = df["Node_Name"].map(type_mapping)

def get_num_nodes_by_name(df: pd.DataFrame, name: str) -> int:
    '''
    Get the number of individual nodes of different types/name
    
    :param df: DataFrame that we use
    :type df: pd.DataFrame
    :param name: Name of the node type (usually 'node' or 'anchor')
    :type name: str
    :return: Number of nodes with specified name
    :rtype: int
    '''
    num_nodes = df[df["Node_Name"].str.contains(name)]["Node_Name"].nunique()
    assert(num_nodes != 0)
    return num_nodes 


def calc_twr_tof(R_a, R_b, D_a, D_b):
    """ Compute the TWR time of flight from round and delay timestamps.

    Args:
        R_a (float): First round duration at side A.
        R_b (float): Second round duration at side B.
        D_a (float): Reply delay at side A.
        D_b (float): Reply delay at side B.

    Returns:
        float: Estimated one-way time of flight.
    """
    numerator = R_a*R_b - D_a*D_b
    denominator = R_a + D_a + R_b + D_b
    tof = numerator/denominator
    return tof

# Below this ratio of smallest to largest singular value, the anchors count as
# (nearly) coplanar and the closed-form start is not used.
TRILATERATION_MIN_REL_SINGULAR_VALUE = 1e-3


def _linear_trilateration_start(
    anchors: np.ndarray, distances: np.ndarray
) -> np.ndarray | None:
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
    system_matrix = 2.0 * (shifted[1:] - shifted[0])
    right_hand_side = (
        np.sum(shifted[1:] ** 2, axis=1)
        - np.sum(shifted[0] ** 2)
        - distances[1:] ** 2
        + distances[0] ** 2
    )

    singular_values = np.linalg.svd(system_matrix, compute_uv=False)
    if (
        singular_values.size < 3
        or singular_values[0] <= 0
        or singular_values[2] / singular_values[0]
        < TRILATERATION_MIN_REL_SINGULAR_VALUE
    ):
        return None

    solution = np.linalg.lstsq(system_matrix, right_hand_side, rcond=None)[0]
    if not np.all(np.isfinite(solution)):
        return None
    return solution + centroid


def _trilaterate_position_from_tof(anchor_positions, ToFs, c=299792458.0):
    """ Estimate a 3D position by trilaterating time-of-flight measurements.

    The range equations are non-convex, so a least-squares run started at the
    anchor centroid can converge to a wrong local minimum (typically on the
    wrong side of the anchor geometry in Z), even for noise-free ranges.  If the
    anchors are not coplanar, the fit therefore starts at a closed-form linear
    estimate (_linear_trilateration_start) that lies in the correct basin.  For
    coplanar anchors, e.g. in 2D experiments, that estimate is undetermined and
    the anchor centroid is used as start.

    Args:
        anchor_positions (list): Anchor coordinates as tuples or arrays.
        ToFs (list): One-way time-of-flight measurements in seconds.
        c (float): Propagation speed used to convert time of flight into distances.

    Returns:
        np.ndarray: Estimated 3D position.
    """
    anchors = np.array(anchor_positions, dtype=float)
    distances = np.array(ToFs, dtype=float) * c  # ToF to distance

    def residual(p, anchors, d):
        """Return the difference between modeled and measured distances."""
        return np.sqrt(((anchors - p)**2).sum(axis=1)) - d

    start = _linear_trilateration_start(anchors, distances)
    if start is None:
        start = anchors.mean(axis=0)

    return least_squares(residual, start, args=(anchors, distances)).x


def filter_df(df:pd.DataFrame, filters: dict)->pd.DataFrame:
    """ Filter a dataframe by exact-match column-value pairs.

    Args:
        df (pd.DataFrame): Input dataframe to filter.
        filters (dict): Mapping from column names to required values.

    Returns:
        pd.DataFrame: Filtered dataframe view.
    """
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        mask &= df[col] == val
    return df.loc[mask]


def _merge_localization_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge self-localization, TWR and v-TWR results into a common output format.

    Creates:
        - est_position_error
        - est_position_x
        - est_position_y
        - est_position_z
        - localization_method

    Returns:
        pd.DataFrame: Copy of the input dataframe containing the merged columns.

    Raises:
        KeyError:
            If a required source column is missing.

        ValueError:
            If more than one localization method is active in a row.
    """
    
    for localization_method in ["v_twr", "twr", "self_loc"]:

        error_column = f"{localization_method}_error"

        # Get the rows where there is a position estimate made
        method_df = df[
            df[error_column].notna()
        ]

        # There must not be anything be in "est_position_error" yet
        if method_df["est_position_error"].notna().any():
            raise ValueError(
                f"Found overlapping localization results for {localization_method}"
            )
        
        # Set the error from the methods column in the overall column
        df.loc[method_df.index, "est_position_error"] = method_df[error_column]

        df.loc[method_df.index, "est_position_x"] = method_df[f"{localization_method}_x"]

        df.loc[method_df.index, "est_position_y"] = method_df[f"{localization_method}_y"]

        df.loc[method_df.index, "est_position_z"] = method_df[f"{localization_method}_z"]

        df.loc[method_df.index, "localization_method"] = localization_method

    return df
    

def _get_git_state() -> dict:
    """Return commit and dirty flag of this repository, or None if unavailable."""
    repository_dir = Path(__file__).parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository_dir,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repository_dir,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(status)}


def _save_df_to_file(
    df: pd.DataFrame,
    run: tuple[int, str, dict[str, int]],
    processing_settings: dict,
    simulation_info: dict,
    results_dir: Path,
):
    rep, method, sweep_parameters = run
    parameter_string = ",".join(f"{sweep_parameter_key}={sweep_parameter_value}" for sweep_parameter_key, sweep_parameter_value in sweep_parameters.items())
    output_dir = results_dir / ",".join(filter(None, [parameter_string, f"Method={method}"])) / f"rep{rep}"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"localization_events.csv"
    df.to_csv(file_path, index=False, encoding="utf-8")

    # Provenance of this result file.  processing_settings must be identical
    # for all runs of one results config; extract_statistics checks this.
    run_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": sys.argv,
        **_get_git_state(),
        "processing_settings": processing_settings,
        # Simulation end time and node count; see _get_simulation_info.
        "simulation": simulation_info,
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as file:
        json.dump(run_metadata, file, indent=2, default=str)


def _prepare_results_for_export(result_df: pd.DataFrame, run: tuple[int, str, dict[str, int]], timestamp_format)->pd.DataFrame:
    rep, method, sweep_parameters = run
    # Merge all errors and positions to one individual pos_error/pos and add a column for the method (should facilitate plotting)
    result_df= _merge_localization_results(result_df)
    # Remove rows that do not contain (relevant) data
    result_df = result_df[
        result_df["est_position_error"].notna()
    ].reset_index(drop=True).copy()
    # Horizontal (projected) error: the estimate is solved as 3D/2D, but only
    # its X/Y part is compared with the true X/Y of the same event row.
    result_df["est_position_error_2d"] = np.hypot(
        pd.to_numeric(result_df["est_position_x"]) - result_df["X"],
        pd.to_numeric(result_df["est_position_y"]) - result_df["Y"],
    )
    result_df["event_time"] = result_df["sim_timestamp"]
    result_df["Rep"] = rep
    result_df["Method"] = method
    result_df["used_timestamp_format"] = timestamp_format

    # Only keep columns that give useful data
    export_columns = [
        "event_time",
        "Rep",
        "Method",
        "NodeId",
        "interfaceIdx",
        "seqNo",
        "srcId",
        "destId",
        "msgType",
        "X",
        "Y",
        "Z",
        "v_X",
        "v_Y",
        "v_Z",
        "est_position_x",
        "est_position_y",
        "est_position_z",
        "est_position_error",
        "est_position_error_2d",
        "residuals",
        "residual_mad_score",
        "PDOP",
        "HDOP",
        "VDOP",
        "cond",
        "tdoa_solver_profile",
        "tdoa_optimizer_method",
        "tdoa_dimension",
        "tdoa_fixed_z",
        "tdoa_loss",
        "tdoa_f_scale",
        "tdoa_initial_PDOP",
        "tdoa_initial_cond",
        "tdoa_initial_rank",
        "tdoa_rank",
        "localization_bounds",
        "using_anchor_stitching",
        "used_anchors_or_pairs",
        "localization_method",
        "used_timestamp_format",
        *sweep_parameters.keys()
    ]

    result_df = result_df[export_columns]

    return result_df

def _localize_nodes(
    run_df: pd.DataFrame,
    method: str,
    timestamp_format: str,
    use_v_twr_pos: bool,
    *,
    tdoa_solver_profile: str,
    tdoa_dimension_mode: str,
    tdoa_loss: str,
    tdoa_f_scale_m: float | None,
) -> pd.DataFrame:
    """Localize the mobile nodes of one run with the given method."""
    # Set required columns to run_df and create lookup table
    _init_results_columns(run_df)
    event_lookup = _build_event_lookup(run_df)

    # Initialize immutable run geometry before selecting the localization
    # method.  Every run is expected to contain physical anchors, including a
    # TWR-only run; missing anchors therefore abort explicitly instead of
    # silently producing no result or event-dependent bounds.
    physical_anchor_positions = _get_physical_anchor_positions(run_df)
    localization_bounds_by_dimension = _build_localization_bounds_by_dimension(
        physical_anchor_positions
    )

    # Get the node ideas of all mobile nodes in this run
    mobile_node_ids = _get_mobile_node_ids(run_df)

    match method: 
        case "TWR":
            # Calculate TWR
            print("Calc TWR")
            return _localize_nodes_twr(run_df, mobile_node_ids, timestamp_format)

        case "HYPERBOLIC_SELF_LOCALIZATION":
            # Calculate self-localization with async. hyperbolic self-localization
            print("Calc standard Self-Loc")
            return _localize_nodes_self_loc(
                run_df,
                mobile_node_ids,
                timestamp_format,
                event_lookup,
                use_v_twr_pos=use_v_twr_pos,
                localization_bounds_by_dimension=localization_bounds_by_dimension,
                tdoa_solver_profile=tdoa_solver_profile,
                tdoa_dimension_mode=tdoa_dimension_mode,
                tdoa_loss=tdoa_loss,
                tdoa_f_scale_m=tdoa_f_scale_m,
            )

        case "CENTRALIZED" | "DISTRIBUTED":
            # Calculate the v-TWR Positions and from that, perform self-localization
            print("Calc HyRASeL Pos")
            _localize_nodes_vtwr(run_df, event_lookup, mobile_node_ids, timestamp_format)
            updated_lookup = _build_event_lookup(run_df)
            return _localize_nodes_self_loc(
                run_df,
                mobile_node_ids,
                timestamp_format,
                updated_lookup,
                use_v_twr_pos,
                localization_bounds_by_dimension=localization_bounds_by_dimension,
                tdoa_solver_profile=tdoa_solver_profile,
                tdoa_dimension_mode=tdoa_dimension_mode,
                tdoa_loss=tdoa_loss,
                tdoa_f_scale_m=tdoa_f_scale_m,
            )
        case _:
            raise Exception(f"{method} is not a valid method") 

def _get_mobile_node_ids(run_df: pd.DataFrame) -> list[int]:
    mobile_node_types = {"node", "persistent_v_anchor"}

    return (
        run_df.loc[
            run_df["NodeType"].isin(mobile_node_types),
            "NodeId",
        ]
        .unique()
        .tolist()
    )


def _parse_number(value: str) -> int | float | str:
    """ Parse a plain decimal number, preferring int over float.

    Strings that are not plain decimal numbers (e.g. method names) are returned unchanged.
    """
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def _parse_sweep_parameters(parameter_string: str):
    """ Parse a command-line parameter specification into a parameter name and numeric values.

    Args:
        parameter_string (str): Parameter string in the form name:v1,v2,v3.

    Returns:
        tuple[str, list[int | float]]: Parsed parameter name and list of values.
            Integers stay integers, decimals (e.g. 0.05) become floats.
    """
    # Separate elements by comma.
    name_part, list_part = parameter_string.split(':')
    parameter_list = []
    for x in list_part.split(','):
        value = _parse_number(x)
        if isinstance(value, str):
            raise ValueError(f"Sweep value '{x}' of parameter '{name_part}' is not a number")
        parameter_list.append(value)
    return (name_part, parameter_list)


def _get_available_reps(path):
    reps = set()
    for directory in Path(path).iterdir():
        if not directory.is_dir():
            continue

        for file in directory.iterdir():
            match = re.search(r"rep(\d+)", file.name)
            if match:
                reps.add(int(match.group(1)))

    return sorted(reps)


def _parse_rep_limit(rep_limit_arguments: list[str] | None) -> set[int] | None:
    """
    Converts repetition specifications into a set of integers.

    Supported formats:
        5
        1-10
        3,5,8-15
        1-5 8 10-12
    """

    if rep_limit_arguments is None:
        return None

    repetitions = set()

    for argument in rep_limit_arguments:
        for token in argument.split(","):
            token = token.strip()

            if not token:
                continue

            if "-" in token:
                start_string, end_string = token.split("-", 1)
                start = int(start_string)
                end = int(end_string)

                if start > end:
                    raise ValueError(
                        f"Invalid repetition range '{token}': start > end."
                    )

                repetitions.update(range(start, end + 1))
            else:
                repetitions.add(int(token))

    return repetitions

def _parse_residual_vector(value) -> np.ndarray | None:
    """Convert a residual value into a 1D numpy array.

    Residuals may already be numpy arrays during post-processing, but may also
    appear as lists, tuples, or strings after CSV round-trips.
    """
    if value is None:
        return None

    if isinstance(value, float) and np.isnan(value):
        return None

    if isinstance(value, np.ndarray):
        return np.ravel(value).astype(float)

    if isinstance(value, (list, tuple)):
        return np.ravel(np.asarray(value, dtype=float))

    if isinstance(value, str):
        cleaned_value = value.strip()

        if cleaned_value in {"", "None", "nan", "NaN"}:
            return None

        # Handles strings such as "[1.0 2.0 3.0]" or "[1.0, 2.0, 3.0]".
        cleaned_value = cleaned_value.strip("[]").replace(",", " ")
        residual_vector = np.fromstring(cleaned_value, sep=" ")

        if residual_vector.size == 0:
            return None

        return residual_vector.astype(float)

    return None


def _add_residual_mad_score_per_run(
    run_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add a per-run residual MAD score for every row with residuals.

    The score is defined as

        max(abs(residual_i - median) / sigma_robust)

    where median and sigma_robust are computed over all residual values of the
    current run. A row passes a later MAD filter with threshold k if

        residual_mad_score <= k

    Rows without residuals receive NaN because the MAD filter is not applicable.
    """
    result_df = run_df.copy()
    result_df["residual_mad_score"] = np.nan

    residual_vectors = result_df["residuals"].apply(_parse_residual_vector)
    valid_residual_mask = residual_vectors.notna()

    if not valid_residual_mask.any():
        return result_df

    all_residuals = np.hstack(
        [
            residual_vector
            for residual_vector in residual_vectors.loc[valid_residual_mask]
        ]
    )

    if all_residuals.size == 0:
        return result_df

    residual_median = np.median(all_residuals)
    residual_mad = np.median(np.abs(all_residuals - residual_median))
    sigma_robust = 1.4826 * residual_mad

    if sigma_robust == 0:
        # If all residuals are identical, nothing is an MAD outlier.
        result_df.loc[valid_residual_mask, "residual_mad_score"] = 0.0
        return result_df

    for row_index, residual_vector in residual_vectors.loc[valid_residual_mask].items():
        normalized_deviation = np.abs(residual_vector - residual_median) / sigma_robust
        result_df.at[row_index, "residual_mad_score"] = float(
            np.max(normalized_deviation)
        )

    return result_df

def _get_simulation_info(raw_df: pd.DataFrame) -> dict:
    """Simulation duration and node count of one run, taken from the raw traces.

    Simulations always start at t = 0; the largest raw timestamp is used as the
    simulation end.  Only regular nodes (node[*]) are counted, not anchors.
    Coverage statistics need both values as common time axis and denominator
    for all techniques.
    """
    if raw_df.empty:
        return {"end_time": None, "num_nodes": None}
    return {
        "end_time": float(raw_df["sim_timestamp"].max()),
        "num_nodes": int(
            raw_df.loc[raw_df["Node_Name"].str.startswith("node"), "Node_Name"].nunique()
        ),
    }


def _get_processing_settings(args: argparse.Namespace) -> dict:
    """Settings that must be shared by all runs written to one results config."""
    return {
        "experiment_path": str(Path(args.experiment_path).resolve()),
        "timestamp_format": args.timestamp_format,
        "use_vtwr_pos": args.use_vtwr_pos,
        "tdoa_solver_profile": args.tdoa_solver_profile,
        "tdoa_dimension": args.tdoa_dimension,
        "tdoa_loss": args.tdoa_loss,
        "tdoa_f_scale": args.tdoa_f_scale,
    }


def _get_arguments_from_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--method",
                        nargs="+",
                        type=str,
                        required=True,
                        help="What method should be used \n\t('TWR', 'HYPERBOLIC_SELF_LOCALIZATION', 'CENTRALIZED', 'DISTRIBUTED')")

    parser.add_argument("--experiment_path",
                        type=str,
                        required=True,
                        help="Path to  experiment to use \n\t('Baseline', 'Speed', 'NodeNumber')")

    parser.add_argument("--sweep_parameters",
                        nargs="+",
                        help="Optional list of sweep parameter name and value pairs (comma-separated) for data extraction\n\te.g. 'speed:1,5,10,20' or  'NodeNum:4,8,16,32'\n\tOmit for experiments without sweep parameters in the directory names")
    
    # The values of the following options are set in configs/defaults.yml.
    parser.add_argument("--timestamp_format",
                        choices=(SIM, CLOCK, NOISY),
                        required=True,
                        help="Timestamps used for the localization")

    parser.add_argument(
        "--tdoa_solver_profile",
        choices=("robust", "legacy"),
        required=True,
        help=(
            "TDoA solver profile. 'robust' uses the configurable 2D/3D "
            "bounded robust solve; 'legacy' overrides the other TDoA solver "
            "options and restores unbounded 3D LM with linear loss"
        ),
    )
    parser.add_argument(
        "--tdoa_dimension",
        choices=("auto", "2d", "3d"),
        required=True,
        help=(
            "TDoA solve dimension. 'auto' uses 2D when all participating "
            "anchors are planar; otherwise 3D"
        ),
    )
    parser.add_argument(
        "--tdoa_loss",
        choices=("linear", "soft_l1", "huber", "cauchy", "arctan"),
        required=True,
        help="Robust scipy least_squares loss",
    )
    parser.add_argument(
        "--tdoa_f_scale",
        type=float,
        help=(
            "Fixed robust-loss transition scale in metres. Without it, it is "
            "estimated separately for each sufficiently redundant event"
        ),
    )

    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Output directory for the runs (<results_dir>/<parameters>/repN)",
    )
    parser.add_argument("--use_vtwr_pos",
        action="store_true",
        default=False,
        help=(
            "Use VTWR-estimated node positions instead of ground-truth positions "
            "for HyRASeL mechanisms."
        ),
    )
    parser.add_argument(
        "--rep_limit",
        nargs="+",
        type=str,
        help=(
            "Repetitions to consider.\n"
            "Examples:\n"
            "  --rep_limit 5\n"
            "  --rep_limit 1-10\n"
            "  --rep_limit 3,5,8-15\n"
            "  --rep_limit 1-5 8 10-12"
        ),
    )
    # Unknown or misspelled options must abort instead of silently running with
    # default settings.
    args = parser.parse_args()
    args.rep_limit = _parse_rep_limit(args.rep_limit)
    if args.tdoa_f_scale is not None and (
        not np.isfinite(args.tdoa_f_scale) or args.tdoa_f_scale <= 0
    ):
        parser.error("--tdoa_f_scale must be finite and greater than zero")

    return args


if __name__ == "__main__":
    args = _get_arguments_from_parser()
    all_available_reps = _get_available_reps(args.experiment_path)
    print(all_available_reps)
    print(args.rep_limit)
    reduced_reps = all_available_reps
    if args.rep_limit is not None: 
        reduced_reps = set(all_available_reps).intersection(args.rep_limit)

    print(reduced_reps)

    reps = reduced_reps    

    for run in product(reps, args.method, get_sweep_parameters(args.sweep_parameters)):
        rep, method, sweep_parameter = run
        print(f"Loading run {run}")
        
        # Extract Simulation Data from CSV
        df = _extract_data_from_csv(path=args.experiment_path, run=run)
        simulation_info = _get_simulation_info(df)

        # Perform Post Processing to get the localization data
        result_df = _localize_nodes(
            df,
            method,
            timestamp_format=args.timestamp_format,
            use_v_twr_pos=args.use_vtwr_pos,
            tdoa_solver_profile=args.tdoa_solver_profile,
            tdoa_dimension_mode=args.tdoa_dimension,
            tdoa_loss=args.tdoa_loss,
            tdoa_f_scale_m=args.tdoa_f_scale,
        )
        result_df = _add_residual_mad_score_per_run(result_df)

        # Drop not necessary (help) columns and only populated rows
        finalized_result_df = _prepare_results_for_export(result_df, run, args.timestamp_format)


        # Save to file based on config-Name and run
        _save_df_to_file(
            finalized_result_df,
            run,
            processing_settings=_get_processing_settings(args),
            simulation_info=simulation_info,
            results_dir=args.results_dir,
        )

        print(finalized_result_df)

        
