"""Localization events of the simulated runs, reconstructed from the raw traces.

Reads the OMNeT++ traces of one config and writes, per run, the TWR, t-TWR and
TDoA position estimates and their errors (localization.py) to
<results_dir>/<parameters>/repN/localization_events.csv.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from localization import METHODS, localize
from runs import (
    CLOCK, EVENTS_FILE_NAME, NOISY, RUN_METADATA_FILE_NAME, SIM, find_run_traces, num_node_traces,
    parameter_combinations, parse_reps, parse_value, raw_dirs, read_traces, run_path, trace_files,
)
from solvers import MAD_TO_SIGMA

# Columns of localization_events.csv, followed by the sweep parameters.
EXPORT_COLUMNS = [
    "event_time", "Rep", "Method", "NodeId", "interfaceIdx", "seqNo", "srcId", "destId", "msgType",
    "X", "Y", "Z", "v_X", "v_Y", "v_Z",
    "est_position_x", "est_position_y", "est_position_z", "est_position_error", "est_position_error_2d",
    "residuals", "residual_mad_score", "PDOP", "HDOP", "VDOP", "cond",
    "tdoa_dimension", "tdoa_fixed_z", "tdoa_loss", "tdoa_f_scale",
    "tdoa_initial_PDOP", "tdoa_initial_cond", "tdoa_initial_rank", "tdoa_rank", "localization_bounds",
    "using_anchor_stitching", "used_anchors_or_pairs", "localization_method", "used_timestamp_format",
]


def read_run(paths: list[Path], parameters: dict) -> pd.DataFrame:
    """All records of the trace files of one run (runs.read_traces) with its parameters as columns."""
    run_df = read_traces(paths)
    for key, value in parameters.items():
        run_df[key] = value
    return run_df


def _simulation_info(run_df: pd.DataFrame, paths: list[Path]) -> dict:
    """Simulation end time and number of regular nodes (node traces, not anchors) of one run.

    Simulations start at t = 0, the largest raw timestamp is taken as their
    end.  Coverage statistics need both as common time axis and denominator
    for all techniques.
    """
    return {"end_time": float(run_df[SIM].max()), "num_nodes": num_node_traces(paths)}


def _residual_mad_scores(residuals: pd.Series) -> pd.Series:
    """Residual MAD score max_i |r_i - median| / sigma of every row with residuals, else NaN.

    median and sigma = 1.4826 MAD are taken over all residuals of the run; a
    later MAD filter with threshold k keeps the rows with a score <= k.
    """
    # Only TDoA events have residuals (an array); TWR and t-TWR events have None.
    vectors = [np.ravel(r).astype(float) for r in residuals if isinstance(r, np.ndarray)]
    if not vectors:
        return pd.Series(np.nan, index=residuals.index)

    all_residuals = np.hstack(vectors)
    median = np.median(all_residuals)
    sigma = MAD_TO_SIGMA * np.median(np.abs(all_residuals - median))
    scores = []
    for r in residuals:
        if not isinstance(r, np.ndarray):
            scores.append(np.nan)
        elif sigma == 0:
            # MAD 0: at least half of all residuals equal the median; every event gets score 0.
            scores.append(0.0)
        else:
            scores.append(float(np.max(np.abs(np.ravel(r).astype(float) - median) / sigma)))
    return pd.Series(scores, index=residuals.index, dtype=float)


def export_table(
    run_df: pd.DataFrame, results: dict[int, dict], run: tuple, timestamp_format: str,
) -> pd.DataFrame:
    """One row per localized event: its trace record, the estimate and the solver diagnostics.

    The result columns are those of the localization results (localization.py);
    every one of them must be in EXPORT_COLUMNS.  Missing entries are empty.
    """
    rep, method, sweep_parameters = run
    result_columns = set()
    for result in results.values():
        result_columns.update(result)
    not_exported = result_columns - set(EXPORT_COLUMNS)
    if not_exported:
        raise ValueError(f"Result columns missing in EXPORT_COLUMNS: {sorted(not_exported)}")

    # Trace record of every localized event, in trace order.
    event_indices = sorted(results)
    table = run_df.loc[event_indices].reset_index(drop=True)
    for column in EXPORT_COLUMNS:
        if column not in run_df.columns:
            # Objects as returned by localization.py; None where a result has no such column.
            values = [results[i].get(column) for i in event_indices]
            table[column] = pd.Series(values, dtype=object)
    # Horizontal error: X/Y of the (2D or 3D) estimate against the true X/Y.
    # to_numeric: the estimate columns are object columns (see above).
    table["est_position_error_2d"] = np.hypot(
        pd.to_numeric(table["est_position_x"]) - table["X"],
        pd.to_numeric(table["est_position_y"]) - table["Y"],
    )
    table["residual_mad_score"] = _residual_mad_scores(table["residuals"])
    # Simulation time, whichever timestamps the localization used.
    table["event_time"] = table[SIM]
    table["Rep"] = rep
    table["Method"] = method
    table["used_timestamp_format"] = timestamp_format
    # The sweep parameters are columns of the trace records (read_run).
    return table.reindex(columns=[*EXPORT_COLUMNS, *sweep_parameters])


def _git_state() -> dict:
    """Commit and dirty flag of this repository, or None if unavailable."""
    repository_dir = Path(__file__).parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository_dir, capture_output=True, text=True, check=True,
        ).stdout.strip()
        # Dirty: changes to tracked files; untracked files are ignored.
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repository_dir,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(status)}


def save_run(
    table: pd.DataFrame, run: tuple, settings: dict, simulation_info: dict, results_dir: Path,
) -> None:
    """Write the localization events and their provenance (run_metadata.json) of one run."""
    rep, method, sweep_parameters = run
    output_dir = run_path(results_dir, {**sweep_parameters, "Method": method}, rep)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / EVENTS_FILE_NAME, index=False, encoding="utf-8")

    run_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        # pipeline.job_is_done compares it with the command of the job (--resume).
        "command": sys.argv,
        **_git_state(),
        # Must be identical for all runs of one results config; extract_statistics checks this.
        "processing_settings": settings,
        "simulation": simulation_info,
    }
    with open(output_dir / RUN_METADATA_FILE_NAME, "w", encoding="utf-8") as file:
        json.dump(run_metadata, file, indent=2, default=str)


def _available_reps(experiment_path: str) -> list[int]:
    """Repetitions of all trace files of the experiment."""
    reps = set()
    for raw_dir in raw_dirs(Path(experiment_path)).values():
        reps.update(trace_files(raw_dir))
    return sorted(reps)


def _sweep_combinations(specifications: list[str] | None) -> list[dict]:
    """All combinations of sweep values, e.g. ["speed:1,5", "NodeNum:4"] -> [{speed: 1, NodeNum: 4}, ...]."""
    sweep = {}
    for specification in specifications or []:
        name, values = specification.split(":")
        # Numbers: columns of the events, and written alike in all directory names (e.g. 0.10 -> 0.1).
        sweep[name] = [parse_value(value) for value in values.split(",")]
        if any(isinstance(value, str) for value in sweep[name]):
            raise ValueError(f"Sweep values of parameter '{name}' must be numbers: {values}")
    return parameter_combinations(sweep)


def _processing_settings(args: argparse.Namespace) -> dict:
    """Settings that must be shared by all runs written to one results config."""
    return {
        "experiment_path": str(Path(args.experiment_path).resolve()),
        "timestamp_format": args.timestamp_format,
        "use_t_twr_pos": args.use_t_twr_pos,
        "copy_t_twr_pos_to_poll": args.copy_t_twr_pos_to_poll,
        "tdoa_dimension": args.tdoa_dimension,
        "tdoa_loss": args.tdoa_loss,
        "tdoa_f_scale": args.tdoa_f_scale,
    }


def parse_arguments() -> argparse.Namespace:
    """Command-line arguments; the values of most options are set in configs/defaults.yml."""
    # Unknown, misspelled or abbreviated options abort instead of silently running with default settings.
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--method", nargs="+", choices=list(METHODS), required=True, help="Localization methods")
    parser.add_argument("--experiment_path", required=True, help="Directory with the traces of one config")
    parser.add_argument(
        "--sweep_parameters", nargs="+",
        help="Sweep parameter names and values (comma-separated), e.g. 'speed:1,5,10,20' 'NodeNum:4,8'; "
             "omit for experiments without sweep parameters in the directory names",
    )
    parser.add_argument(
        "--timestamp_format", choices=(SIM, CLOCK, NOISY), required=True,
        help="Timestamps used for the localization",
    )
    parser.add_argument(
        "--tdoa_dimension", choices=("auto", "2d", "3d"), required=True,
        help="TDoA solve dimension. 'auto' uses 2D when all participating anchors are planar; otherwise 3D",
    )
    parser.add_argument(
        "--tdoa_loss", choices=("linear", "soft_l1", "huber", "cauchy", "arctan"), required=True,
        help="Robust scipy least_squares loss",
    )
    parser.add_argument(
        "--tdoa_f_scale", type=float,
        help="Fixed robust-loss transition scale in metres. Without it, it is estimated separately for each "
             "sufficiently redundant event",
    )
    parser.add_argument(
        "--results_dir", type=Path, required=True,
        help="Output directory for the runs (<results_dir>/<parameters>/repN)",
    )
    parser.add_argument(
        "--use_t_twr_pos", action="store_true",
        help="Use t-TWR-estimated node positions instead of ground-truth positions for HyRASeL mechanisms.",
    )
    parser.add_argument(
        "--copy_t_twr_pos_to_poll", action="store_true",
        help="Also store the t-TWR position of a FINAL in its POLLs (transmissions of the previous round) "
             "without an own t-TWR position, so that TDoA uses them as temporary anchors",
    )
    parser.add_argument(
        "--rep_limit", nargs="+",
        help="Repetitions to consider, e.g. '--rep_limit 5', '1-10', '3,5,8-15' or '1-5 8 10-12'",
    )
    args = parser.parse_args()
    args.rep_limit = None if args.rep_limit is None else set(parse_reps(args.rep_limit))
    if args.tdoa_f_scale is not None and (not np.isfinite(args.tdoa_f_scale) or args.tdoa_f_scale <= 0):
        parser.error("--tdoa_f_scale must be finite and greater than zero")
    return args


def main() -> None:
    """Post-process all runs of the given methods, sweep values and repetitions."""
    args = parse_arguments()
    settings = _processing_settings(args)
    reps = _available_reps(args.experiment_path)
    if args.rep_limit is not None:
        reps = [rep for rep in reps if rep in args.rep_limit]

    # run = (rep, method, sweep values), as passed to export_table and save_run.
    for run in product(reps, args.method, _sweep_combinations(args.sweep_parameters)):
        rep, method, sweep_parameters = run
        parameters = {**sweep_parameters, "Method": method}
        print(f"Run {run} @ {datetime.now()}")
        paths = find_run_traces(Path(args.experiment_path), parameters, rep)
        if not paths:
            print(f"WARNING: no traces of run {run} in {args.experiment_path}; skipped")
            continue
        run_df = read_run(paths, parameters)
        results = localize(run_df, method, settings)
        table = export_table(run_df, results, run, args.timestamp_format)
        save_run(table, run, settings, _simulation_info(run_df, paths), args.results_dir)
        print(f"{len(table)} localization events @ {datetime.now()}")


if __name__ == "__main__":
    main()
