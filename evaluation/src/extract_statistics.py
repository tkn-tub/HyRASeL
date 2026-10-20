"""Statistics of the post-processed localization events.

Filters the events of every run and writes error, coverage, availability and
v-anchor opportunity statistics per run and aggregated over the repetitions.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import re

import opportunities

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATA_FILE_NAME = "localization_events.csv"
RUN_METADATA_FILE_NAME = "run_metadata.json"

# Output layout:
#   <config>/postprocessed/<parameters>/repN/localization_events.csv   (input)
#   <config>/<estimator>/<filter_profile>/                             (evaluation dir)
#       runs/<parameters>/repN/*.csv
#       statistics_run_all.csv, statistics_run_aggregated.csv
POSTPROCESSED_DIR_NAME = "postprocessed"
RUNS_DIR_NAME = "runs"

# Runs whose simulation ended before this fraction of the median end time of
# the other repetitions of the same parameter directory are treated as aborted
# simulations and skipped.
MIN_RELATIVE_SIMULATION_END_TIME = 0.9

# Error dimension of the evaluation. "3d" uses est_position_error and filters
# by PDOP; "2d" uses est_position_error_2d and filters by HDOP.
ERROR_DIMENSIONS = ("2d", "3d")
DOP_COLUMN_BY_DIMENSION = {"2d": "HDOP", "3d": "PDOP"}

# Real anchors use the zero-based IDs 0 through NUM_REAL_ANCHORS - 1.
# Nodes with an ID greater than or equal to this value are virtual anchors.
NUM_REAL_ANCHORS = 4

ALL_LOCALIZATION_METHODS = ("self_loc", "v_twr", "twr")


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for statistics extraction."""
    parser = argparse.ArgumentParser(
        description="Extract and aggregate HyRASeL localization statistics.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        help="Results directory of one config (contains postprocessed/).",
    )
    # The values of the following options are set in configs/defaults.yml.
    parser.add_argument(
        "--coverage-window-size",
        type=float,
        required=True,
        help="Coverage-window duration in seconds.",
    )
    parser.add_argument(
        "--error-dimension",
        choices=ERROR_DIMENSIONS,
        required=True,
        help=(
            "Dimension of the evaluated position error. 3d uses "
            "est_position_error and filters by PDOP, 2d uses "
            "est_position_error_2d and filters by HDOP."
        ),
    )
    parser.add_argument(
        "--pdop-max",
        type=float,
        required=True,
        help=(
            "Maximum accepted dilution of precision. Applied to PDOP for "
            "--error-dimension 3d and to HDOP for 2d."
        ),
    )
    parser.add_argument(
        "--residual-mad-max",
        type=float,
        required=True,
        help="Maximum accepted residual MAD score.",
    )
    parser.add_argument(
        "--use-anchor-stitching",
        action="store_true",
        help="Include anchor-stitched events.",
    )
    parser.add_argument(
        "--localization-methods",
        nargs="+",
        choices=ALL_LOCALIZATION_METHODS,
        required=True,
        metavar="METHOD",
        help=(
            "One or more localization methods used jointly for all error, "
            "coverage, and interval statistics."
        ),
    )
    parser.add_argument(
        "--only-use-real-anchors-as-reference",
        action="store_true",
        help="Exclude self-localizations using virtual reference anchors.",
    )
    parser.add_argument(
        "--skip-empty-runs",
        action="store_true",
        help=(
            "Skip runs without any localization event. By default they are "
            "kept with coverage and availability 0 and NaN error statistics."
        ),
    )
    return parser


def evaluation_subdir(argv: list[str], processing_settings: dict) -> Path:
    """<estimator>/<filter_profile> of the extraction arguments argv for runs
    post-processed with processing_settings (also used by pipeline.py)."""
    args = _build_argument_parser().parse_args(argv)
    estimator_name = _get_estimator_name({"processing_settings": processing_settings})
    return Path(estimator_name) / _filter_profile_of(args)


# -----------------------------------------------------------------------------
# Metadata helpers
# -----------------------------------------------------------------------------

def _convert_parameter_value(value: str):
    """Convert path parameter values to bool, int, float, or keep as string."""
    value = value.strip()

    if value.lower() == "true":
        return True

    if value.lower() == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def _parse_key_value_path_part(path_part: str) -> dict:
    """
    Parse a path part containing key-value pairs.

    Example:
        interAnchorDist=130,Method=CENTRALIZED

    Returns:
        {
            "interAnchorDist": 130,
            "Method": "CENTRALIZED",
        }
    """
    parameters = {}

    for token in path_part.split(","):
        token = token.strip()

        if "=" not in token:
            continue

        key, value = token.split("=", 1)
        parameters[key.strip()] = _convert_parameter_value(value)

    return parameters


def _extract_rep_from_path(run_path: Path) -> int:
    """Extract repetition id from directory names like rep0, rep_0, or rep=0."""
    match = re.search(r"rep[=_-]?(\d+)", run_path.name, re.IGNORECASE)

    if match is None:
        raise ValueError(f"Could not extract repetition from path: {run_path}")

    return int(match.group(1))


def _extract_metadata_from_run_path(config_path: Path, run_path: Path) -> dict:
    """
    Extract metadata from the assumed structure:

        <Config>/<Parameters>/<repX>

    Example:
        results/AnchorDensity/interAnchorDist=130,Method=CENTRALIZED/rep0

    Returns:
        {
            "Config": "AnchorDensity",
            "Rep": 0,
            "interAnchorDist": 130,
            "Method": "CENTRALIZED",
        }
    """
    metadata = {
        "Config": config_path.name,
        "Rep": _extract_rep_from_path(run_path),
    }

    # Everything between config_path and rep-directory is treated as parameter path.
    parameter_path_parts = run_path.relative_to(config_path).parts[:-1]

    for path_part in parameter_path_parts:
        metadata.update(_parse_key_value_path_part(path_part))

    return metadata


def _build_filter_profile(
    pdop_max: float,
    residual_mad_max: float,
    use_anchor_stitching: bool,
    num_real_anchors: int,
    only_use_real_anchors_as_reference: bool,
    localization_methods: tuple[str, ...],
    error_dimension: str,
) -> str:
    """Build a compact filter-profile name for the output folder."""
    stitching_label = "stitch" if use_anchor_stitching else "nostitch"

    # Give the two reference-anchor filtering modes different output folders
    # so that running both configurations does not mix or overwrite results.
    reference_anchor_label = (
        f"real-ref-only-{num_real_anchors}"
        if only_use_real_anchors_as_reference
        else "all-ref"
    )

    # The prefix names the DOP measure that is actually filtered on.
    dop_label = DOP_COLUMN_BY_DIMENSION[error_dimension].lower()

    filter_profile = (
        f"{dop_label}-{pdop_max:g}_"
        f"mad-{residual_mad_max:g}_"
        f"{stitching_label}_"
        f"{reference_anchor_label}"
    )

    selected_localization_methods = set(localization_methods)
    all_localization_methods = set(ALL_LOCALIZATION_METHODS)
    if selected_localization_methods != all_localization_methods:
        ordered_methods = [
            method
            for method in ALL_LOCALIZATION_METHODS
            if method in selected_localization_methods
        ]
        filter_profile += f"_loc-{'-'.join(ordered_methods)}"

    return filter_profile


def _filter_profile_of(args: argparse.Namespace) -> str:
    """Filter-profile name of the parsed command-line arguments."""
    return _build_filter_profile(
        pdop_max=args.pdop_max,
        residual_mad_max=args.residual_mad_max,
        use_anchor_stitching=args.use_anchor_stitching,
        num_real_anchors=NUM_REAL_ANCHORS,
        only_use_real_anchors_as_reference=args.only_use_real_anchors_as_reference,
        localization_methods=tuple(args.localization_methods),
        error_dimension=args.error_dimension,
    )


# -----------------------------------------------------------------------------
# Statistics helpers
# -----------------------------------------------------------------------------

def _calculate_rmse(series: pd.Series) -> float:
    """Calculate the RMSE of a numerical error series."""
    return np.sqrt((series ** 2).mean())


def _calculate_series_statistics(series: pd.Series, prefix: str) -> dict:
    """Calculate common descriptive statistics for one numerical series."""
    return {
        f"mean_{prefix}": series.mean(),
        f"median_{prefix}": series.median(),
        f"std_{prefix}": series.std(),
        f"min_{prefix}": series.min(),
        f"max_{prefix}": series.max(),
        f"q25_{prefix}": series.quantile(0.25),
        f"q75_{prefix}": series.quantile(0.75),
        f"q95_{prefix}": series.quantile(0.95),
    }


def _make_metric_rows(
    metadata: dict,
    metrics: dict,
    units: dict | None = None,
    sample_sizes: dict | None = None,
) -> list[dict]:
    """Convert a metric dictionary into long-format statistic rows."""
    units = units or {}
    sample_sizes = sample_sizes or {}

    rows = []

    for metric_name, metric_value in metrics.items():
        rows.append({
            **metadata,
            "metric": metric_name,
            "value": metric_value,
            "n": sample_sizes.get(metric_name, np.nan),
            "unit": units.get(metric_name, ""),
        })

    return rows


def _get_metric_units() -> dict:
    """Define units for all statistics used in this script."""
    units = {}

    error_metrics = [
        "mean_error",
        "median_error",
        "std_error",
        "min_error",
        "max_error",
        "q25_error",
        "q75_error",
        "q95_error",
        "rmse_error",
    ]

    localization_interval_metrics = [
        "mean_localization_interval",
        "median_localization_interval",
        "std_localization_interval",
        "min_localization_interval",
        "max_localization_interval",
        "q25_localization_interval",
        "q75_localization_interval",
        "q95_localization_interval",
    ]

    coverage_metrics = [
        "mean_coverage",
        "median_coverage",
        "std_coverage",
        "min_coverage",
        "max_coverage",
        "q25_coverage",
        "q75_coverage",
        "q95_coverage",
    ]

    for metric_name in error_metrics:
        units[metric_name] = "m"

    for metric_name in localization_interval_metrics:
        units[metric_name] = "s"

    availability_metrics = [
        "q10_availability",
        "median_availability",
        "q90_availability",
    ]

    for metric_name in coverage_metrics + availability_metrics:
        units[metric_name] = "fraction"

    count_metrics = [
        "num_events_before_filtering",
        "num_events_after_filtering",
        "num_events_removed_by_filtering",
        "num_nodes",
        "num_coverage_windows",
    ]

    for metric_name in count_metrics:
        units[metric_name] = "count"

    units["update_rate"] = "1/s"

    opportunity_metrics = [
        *(f"opportunity_{outcome}" for outcome in opportunities.OUTCOMES),
        "opportunity_missed",
        "opportunity_efficiency",
        "mean_eligible_fraction",
    ]
    for metric_name in opportunity_metrics:
        units[metric_name] = "fraction"
    units["num_opportunities"] = "count"

    return units


def _get_sample_sizes(
    metrics: dict,
    num_error_samples: int,
    num_localization_interval_samples: int,
    num_coverage_samples: int | None = None,
    num_availability_samples: int | None = None,
) -> dict:
    """Assign sample sizes to metrics based on their metric group."""
    sample_sizes = {}

    for metric_name in metrics:
        if metric_name.endswith("_error") or metric_name == "rmse_error":
            sample_sizes[metric_name] = num_error_samples
        elif "localization_interval" in metric_name:
            sample_sizes[metric_name] = num_localization_interval_samples
        elif "coverage" in metric_name:
            sample_sizes[metric_name] = num_coverage_samples
        elif "availability" in metric_name:
            sample_sizes[metric_name] = num_availability_samples
        else:
            sample_sizes[metric_name] = np.nan

    return sample_sizes


# -----------------------------------------------------------------------------
# Processing helpers
# -----------------------------------------------------------------------------

def _make_estimator_view(
    results_df: pd.DataFrame,
    localization_methods: tuple[str, ...],
    error_dimension: str,
) -> pd.DataFrame:
    """Events of the selected localization methods in the statistics schema."""
    estimator_df = results_df.loc[
        results_df["localization_method"].isin(localization_methods)
    ].copy()

    if error_dimension == "2d":
        dop_column = DOP_COLUMN_BY_DIMENSION[error_dimension]
        missing_columns = [
            column
            for column in ("est_position_error_2d", dop_column)
            if column not in estimator_df.columns
        ]
        if missing_columns:
            raise ValueError(
                "--error-dimension 2d needs the columns "
                + ", ".join(missing_columns)
                + "; re-run the post-processing or use --error-dimension 3d."
            )
        # Downstream code always evaluates "est_position_error" and "PDOP".
        # Keep the 3D values under their own name and expose 2D error and HDOP
        # through the generic names.
        estimator_df["est_position_error_3d"] = estimator_df["est_position_error"]
        estimator_df["PDOP_3d"] = estimator_df["PDOP"]
        estimator_df["est_position_error"] = estimator_df["est_position_error_2d"]
        estimator_df["PDOP"] = estimator_df[dop_column]

    return estimator_df.loc[
        estimator_df["est_position_error"].notna()
    ].reset_index(drop=True)


def _evaluation_dir(config_path: Path, estimator_name: str, filter_profile: str) -> Path:
    """Directory of one evaluation, e.g. results/speed/huber/hdop-2_mad-3_stitch_all-ref."""
    return config_path / estimator_name / filter_profile


def _run_relative_path(config_path: Path, run_dir: Path) -> Path:
    """Run path below the config without the postprocessed level, e.g. speed=2,Method=TWR/rep0."""
    relative_path = run_dir.relative_to(config_path)
    if relative_path.parts and relative_path.parts[0] == POSTPROCESSED_DIR_NAME:
        relative_path = Path(*relative_path.parts[1:])
    return relative_path


def _load_run_metadata(run_dir: Path) -> dict:
    """Load run_metadata.json of one run (empty dict if it does not exist)."""
    metadata_path = run_dir / RUN_METADATA_FILE_NAME
    if not metadata_path.is_file():
        return {}
    with open(metadata_path, encoding="utf-8") as file:
        return json.load(file)


def _get_simulation_info(run_metadata: dict, run_dir: Path) -> tuple[int, float]:
    """Return (number of simulated nodes, simulation end time) of one run."""
    simulation_info = run_metadata.get("simulation") or {}
    if simulation_info.get("num_nodes") is None or simulation_info.get("end_time") is None:
        raise ValueError(
            f"{run_dir / RUN_METADATA_FILE_NAME} has no simulation end time / "
            "node count; run the post-processing again."
        )
    return int(simulation_info["num_nodes"]), float(simulation_info["end_time"])


def _get_estimator_name(run_metadata: dict) -> str:
    """Name of the TDoA estimator from the post-processing settings of a run."""
    settings = run_metadata.get("processing_settings") or {}
    if settings.get("tdoa_solver_profile") == "legacy":
        return "legacy"
    return settings.get("tdoa_loss") or "selected"

def _filter_localization_events(
    results_df: pd.DataFrame,
    pdop_max: float,
    residual_mad_max: float,
    use_anchor_stitching: bool,
    num_real_anchors: int,
    only_use_real_anchors_as_reference: bool,
) -> pd.DataFrame:
    """
    Filter localization events.

    Rules:
        - Keep rows with missing PDOP or PDOP <= pdop_max.
        - Keep rows with missing residual_mad_score or residual_mad_score <= residual_mad_max.
        - If use_anchor_stitching is False, remove rows where using_anchor_stitching is True.
        - If use_anchor_stitching is True, keep both stitched and non-stitched rows.
        - If only_use_real_anchors_as_reference is True, remove self-localization
          rows whose current reference anchor is virtual. Anchor IDs are
          zero-based, so IDs >= num_real_anchors are virtual.
    """
    # Allow all rows by default. This mask remains unchanged when virtual
    # reference anchors should be included in the statistics.
    reference_anchor_is_allowed = pd.Series(True, index=results_df.index)

    if only_use_real_anchors_as_reference:
        # srcId identifies the reference anchor only for self-localization rows.
        # Retain every other localization method regardless of srcId, while
        # retaining self-localizations only when srcId is a real anchor.
        # This excludes exactly:
        # localization_method == "self_loc" AND srcId >= num_real_anchors.
        reference_anchor_is_allowed = (
            (results_df["localization_method"] != "self_loc")
            | (results_df["srcId"] < num_real_anchors)
        )

    filtered_results_df = results_df.loc[
        (results_df["PDOP"].isna() | (results_df["PDOP"] <= pdop_max))
        & (
            results_df["residual_mad_score"].isna()
            | (results_df["residual_mad_score"] <= residual_mad_max)
        )
        & (use_anchor_stitching | (~results_df["using_anchor_stitching"]))
        & reference_anchor_is_allowed
    ].copy()

    return filtered_results_df


def _add_localization_interval(
    filtered_results_df: pd.DataFrame,
    time_column: str = "event_time",
) -> pd.DataFrame:
    """Add time difference between consecutive localization events per node."""
    filtered_results_df = filtered_results_df.sort_values(
        by=["NodeId", time_column]
    ).copy()

    filtered_results_df["localization_interval"] = (
        filtered_results_df
        .groupby("NodeId")[time_column]
        .diff()
    )

    return filtered_results_df


def _make_coverage_df(
    filtered_results_df: pd.DataFrame,
    metadata: dict,
    coverage_window_size: float,
    time_column: str = "event_time",
    total_nodes: int | None = None,
    end_time: float | None = None,
) -> pd.DataFrame:
    """Calculate coverage per time window.

    The windows cover t = 0 up to end_time (default: the last filtered event)
    without gaps. Windows without any localized node have coverage 0.
    """
    coverage_input_df = filtered_results_df.copy()

    coverage_input_df["time_bin_start"] = (
        np.floor(coverage_input_df[time_column] / coverage_window_size)
        * coverage_window_size
    )

    if total_nodes is None:
        total_nodes = coverage_input_df["NodeId"].nunique()

    coverage_df = (
        coverage_input_df
        .groupby(["time_bin_start", "NodeId"])
        .size()
        .reset_index(name="num_localizations")
    )

    coverage_df["localized"] = True

    coverage_df = (
        coverage_df
        .groupby("time_bin_start", as_index=False)
        .agg(localized_nodes=("localized", "sum"))
    )

    # Windows without any localization do not appear above. Reindex over the
    # complete time axis so that they count as zero coverage.
    if end_time is None:
        end_time = (
            float(coverage_input_df[time_column].max())
            if len(coverage_input_df)
            else 0.0
        )
    num_windows = int(np.floor(end_time / coverage_window_size)) + 1
    all_window_starts = np.arange(num_windows) * coverage_window_size
    # Bin starts were computed by floor(t / w) * w; map them to integer indices
    # to avoid float mismatches.
    coverage_df["window_index"] = np.rint(
        coverage_df["time_bin_start"] / coverage_window_size
    ).astype(int)
    localized_by_index = coverage_df.set_index("window_index")["localized_nodes"]
    coverage_df = pd.DataFrame({
        "time_bin_start": all_window_starts,
        "localized_nodes": (
            localized_by_index
            .reindex(range(num_windows), fill_value=0)
            .to_numpy()
        ),
    })

    coverage_df["total_nodes"] = total_nodes
    coverage_df["coverage"] = coverage_df["localized_nodes"] / total_nodes
    coverage_df["coverage_window_size"] = coverage_window_size
    coverage_df["time_bin_end"] = (
        coverage_df["time_bin_start"] + coverage_window_size
    )

    for metadata_key, metadata_value in metadata.items():
        coverage_df[metadata_key] = metadata_value

    coverage_columns = [
        *metadata.keys(),
        "coverage_window_size",
        "time_bin_start",
        "time_bin_end",
        "localized_nodes",
        "total_nodes",
        "coverage",
    ]

    return coverage_df[coverage_columns]


def _make_availability_df(
    filtered_results_df: pd.DataFrame,
    metadata: dict,
    coverage_window_size: float,
    total_nodes: int,
    end_time: float,
    time_column: str = "event_time",
) -> pd.DataFrame:
    """Calculate the localization availability A_i of every simulated node.

    A_i is the fraction of the coverage windows (same windows as in
    _make_coverage_df) in which node i was localized at least once.  Nodes
    without any filtered localization get A_i = 0 and NodeId NaN.  The mean of
    all A_i therefore equals the mean coverage of the run.
    """
    num_windows = int(np.floor(end_time / coverage_window_size)) + 1
    window_index = np.floor(
        filtered_results_df[time_column] / coverage_window_size
    ).astype(int)
    localized_windows = (
        pd.DataFrame({
            "NodeId": filtered_results_df["NodeId"],
            "window_index": window_index,
        })
        .loc[lambda dataframe: dataframe["window_index"] < num_windows]
        .drop_duplicates()
        .groupby("NodeId")
        .size()
    )

    num_unlocalized_nodes = total_nodes - len(localized_windows)
    if num_unlocalized_nodes < 0:
        raise ValueError(
            f"{len(localized_windows)} localized nodes but only {total_nodes} simulated nodes"
        )
    availability_df = pd.DataFrame({
        "NodeId": [*localized_windows.index, *[np.nan] * num_unlocalized_nodes],
        "availability": [
            *(localized_windows.to_numpy() / num_windows),
            *[0.0] * num_unlocalized_nodes,
        ],
    })
    availability_df["coverage_window_size"] = coverage_window_size

    for metadata_key, metadata_value in metadata.items():
        availability_df[metadata_key] = metadata_value

    return availability_df[
        [*metadata.keys(), "coverage_window_size", "NodeId", "availability"]
    ]


def _make_run_statistics_df(
    filtered_results_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    availability_df: pd.DataFrame,
    metadata: dict,
    len_before_filtering: int,
    len_after_filtering: int,
    total_nodes: int,
    end_time: float,
) -> pd.DataFrame:
    """Create long-format run-level statistics.

    update_rate: filtered position updates per simulated node and second
    (len_after_filtering / (total_nodes * end_time)).  It combines how often
    a localized node is updated with how many nodes are localized at all
    (coverage), e.g. for the scaling with the number of nodes.
    """
    error_series = filtered_results_df["est_position_error"]
    localization_interval_series = filtered_results_df["localization_interval"]
    coverage_series = coverage_df["coverage"]
    availability_series = availability_df["availability"]

    error_statistics = _calculate_series_statistics(
        error_series,
        prefix="error",
    )
    error_statistics["rmse_error"] = _calculate_rmse(error_series)

    localization_interval_statistics = _calculate_series_statistics(
        localization_interval_series,
        prefix="localization_interval",
    )

    coverage_statistics = _calculate_series_statistics(
        coverage_series,
        prefix="coverage",
    )

    # The mean availability equals mean_coverage and is therefore not stored.
    availability_statistics = {
        "q10_availability": availability_series.quantile(0.1),
        "median_availability": availability_series.median(),
        "q90_availability": availability_series.quantile(0.9),
    }

    run_metrics = {
        **error_statistics,
        **localization_interval_statistics,
        **coverage_statistics,
        **availability_statistics,
        "num_events_before_filtering": len_before_filtering,
        "num_events_after_filtering": len_after_filtering,
        "num_events_removed_by_filtering": len_before_filtering - len_after_filtering,
        "num_nodes": filtered_results_df["NodeId"].nunique(),
        "num_coverage_windows": len(coverage_df),
        "update_rate": len_after_filtering / (total_nodes * end_time),
    }

    sample_sizes = _get_sample_sizes(
        metrics=run_metrics,
        num_error_samples=error_series.notna().sum(),
        num_localization_interval_samples=localization_interval_series.notna().sum(),
        num_coverage_samples=coverage_series.notna().sum(),
        num_availability_samples=len(availability_series),
    )

    statistics_run_rows = _make_metric_rows(
        metadata=metadata,
        metrics=run_metrics,
        units=_get_metric_units(),
        sample_sizes=sample_sizes,
    )

    return pd.DataFrame(statistics_run_rows)


def _make_node_statistics_df(
    filtered_results_df: pd.DataFrame,
    metadata: dict,
) -> pd.DataFrame:
    """Create long-format node-level statistics."""
    statistics_node_rows = []
    metric_units = _get_metric_units()

    for node_id, node_df in filtered_results_df.groupby("NodeId"):
        node_metadata = {
            **metadata,
            "NodeId": node_id,
        }

        error_series = node_df["est_position_error"]
        localization_interval_series = node_df["localization_interval"]

        node_error_statistics = _calculate_series_statistics(
            error_series,
            prefix="error",
        )
        node_error_statistics["rmse_error"] = _calculate_rmse(error_series)

        node_localization_interval_statistics = _calculate_series_statistics(
            localization_interval_series,
            prefix="localization_interval",
        )

        node_metrics = {
            **node_error_statistics,
            **node_localization_interval_statistics,
            "num_events_after_filtering": len(node_df),
        }

        sample_sizes = _get_sample_sizes(
            metrics=node_metrics,
            num_error_samples=error_series.notna().sum(),
            num_localization_interval_samples=localization_interval_series.notna().sum(),
            num_coverage_samples=None,
        )

        statistics_node_rows.extend(
            _make_metric_rows(
                metadata=node_metadata,
                metrics=node_metrics,
                units=metric_units,
                sample_sizes=sample_sizes,
            )
        )

    return pd.DataFrame(statistics_node_rows)


# -----------------------------------------------------------------------------
# Statistics per localization method (e.g. v-TWR only)
# -----------------------------------------------------------------------------

def _make_localization_method_statistics_df(
    filtered_results_df: pd.DataFrame,
    metadata: dict,
    localization_methods: list[str],
) -> pd.DataFrame:
    """Error and localization-interval statistics of every localization method on its own.

    The regular run statistics combine all selected localization methods into
    one stream of node positions (e.g. HyRASeL = self_loc + v_twr).  To show
    the techniques separately (e.g. v-TWR in the numSlots figures), the same
    statistics are additionally computed on the events of one localization
    method only.  The metric names get the method as suffix:

        mean_error_v_twr, rmse_error_v_twr, mean_localization_interval_v_twr, ...

    The localization interval is recomputed on the subset, i.e. it is the time
    between two consecutive events of the same node *and* method.  Methods
    without filtered events of this run yield NaN values (the aggregation
    ignores them).
    """
    units = {}
    sample_sizes = {}
    run_metrics = {}

    for localization_method in localization_methods:
        method_df = _add_localization_interval(
            filtered_results_df.loc[
                filtered_results_df["localization_method"] == localization_method
            ],
            time_column="event_time",
        )
        error_series = method_df["est_position_error"]
        interval_series = method_df["localization_interval"]

        error_statistics = _calculate_series_statistics(
            error_series, prefix=f"error_{localization_method}"
        )
        error_statistics[f"rmse_error_{localization_method}"] = _calculate_rmse(error_series)
        interval_statistics = _calculate_series_statistics(
            interval_series, prefix=f"localization_interval_{localization_method}"
        )

        for metric_name in error_statistics:
            units[metric_name] = "m"
            sample_sizes[metric_name] = error_series.notna().sum()
        for metric_name in interval_statistics:
            units[metric_name] = "s"
            sample_sizes[metric_name] = interval_series.notna().sum()
        run_metrics.update(error_statistics)
        run_metrics.update(interval_statistics)

    return pd.DataFrame(
        _make_metric_rows(
            metadata=metadata,
            metrics=run_metrics,
            units=units,
            sample_sizes=sample_sizes,
        )
    )


# -----------------------------------------------------------------------------
# v-anchor opportunities (HyRASeL Centralized / Distributed, see opportunities.py)
# -----------------------------------------------------------------------------
#
# An opportunity is the first v-anchor slot of every anchor sequence.  The
# analysis needs the raw OMNeT++ traces (which node heard which anchor, who sent
# in which slot), not the localization events, and is independent of
# estimator and event filter.  It is therefore computed once per run here and
#   - written per run to runs/<parameters>/repN/opportunities.csv,
#   - appended to statistics_run.csv as opportunity_* metrics, so that it is
#     aggregated over the repetitions like every other metric.
# Doing it here instead of in post_processing.py means that existing results
# only need a new extraction, not a new (hours long) post-processing.

def _load_run_opportunities(
    run_dir: Path,
    base_metadata: dict,
    run_metadata: dict,
) -> pd.DataFrame | None:
    """v-anchor opportunities of one run, read from the raw traces (see opportunities.py).

    The raw traces are located via experiment_path in run_metadata.json and
    the parameter directory of the run (the raw directory has the same name
    plus '#' escapes and a trailing '-').

    Returns None for methods without v-anchor opportunities (TWR, hyperbolic
    self-localization, ...) and, with a warning, if the raw traces of the run
    are not available.  The other statistics are computed in both cases.
    """
    method = base_metadata.get("Method")
    if method not in opportunities.SUPPORTED_METHODS:
        return None

    experiment_path = (run_metadata.get("processing_settings") or {}).get("experiment_path")
    opportunity_df = (
        opportunities.load_run_opportunities(
            Path(experiment_path), run_dir.parent.name, base_metadata["Rep"], method
        )
        if experiment_path
        else None
    )
    if opportunity_df is None:
        print(
            f"WARNING: no raw traces for {run_dir} (experiment_path: "
            f"{experiment_path}); skipping v-anchor opportunities."
        )
    return opportunity_df


def _make_opportunity_statistics_df(
    opportunity_df: pd.DataFrame,
    metadata: dict,
) -> pd.DataFrame:
    """Long-format run-level opportunity statistics (same layout as statistics_run.csv).

    The sample size n of every opportunity metric is the number of evaluated
    anchor sequences (= opportunities) of the run.
    """
    run_metrics = opportunities.opportunity_metrics(opportunity_df)
    return pd.DataFrame(
        _make_metric_rows(
            metadata=metadata,
            metrics=run_metrics,
            units=_get_metric_units(),
            sample_sizes={metric_name: len(opportunity_df) for metric_name in run_metrics},
        )
    )


def _add_metadata_columns(dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Add metadata columns to a dataframe."""
    dataframe = dataframe.copy()

    for metadata_key, metadata_value in metadata.items():
        dataframe[metadata_key] = metadata_value

    return dataframe


def _reorder_filtered_event_columns(
    filtered_results_df: pd.DataFrame,
    metadata: dict,
) -> pd.DataFrame:
    """Put metadata columns first for easier inspection."""
    metadata_columns = list(metadata.keys())

    preferred_columns = [
        "event_time",
        "NodeId",
        "est_position_error",
        "localization_interval",
    ]

    remaining_columns = [
        column
        for column in filtered_results_df.columns
        if column not in metadata_columns and column not in preferred_columns
    ]

    ordered_columns = [
        *metadata_columns,
        *[column for column in preferred_columns if column in filtered_results_df.columns],
        *remaining_columns,
    ]

    return filtered_results_df[ordered_columns]

def _make_aggregated_run_statistics_df(statistics_run_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate run-level statistics across repetitions.

    Input:
        statistics_run_df:
            Long-format per-run statistics with columns:
            Config, Rep, Method, <sweep parameters>, <filter parameters>, metric, value, n, unit

    Output:
        Long-format aggregated statistics with:
            Config, Method, <sweep parameters>, <filter parameters>, metric, aggregate, value, n_runs, unit

    Notes:
        SEM is computed across per-run metric values.
    """
    if statistics_run_df.empty:
        return pd.DataFrame()

    # Columns that should not be used to define the aggregation group.
    non_group_columns = {
        "Rep",
        "value",
        "n",
        "aggregate",
    }

    group_columns = [
        column
        for column in statistics_run_df.columns
        if column not in non_group_columns
    ]

    grouped = statistics_run_df.groupby(group_columns, dropna=False)["value"]

    aggregated_wide_df = grouped.agg(
        mean="mean",
        std="std",
        min="min",
        q25=lambda values: values.quantile(0.25),
        median="median",
        q75=lambda values: values.quantile(0.75),
        max="max",
        n_runs="count",
    ).reset_index()

    aggregated_wide_df["sem"] = (
        aggregated_wide_df["std"] / np.sqrt(aggregated_wide_df["n_runs"])
    )

    value_columns = [
        "mean",
        "std",
        "sem",
        "min",
        "q25",
        "median",
        "q75",
        "max",
    ]

    aggregated_long_df = aggregated_wide_df.melt(
        id_vars=group_columns + ["n_runs"],
        value_vars=value_columns,
        var_name="aggregate",
        value_name="value",
    )

    return aggregated_long_df

def _warn_on_inconsistent_processing_settings(config_path: Path) -> None:
    """Warn if runs of one config were post-processed with different settings.

    Runs without a metadata file (results created before it existed) are
    reported but otherwise ignored.
    """
    settings_by_run: dict[str, list[Path]] = {}
    runs_without_metadata = 0
    for data_path in config_path.rglob(DATA_FILE_NAME):
        metadata_path = data_path.parent / RUN_METADATA_FILE_NAME
        if not metadata_path.is_file():
            runs_without_metadata += 1
            continue
        with open(metadata_path, encoding="utf-8") as file:
            run_metadata = json.load(file)
        settings_key = json.dumps(
            {
                "git_commit": run_metadata.get("git_commit"),
                "git_dirty": run_metadata.get("git_dirty"),
                "processing_settings": run_metadata.get("processing_settings"),
            },
            sort_keys=True,
        )
        settings_by_run.setdefault(settings_key, []).append(data_path.parent)

    if runs_without_metadata:
        print(
            f"WARNING: {runs_without_metadata} run(s) without "
            f"{RUN_METADATA_FILE_NAME}; their provenance is unknown."
        )
    if len(settings_by_run) > 1:
        print(
            f"WARNING: runs in {config_path} were post-processed with "
            f"{len(settings_by_run)} different settings/code versions:"
        )
        for settings_key, run_dirs in settings_by_run.items():
            print(f"  {len(run_dirs)} run(s), e.g. {run_dirs[0]}: {settings_key}")


def main(argv: list[str] | None = None) -> None:
    """Extract statistics using command-line configuration."""
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    if not args.config_path.is_dir():
        parser.error(f"config path is not a directory: {args.config_path}")
    if args.coverage_window_size <= 0:
        parser.error("--coverage-window-size must be greater than zero")
    if args.pdop_max <= 0:
        parser.error("--pdop-max must be greater than zero")
    if args.residual_mad_max <= 0:
        parser.error("--residual-mad-max must be greater than zero")

    _warn_on_inconsistent_processing_settings(args.config_path)

    localization_methods = tuple(args.localization_methods)
    localization_methods_label = ",".join(localization_methods)

    filter_profile = _filter_profile_of(args)
    print(f"Config path: {args.config_path}")
    print(
        f"Error dimension: {args.error_dimension} "
        f"(DOP filter on {DOP_COLUMN_BY_DIMENSION[args.error_dimension]})"
    )
    print(f"Filter profile: {filter_profile}")
    print(f"Localization methods: {localization_methods_label}")

    processed_runs = 0
    statistics_run_dfs_by_estimator: dict[str, list[pd.DataFrame]] = {}

    data_paths = sorted(args.config_path.rglob(DATA_FILE_NAME))
    run_metadata_by_dir = {
        data_path.parent: _load_run_metadata(data_path.parent)
        for data_path in data_paths
    }
    simulation_info_by_dir = {
        run_dir: _get_simulation_info(run_metadata, run_dir)
        for run_dir, run_metadata in run_metadata_by_dir.items()
    }
    # Median simulation end time of all repetitions of one parameter directory;
    # used to detect aborted simulations.
    median_end_time_by_parameter_dir = (
        pd.Series({
            run_dir: end_time
            for run_dir, (_, end_time) in simulation_info_by_dir.items()
        })
        .groupby(lambda run_dir: run_dir.parent)
        .median()
    )

    for data_path in data_paths:
        run_dir = data_path.parent

        base_metadata = _extract_metadata_from_run_path(
            config_path=args.config_path,
            run_path=run_dir,
        )

        # Every technique uses the same denominator (all
        # simulated nodes) and the same time axis (t = 0 up to the end of the
        # simulation), independent of how many events survive filtering.
        total_nodes, run_end_time = simulation_info_by_dir[run_dir]
        median_end_time = median_end_time_by_parameter_dir[run_dir.parent]
        if run_end_time < MIN_RELATIVE_SIMULATION_END_TIME * median_end_time:
            print(
                f"\nWARNING: skipping {run_dir}: simulation ended at "
                f"{run_end_time:.3f} s instead of ~{median_end_time:.3f} s "
                "(aborted simulation)."
            )
            continue

        print(f"\nProcessing: {data_path}")
        results_df = pd.read_csv(data_path)
        if results_df.empty and args.skip_empty_runs:
            print("Skipping run without localization events (--skip-empty-runs).")
            continue

        # v-anchor opportunities (Centralized/Distributed only) come from the
        # raw traces and are independent of estimator and filter.
        opportunity_df = _load_run_opportunities(
            run_dir, base_metadata, run_metadata_by_dir[run_dir]
        )

        estimator_df = _make_estimator_view(
            results_df,
            localization_methods,
            args.error_dimension,
        )
        if estimator_df.empty and args.skip_empty_runs:
            continue

        estimator_name = _get_estimator_name(run_metadata_by_dir[run_dir])
        metadata = {
            **base_metadata,
            "error_dimension": args.error_dimension,
            "pdop_max": args.pdop_max,
            "residual_mad_max": args.residual_mad_max,
            "use_anchor_stitching": args.use_anchor_stitching,
            "num_real_anchors": NUM_REAL_ANCHORS,
            "only_use_real_anchors_as_reference": (
                args.only_use_real_anchors_as_reference
            ),
            "localization_methods": localization_methods_label,
            "tdoa_estimator": estimator_name,
        }
        print(f"Estimator: {estimator_name}; metadata: {metadata}")

        filtered_results_df = _filter_localization_events(
            results_df=estimator_df,
            pdop_max=args.pdop_max,
            residual_mad_max=args.residual_mad_max,
            use_anchor_stitching=args.use_anchor_stitching,
            num_real_anchors=NUM_REAL_ANCHORS,
            only_use_real_anchors_as_reference=(
                args.only_use_real_anchors_as_reference
            ),
        )

        len_before_filtering = len(estimator_df)
        len_after_filtering = len(filtered_results_df)
        print(
            f"{estimator_name}: removed "
            f"{len_before_filtering - len_after_filtering} of "
            f"{len_before_filtering} entries"
        )

        filtered_results_df = _add_localization_interval(
            filtered_results_df=filtered_results_df,
            time_column="event_time",
        )
        coverage_df = _make_coverage_df(
            filtered_results_df=filtered_results_df,
            metadata=metadata,
            coverage_window_size=args.coverage_window_size,
            time_column="event_time",
            total_nodes=total_nodes,
            end_time=run_end_time,
        )
        availability_df = _make_availability_df(
            filtered_results_df=filtered_results_df,
            metadata=metadata,
            coverage_window_size=args.coverage_window_size,
            total_nodes=total_nodes,
            end_time=run_end_time,
        )
        statistics_run_df = _make_run_statistics_df(
            filtered_results_df=filtered_results_df,
            coverage_df=coverage_df,
            availability_df=availability_df,
            metadata=metadata,
            len_before_filtering=len_before_filtering,
            len_after_filtering=len_after_filtering,
            total_nodes=total_nodes,
            end_time=run_end_time,
        )
        statistics_node_df = _make_node_statistics_df(
            filtered_results_df=filtered_results_df,
            metadata=metadata,
        )
        # Additional run metrics, appended to the same statistics_run.csv:
        # error/interval per localization method (e.g. mean_error_v_twr)
        # for every method that occurs in this run, and the v-anchor
        # opportunity metrics.
        run_localization_methods = [
            method
            for method in ALL_LOCALIZATION_METHODS
            if method in set(estimator_df["localization_method"])
        ]
        additional_statistics_dfs = [
            _make_localization_method_statistics_df(
                filtered_results_df, metadata, run_localization_methods
            )
        ]
        if opportunity_df is not None:
            additional_statistics_dfs.append(
                _make_opportunity_statistics_df(opportunity_df, metadata)
            )
        statistics_run_df = pd.concat(
            [statistics_run_df, *additional_statistics_dfs],
            ignore_index=True,
        )
        statistics_run_dfs_by_estimator.setdefault(
            estimator_name, []
        ).append(statistics_run_df)

        filtered_results_df = _add_metadata_columns(
            dataframe=filtered_results_df,
            metadata=metadata,
        )
        filtered_results_df = _reorder_filtered_event_columns(
            filtered_results_df=filtered_results_df,
            metadata=metadata,
        )

        # Each estimator has its own evaluation directory, e.g.
        # <config>/huber/<filter_profile>/runs/<parameters>/repN/.
        run_output_dir = (
            _evaluation_dir(args.config_path, estimator_name, filter_profile)
            / RUNS_DIR_NAME
            / _run_relative_path(args.config_path, run_dir)
        )
        run_output_dir.mkdir(parents=True, exist_ok=True)
        filtered_results_df.to_csv(
            run_output_dir / "localization_events_filtered.csv",
            index=False,
        )
        coverage_df.to_csv(run_output_dir / "coverage.csv", index=False)
        availability_df.to_csv(run_output_dir / "availability.csv", index=False)
        statistics_run_df.to_csv(run_output_dir / "statistics_run.csv", index=False)
        statistics_node_df.to_csv(run_output_dir / "statistics_node.csv", index=False)
        if opportunity_df is not None:
            _add_metadata_columns(opportunity_df, metadata).to_csv(
                run_output_dir / "opportunities.csv", index=False
            )
        print(f"Saved outputs to: {run_output_dir}")
        processed_runs += 1

    print(f"\nProcessed {processed_runs} runs.")

    for estimator_name, statistics_run_dfs in (
        statistics_run_dfs_by_estimator.items()
    ):
        all_statistics_run_df = pd.concat(
            statistics_run_dfs,
            ignore_index=True,
        )

        aggregated_statistics_run_df = _make_aggregated_run_statistics_df(
            statistics_run_df=all_statistics_run_df,
        )

        evaluation_dir = _evaluation_dir(
            args.config_path, estimator_name, filter_profile
        )
        evaluation_dir.mkdir(parents=True, exist_ok=True)

        all_statistics_run_df.to_csv(
            evaluation_dir / "statistics_run_all.csv",
            index=False,
        )

        aggregated_statistics_run_df.to_csv(
            evaluation_dir / "statistics_run_aggregated.csv",
            index=False,
        )

        print(f"Saved {estimator_name} aggregated statistics to: {evaluation_dir}")


if __name__ == "__main__":
    main()



# Input:
# localization_events.csv

# Output (per run in <config>/<estimator>/<filter_profile>/runs/<parameters>/repN/):
# localization_events_filtered.csv
#   Usage:      Filtered localization events; basis for all following statistics and localization-error ECDFs.
#   CSV Header: Config, Rep, Method, <Sweep Parameters>, filter_profile, event_time, NodeId, position_error, inter_localization_time

# coverage.csv
#   Usage: Coverage per time window; used for coverage ECDFs, coverage-over-time plots, and coverage statistics.
#   CSV Header: Config, Rep, Method, <Sweep Parameters>, filter_profile, coverage_window_size, time_bin_start, time_bin_end, localized_nodes, total_nodes, coverage

# availability.csv
#   Usage: Localization availability A_i per simulated node (unlocalized nodes: NodeId NaN, A_i = 0); used for availability CDFs.
#   CSV Header: Config, Rep, Method, <Sweep Parameters>, filter_profile, coverage_window_size, NodeId, availability

# statistics_run.csv
#   Usage: Aggregated statistics per run; used for line plots and tables (one metric per run).
#   CSV Header: Config, Rep, Method, <Sweep Parameters>, filter_profile, metric, value, n, unit
#   Besides the statistics of the combined event stream it contains
#   <statistic>_<localization method> (e.g. mean_error_v_twr), update_rate
#   (position updates per simulated node and second) and the
#   opportunity_* metrics (see opportunities.csv).

# statistics_node.csv
#   Usage: Aggregated statistics per node; used for node fairness analysis, node-wise statistics, and debugging.
#   CSV Header: Config, Rep, Method, <Sweep Parameters>, filter_profile, NodeId, metric, value, n, unit

# opportunities.csv (CENTRALIZED and DISTRIBUTED only, read from the raw traces; see opportunities.py)
#   Usage: Outcome of every v-anchor opportunity (one per anchor sequence); used for opportunity plots.
#   CSV Header: seqNo, seq_start_time, num_nodes, num_eligible_nodes, num_first_slot_senders, num_stale_first_slot_senders, used, outcome, <metadata>
#   The run-level opportunity_* metrics are appended to statistics_run.csv.
