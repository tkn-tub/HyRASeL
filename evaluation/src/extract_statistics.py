"""Statistics of the post-processed localization events.

Filters the events of every run (EventFilter) and writes their metrics
(metrics.py) per run and aggregated over the repetitions:

    <config>/postprocessed/<parameters>/repN/   localization_events.csv and run_metadata.json (input)
    <config>/<estimator>/<filter profile>/      evaluation directory, e.g. huber/hdop-2_mad-3_stitch
        runs/<parameters>/repN/
            localization_events_filtered.csv    filtered events with localization_interval
            coverage.csv                        coverage of every window
            availability.csv                    availability of every simulated node
            statistics_run.csv                  metrics of the run (metric, value, n, unit)
            statistics_node.csv                 metrics of every node
            opportunities.csv                   t-anchor opportunities (opportunities.SUPPORTED_METHODS)
        statistics_run_all.csv                  statistics_run.csv of all runs
        statistics_run_aggregated.csv           aggregates of every metric over the repetitions

All files have the metadata columns of their run: Config, Rep, the parameters
of the run directory, the filter settings and tdoa_estimator.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import metrics
import opportunities
from runs import (
    AGGREGATED_FILE_NAME, AVAILABILITY_FILE_NAME, COVERAGE_FILE_NAME, EVENTS_FILE_NAME, FILTERED_EVENTS_FILE_NAME,
    POSTPROCESSED_DIR_NAME, RUN_METADATA_FILE_NAME, RUNS_DIR_NAME, parse_run_path,
)

# Runs that ended before this fraction of the median end time of the repetitions with the same parameters are
# aborted simulations and skipped.
MIN_RELATIVE_SIMULATION_END_TIME = 0.9

# DOP column of the filter per error dimension; 2d evaluates est_position_error_2d instead of est_position_error.
DOP_COLUMN_BY_DIMENSION = {"2d": "HDOP", "3d": "PDOP"}

# Columns of localization_events_filtered.csv after the metadata, followed by the other event columns.
FIRST_EVENT_COLUMNS = ("event_time", "NodeId", "est_position_error", "localization_interval")

# Aggregates of every metric over the repetitions, in output order; sem is std / sqrt(n_runs).
AGGREGATES = ["mean", "std", "sem", "min", "q25", "median", "q75", "max"]


def _argument_parser() -> argparse.ArgumentParser:
    """Command-line interface; the values of most options are set in configs/defaults.yml."""
    parser = argparse.ArgumentParser(
        description="Extract and aggregate HyRASeL localization statistics.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config-path", type=Path, required=True, help="Results directory of one config (contains postprocessed/).",
    )
    parser.add_argument(
        "--coverage-window-size", type=float, required=True, help="Coverage-window duration in seconds.",
    )
    parser.add_argument(
        "--error-dimension",
        choices=tuple(DOP_COLUMN_BY_DIMENSION),
        required=True,
        help=(
            "Dimension of the evaluated position error. 3d uses est_position_error and filters by PDOP, "
            "2d uses est_position_error_2d and filters by HDOP."
        ),
    )
    parser.add_argument(
        "--pdop-max",
        type=float,
        required=True,
        help="Maximum accepted dilution of precision. Applied to PDOP for --error-dimension 3d and to HDOP for 2d.",
    )
    parser.add_argument("--residual-mad-max", type=float, required=True, help="Maximum accepted residual MAD score.")
    parser.add_argument("--use-anchor-stitching", action="store_true", help="Include anchor-stitched events.")
    parser.add_argument(
        "--localization-methods",
        nargs="+",
        choices=metrics.LOCALIZATION_METHODS,
        required=True,
        metavar="METHOD",
        help="One or more localization methods used jointly for all error, coverage, and interval statistics.",
    )
    parser.add_argument(
        "--skip-empty-runs",
        action="store_true",
        help=(
            "Skip runs without any localization event. By default they are kept with coverage and "
            "availability 0 and NaN error statistics."
        ),
    )
    return parser


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Command-line arguments; exits on invalid values."""
    parser = _argument_parser()
    args = parser.parse_args(argv)
    if not args.config_path.is_dir():
        parser.error(f"config path is not a directory: {args.config_path}")
    for option in ("coverage_window_size", "pdop_max", "residual_mad_max"):
        if getattr(args, option) <= 0:
            parser.error(f"--{option.replace('_', '-')} must be greater than zero")
    return args


@dataclass(frozen=True)
class EventFilter:
    """Selection of the evaluated localization events; every field is an option and a metadata column."""

    error_dimension: str
    pdop_max: float
    residual_mad_max: float
    use_anchor_stitching: bool
    localization_methods: tuple[str, ...]

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> EventFilter:
        """Filter of the parsed command-line arguments."""
        return cls(
            error_dimension=args.error_dimension,
            pdop_max=args.pdop_max,
            residual_mad_max=args.residual_mad_max,
            use_anchor_stitching=args.use_anchor_stitching,
            localization_methods=tuple(args.localization_methods),
        )

    @property
    def profile(self) -> str:
        """Name of the evaluation directory, e.g. hdop-2_mad-3_stitch or hdop-2_mad-3_stitch_loc-t_twr."""
        dop_name = DOP_COLUMN_BY_DIMENSION[self.error_dimension].lower()
        stitching = "stitch" if self.use_anchor_stitching else "nostitch"
        # :g drops trailing zeros, e.g. 2.0 -> 2.
        profile = f"{dop_name}-{self.pdop_max:g}_mad-{self.residual_mad_max:g}_{stitching}"
        # Only a subset of the methods is named, in the order of LOCALIZATION_METHODS.
        if set(self.localization_methods) != set(metrics.LOCALIZATION_METHODS):
            methods = []
            for method in metrics.LOCALIZATION_METHODS:
                if method in self.localization_methods:
                    methods.append(method)
            profile += f"_loc-{'-'.join(methods)}"
        return profile

    @property
    def metadata(self) -> dict:
        """Metadata columns of the filter settings."""
        return {
            "error_dimension": self.error_dimension,
            "pdop_max": self.pdop_max,
            "residual_mad_max": self.residual_mad_max,
            "use_anchor_stitching": self.use_anchor_stitching,
            "localization_methods": ",".join(self.localization_methods),
        }

    def select(self, events: pd.DataFrame) -> pd.DataFrame:
        """Events of the evaluated localization methods that have an error in the error dimension.

        For 2d, est_position_error and PDOP hold the 2D error and HDOP, the 3D
        values are kept as est_position_error_3d and PDOP_3d.
        """
        selected = events.loc[events["localization_method"].isin(self.localization_methods)]
        if self.error_dimension == "2d":
            dop_column = DOP_COLUMN_BY_DIMENSION["2d"]
            missing_columns = []
            for column in ("est_position_error_2d", dop_column):
                if column not in selected:
                    missing_columns.append(column)
            if missing_columns:
                raise ValueError(
                    f"--error-dimension 2d needs the columns {', '.join(missing_columns)}; "
                    "re-run the post-processing or use --error-dimension 3d."
                )
            # The 2D values move into the columns that apply() and the metrics read.
            selected = selected.assign(
                est_position_error_3d=selected["est_position_error"],
                PDOP_3d=selected["PDOP"],
                est_position_error=selected["est_position_error_2d"],
                PDOP=selected[dop_column],
            )
        return selected.loc[selected["est_position_error"].notna()].reset_index(drop=True)

    def apply(self, events: pd.DataFrame) -> pd.DataFrame:
        """Selected events within the DOP and residual MAD limits (missing values pass), stitched ones only if used."""
        keep = (events["PDOP"].isna() | (events["PDOP"] <= self.pdop_max)) & (
            events["residual_mad_score"].isna() | (events["residual_mad_score"] <= self.residual_mad_max)
        )
        if not self.use_anchor_stitching:
            keep &= ~events["using_anchor_stitching"]
        return events.loc[keep]


def filter_profile(argv: list[str]) -> str:
    """Filter profile of the extraction arguments argv (also used by pipeline.py)."""
    # Parsed like the extraction parses its arguments, so the name always matches.
    return EventFilter.from_args(_argument_parser().parse_args(argv)).profile


@dataclass(frozen=True)
class PostprocessedRun:
    """One post-processed run."""

    dir: Path
    parameters: dict  # sweep values and Method of the run directory
    metadata: dict  # Config, Rep and the parameters
    processing_settings: dict
    provenance: str  # processing settings and code version, to compare runs
    num_nodes: int  # simulated nodes, the denominator of coverage and availability
    end_time: float  # [s] end of the simulation, the time axis of coverage and availability


def _load_run(config_path: Path, run_dir: Path) -> PostprocessedRun:
    """Post-processed run of the directory postprocessed/<parameters>/repN with its run_metadata.json."""
    parameters, rep = parse_run_path(run_dir)
    metadata = {"Config": config_path.name, "Rep": rep, **parameters}

    metadata_path = run_dir / RUN_METADATA_FILE_NAME
    if not metadata_path.is_file():
        raise ValueError(f"{metadata_path} is missing; run the post-processing again.")
    run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    simulation = run_metadata.get("simulation") or {}
    if simulation.get("num_nodes") is None or simulation.get("end_time") is None:
        raise ValueError(f"{metadata_path} has no simulation end time / node count; run the post-processing again.")
    # Runs with the same JSON string were post-processed alike (_warn_on_inconsistent_processing).
    provenance = {key: run_metadata.get(key) for key in ("git_commit", "git_dirty", "processing_settings")}
    return PostprocessedRun(
        dir=run_dir,
        parameters=parameters,
        metadata=metadata,
        processing_settings=run_metadata.get("processing_settings") or {},
        provenance=json.dumps(provenance, sort_keys=True),
        num_nodes=int(simulation["num_nodes"]),
        end_time=float(simulation["end_time"]),
    )


def find_runs(config_path: Path) -> list[PostprocessedRun]:
    """Post-processed runs of a config, ordered by their directory; exits if there is none."""
    events_paths = sorted((config_path / POSTPROCESSED_DIR_NAME).glob(f"*/rep*/{EVENTS_FILE_NAME}"))
    if not events_paths:
        raise SystemExit(f"No post-processed runs in {config_path / POSTPROCESSED_DIR_NAME}")
    runs = []
    for events_path in events_paths:
        runs.append(_load_run(config_path, events_path.parent))
    return runs


def tdoa_estimator(runs: list[PostprocessedRun]) -> str:
    """TDoA estimator (tdoa_loss of the post-processing) of the runs (also used by pipeline.py); exits if it differs."""
    estimators = {run.processing_settings["tdoa_loss"] for run in runs}
    if len(estimators) > 1:
        raise SystemExit(
            f"The runs were post-processed with different tdoa_loss ({', '.join(sorted(estimators))}); "
            "post-process them again with one setting."
        )
    return estimators.pop()


def _warn_on_inconsistent_processing(config_path: Path, runs: list[PostprocessedRun]) -> None:
    """Warn if the runs were post-processed with different settings or code versions."""
    run_dirs_by_provenance = {}
    for run in runs:
        run_dirs_by_provenance.setdefault(run.provenance, []).append(run.dir)
    if len(run_dirs_by_provenance) > 1:
        print(
            f"WARNING: runs in {config_path} were post-processed with "
            f"{len(run_dirs_by_provenance)} different settings/code versions:"
        )
        for provenance, run_dirs in run_dirs_by_provenance.items():
            print(f"  {len(run_dirs)} run(s), e.g. {run_dirs[0]}: {provenance}")


def _median_end_times(runs: list[PostprocessedRun]) -> dict[Path, float]:
    """Median simulation end time of the repetitions of every parameter directory."""
    end_times = {}
    for run in runs:
        end_times.setdefault(run.dir.parent, []).append(run.end_time)
    medians = {}
    for parameter_dir, parameter_end_times in end_times.items():
        medians[parameter_dir] = float(np.median(parameter_end_times))
    return medians


def _opportunities(run: PostprocessedRun, events: pd.DataFrame) -> pd.DataFrame | None:
    """t-anchor opportunities of a run of opportunities.SUPPORTED_METHODS from its raw traces, otherwise None.

    The raw traces are found via experiment_path of the post-processing;
    without them, the run only gets a warning.  events: all localization
    events of the run, for the t-TWR positions.
    """
    if run.parameters["Method"] not in opportunities.SUPPORTED_METHODS:
        return None
    experiment_path = run.processing_settings.get("experiment_path")
    opportunity_table = None
    if experiment_path:
        t_twr_events = events.loc[events["localization_method"] == "t_twr"]
        t_twr_positions = set(zip(t_twr_events["NodeId"], t_twr_events["seqNo"]))
        opportunity_table = opportunities.load_run_opportunities(
            Path(experiment_path), run.parameters, run.metadata["Rep"], t_twr_positions
        )
    if opportunity_table is None:
        print(
            f"WARNING: no raw traces for {run.dir} (experiment_path: {experiment_path}); "
            "skipping t-anchor opportunities."
        )
    return opportunity_table


def _with_metadata(table: pd.DataFrame, metadata: dict, first_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    """table with the metadata columns and then first_columns in front."""
    table = table.assign(**metadata)
    columns = [*metadata, *first_columns]
    return table[[*columns, *table.columns.drop(columns)]]


def extract_run(
    run: PostprocessedRun,
    event_filter: EventFilter,
    estimator: str,
    coverage_window_size: float,
    skip_empty_runs: bool,
    evaluation_dir: Path,
) -> pd.DataFrame | None:
    """Filter the events of one run and write its outputs; returns its metrics, None if it is skipped."""
    events_path = run.dir / EVENTS_FILE_NAME
    print(f"\nProcessing: {events_path}")
    events = pd.read_csv(events_path)
    if events.empty and skip_empty_runs:
        print("Skipping run without localization events (--skip-empty-runs).")
        return None
    # selected: events of the evaluated methods before the filter; filtered: after it.
    selected = event_filter.select(events)
    if selected.empty and skip_empty_runs:
        return None

    metadata = {**run.metadata, **event_filter.metadata, "tdoa_estimator": estimator}
    print(f"Estimator: {estimator}; metadata: {metadata}")
    filtered = metrics.add_localization_interval(event_filter.apply(selected))
    print(f"{estimator}: removed {len(selected) - len(filtered)} of {len(selected)} entries")

    coverage = metrics.coverage_table(filtered, coverage_window_size, run.num_nodes, run.end_time)
    availability = metrics.availability_table(filtered, coverage_window_size, run.num_nodes, run.end_time)
    opportunity_table = _opportunities(run, events)
    statistics = metrics.run_statistics(
        selected, filtered, coverage, availability, run.num_nodes, run.end_time, opportunity_table
    )
    statistics = _with_metadata(pd.DataFrame(statistics), metadata)
    node_statistics = metrics.node_statistics(filtered)
    # Without filtered events, statistics_node.csv is empty (not even a header).
    if node_statistics.empty:
        node_statistics = pd.DataFrame()
    else:
        node_statistics = _with_metadata(node_statistics, metadata)

    output_dir = evaluation_dir / RUNS_DIR_NAME / run.dir.parent.name / run.dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    events_table = _with_metadata(filtered, metadata, FIRST_EVENT_COLUMNS)
    events_table.to_csv(output_dir / FILTERED_EVENTS_FILE_NAME, index=False)
    _with_metadata(coverage, metadata).to_csv(output_dir / COVERAGE_FILE_NAME, index=False)
    _with_metadata(availability, metadata).to_csv(output_dir / AVAILABILITY_FILE_NAME, index=False)
    statistics.to_csv(output_dir / "statistics_run.csv", index=False)
    node_statistics.to_csv(output_dir / "statistics_node.csv", index=False)
    if opportunity_table is not None:
        # Metadata columns after the opportunity columns.
        opportunity_table.assign(**metadata).to_csv(output_dir / "opportunities.csv", index=False)
    print(f"Saved outputs to: {output_dir}")
    return statistics


def aggregate(statistics: pd.DataFrame) -> pd.DataFrame:
    """Aggregates (AGGREGATES) of every metric over the repetitions, with the number of runs n_runs."""
    # All columns except Rep and the values: one group per metric with the repetitions of one parameter directory.
    group_columns = list(statistics.columns.drop(["Rep", "value", "n"]))
    table = statistics.groupby(group_columns, dropna=False)["value"].agg(
        mean="mean",
        std="std",
        min="min",
        q25=lambda values: values.quantile(0.25),
        median="median",
        q75=lambda values: values.quantile(0.75),
        max="max",
        n_runs="count",
    ).reset_index()
    table["sem"] = table["std"] / np.sqrt(table["n_runs"])
    # Long format: one row per metric and aggregate.
    return table.melt(
        id_vars=[*group_columns, "n_runs"], value_vars=AGGREGATES, var_name="aggregate", value_name="value"
    )


def main(argv: list[str] | None = None) -> None:
    """Extract the statistics of all runs of one config."""
    args = parse_arguments(argv)
    event_filter = EventFilter.from_args(args)
    runs = find_runs(args.config_path)
    _warn_on_inconsistent_processing(args.config_path, runs)
    estimator = tdoa_estimator(runs)
    evaluation_dir = args.config_path / estimator / event_filter.profile
    print(f"Config path: {args.config_path}")
    print(f"Error dimension: {args.error_dimension} (DOP filter on {DOP_COLUMN_BY_DIMENSION[args.error_dimension]})")
    print(f"Filter profile: {event_filter.profile}")
    print(f"Localization methods: {event_filter.metadata['localization_methods']}")

    # Reference for detecting aborted simulations, per parameter directory.
    median_end_times = _median_end_times(runs)
    run_statistics = []
    for run in runs:
        median_end_time = median_end_times[run.dir.parent]
        if run.end_time < MIN_RELATIVE_SIMULATION_END_TIME * median_end_time:
            print(
                f"\nWARNING: skipping {run.dir}: simulation ended at {run.end_time:.3f} s "
                f"instead of ~{median_end_time:.3f} s (aborted simulation)."
            )
            continue
        statistics = extract_run(
            run, event_filter, estimator, args.coverage_window_size, args.skip_empty_runs, evaluation_dir
        )
        if statistics is not None:
            run_statistics.append(statistics)
    print(f"\nProcessed {len(run_statistics)} runs.")
    if not run_statistics:
        return

    all_statistics = pd.concat(run_statistics, ignore_index=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    all_statistics.to_csv(evaluation_dir / "statistics_run_all.csv", index=False)
    aggregate(all_statistics).to_csv(evaluation_dir / AGGREGATED_FILE_NAME, index=False)
    print(f"Saved {estimator} aggregated statistics to: {evaluation_dir}")


if __name__ == "__main__":
    main()
