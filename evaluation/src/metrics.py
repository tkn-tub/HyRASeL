"""Metrics of the filtered localization events of one run.

A metric is one Statistic: its name, value, sample size n and unit.  The run
metrics treat the events of all evaluated localization methods as one stream
of node positions; they also hold the error and interval statistics of every
localization method on its own (suffix _<method>, e.g. mean_error_t_twr) and
the t-anchor opportunity metrics.  A new metric is one more Statistic in
run_statistics or node_statistics.

Coverage C(t) is the fraction of all simulated nodes localized in a window;
the availability A_i of node i is the fraction of windows in which it was
localized.  The windows cover t = 0 up to the end of the simulation, so all
methods share the same time axis and the same number of nodes.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

import opportunities

# Values of localization_method, in the order of their metrics.
LOCALIZATION_METHODS = ("self_loc", "t_twr", "twr")

# Descriptive statistics of a sample in output order; NaN values are ignored.
SAMPLE_STATISTICS = {
    "mean": lambda sample: sample.mean(),
    "median": lambda sample: sample.median(),
    "std": lambda sample: sample.std(),
    "min": lambda sample: sample.min(),
    "max": lambda sample: sample.max(),
    "q25": lambda sample: sample.quantile(0.25),
    "q75": lambda sample: sample.quantile(0.75),
    "q95": lambda sample: sample.quantile(0.95),
}


class Statistic(NamedTuple):
    """One metric; n is its number of samples (NaN for counts)."""

    metric: str
    value: float
    n: float
    unit: str


def sample_statistics(sample: pd.Series, name: str, unit: str) -> list[Statistic]:
    """mean_<name>, median_<name>, ... of a sample (SAMPLE_STATISTICS); n is the number of values."""
    n = sample.notna().sum()
    statistics = []
    for statistic, function in SAMPLE_STATISTICS.items():
        statistics.append(Statistic(f"{statistic}_{name}", function(sample), n, unit))
    return statistics


def error_statistics(errors: pd.Series, name: str = "error") -> list[Statistic]:
    """Sample statistics and RMSE of position errors."""
    rmse = np.sqrt((errors ** 2).mean())
    return [*sample_statistics(errors, name, "m"), Statistic(f"rmse_{name}", rmse, errors.notna().sum(), "m")]


def add_localization_interval(events: pd.DataFrame) -> pd.DataFrame:
    """Events sorted by node and time, with the time since the previous event of the node [s]."""
    events = events.sort_values(by=["NodeId", "event_time"])
    # Difference within each node; NaN for its first event.
    events["localization_interval"] = events.groupby("NodeId")["event_time"].diff()
    return events


def _num_windows(window_size: float, end_time: float) -> int:
    """Number of windows from t = 0 up to end_time."""
    # +1: the window that contains end_time.
    return int(np.floor(end_time / window_size)) + 1


def _window_index(events: pd.DataFrame, window_size: float) -> pd.Series:
    """Window of every event."""
    return np.floor(events["event_time"] / window_size).astype(int)


def coverage_table(events: pd.DataFrame, window_size: float, num_nodes: int, end_time: float) -> pd.DataFrame:
    """Coverage of every window: fraction of the num_nodes simulated nodes with an event in it."""
    num_windows = _num_windows(window_size, end_time)
    windows = pd.DataFrame({"window": _window_index(events, window_size), "NodeId": events["NodeId"]})
    # Distinct nodes per window; windows without events get 0.
    localized_nodes = windows.drop_duplicates().groupby("window").size()
    localized_nodes = localized_nodes.reindex(range(num_windows), fill_value=0).to_numpy()
    window_starts = np.arange(num_windows) * window_size
    return pd.DataFrame({
        "coverage_window_size": window_size,
        "time_bin_start": window_starts,
        "time_bin_end": window_starts + window_size,
        "localized_nodes": localized_nodes,
        "total_nodes": num_nodes,
        "coverage": localized_nodes / num_nodes,
    })


def availability_table(events: pd.DataFrame, window_size: float, num_nodes: int, end_time: float) -> pd.DataFrame:
    """Availability of every simulated node: fraction of the windows with an event of the node.

    Nodes without events get availability 0 and NodeId NaN, so the mean
    availability equals the mean coverage.
    """
    num_windows = _num_windows(window_size, end_time)
    windows = pd.DataFrame({"NodeId": events["NodeId"], "window": _window_index(events, window_size)})
    # Distinct windows per localized node, within the windows of coverage_table.
    localized_windows = windows.loc[windows["window"] < num_windows].drop_duplicates().groupby("NodeId").size()
    num_unlocalized_nodes = num_nodes - len(localized_windows)
    if num_unlocalized_nodes < 0:
        raise ValueError(f"{len(localized_windows)} localized nodes but only {num_nodes} simulated nodes")
    return pd.DataFrame({
        "coverage_window_size": window_size,
        "NodeId": [*localized_windows.index, *[np.nan] * num_unlocalized_nodes],
        "availability": [*(localized_windows.to_numpy() / num_windows), *[0.0] * num_unlocalized_nodes],
    })


def run_statistics(
    selected: pd.DataFrame,
    filtered: pd.DataFrame,
    coverage: pd.DataFrame,
    availability: pd.DataFrame,
    num_nodes: int,
    end_time: float,
    opportunity_table: pd.DataFrame | None,
) -> list[Statistic]:
    """Metrics of one run.

    selected:           events of the evaluated localization methods before filtering
    filtered:           filtered events with localization_interval (add_localization_interval)
    coverage:           coverage_table of the filtered events
    availability:       availability_table of the filtered events
    num_nodes:          simulated nodes
    end_time:           [s] end of the simulation
    opportunity_table:  t-anchor opportunities of the run (opportunities.py) or None
    """
    errors = filtered["est_position_error"]
    coverage_values = coverage["coverage"]
    availability_values = availability["availability"]
    num_availability_values = len(availability_values)
    statistics = [
        *error_statistics(errors),
        *sample_statistics(filtered["localization_interval"], "localization_interval", "s"),
        *sample_statistics(coverage_values, "coverage", "fraction"),
        # The mean availability equals mean_coverage.
        Statistic("q10_availability", availability_values.quantile(0.1), num_availability_values, "fraction"),
        Statistic("median_availability", availability_values.median(), num_availability_values, "fraction"),
        Statistic("q90_availability", availability_values.quantile(0.9), num_availability_values, "fraction"),
        Statistic("num_events_before_filtering", len(selected), np.nan, "count"),
        Statistic("num_events_after_filtering", len(filtered), np.nan, "count"),
        Statistic("num_events_removed_by_filtering", len(selected) - len(filtered), np.nan, "count"),
        Statistic("num_nodes", filtered["NodeId"].nunique(), np.nan, "count"),
        # n is the number of coverage values, as for the coverage statistics.
        Statistic("num_coverage_windows", len(coverage), coverage_values.notna().sum(), "count"),
        # Filtered position updates per simulated node and second: update rate and coverage in one number.
        Statistic("update_rate", len(filtered) / (num_nodes * end_time), np.nan, "1/s"),
    ]

    # Every localization method of the run on its own; the interval is the one between its own events.
    run_methods = set(selected["localization_method"])
    for method in LOCALIZATION_METHODS:
        if method not in run_methods:
            continue
        method_events = add_localization_interval(filtered.loc[filtered["localization_method"] == method])
        statistics += error_statistics(method_events["est_position_error"], f"error_{method}")
        interval_name = f"localization_interval_{method}"
        statistics += sample_statistics(method_events["localization_interval"], interval_name, "s")

    # Only runs of opportunities.SUPPORTED_METHODS have an opportunity table.
    if opportunity_table is not None:
        for metric, value in opportunities.opportunity_metrics(opportunity_table).items():
            unit = "count" if metric == "num_opportunities" else "fraction"
            statistics.append(Statistic(metric, value, len(opportunity_table), unit))
    return statistics


def node_statistics(filtered: pd.DataFrame) -> pd.DataFrame:
    """Error and interval statistics of every node: columns NodeId and those of Statistic."""
    rows = []
    for node_id, node_events in filtered.groupby("NodeId"):
        statistics = [
            *error_statistics(node_events["est_position_error"]),
            *sample_statistics(node_events["localization_interval"], "localization_interval", "s"),
            Statistic("num_events_after_filtering", len(node_events), np.nan, "count"),
        ]
        # Long format: one row per node and metric.
        for statistic in statistics:
            rows.append((node_id, *statistic))
    return pd.DataFrame(rows, columns=["NodeId", *Statistic._fields])
