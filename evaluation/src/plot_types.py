"""Plot types: what a figure shows and how it is drawn.

A plot type is a class derived from PlotType with
    options   its own option keys and their defaults (None: the option is required)
    axes      (x label, y label): "sweep" (the sweep parameter), "metric" (the
              option metric) or a fixed text; the config can override both
and two methods:
    prepare(data, figure) -> table
        All computations.  data.read(<CSV file>) returns a file of the
        evaluation directory; the table has the columns method (entry of the
        option methods) and sweep (sweep value) plus whatever draw needs.
    draw(ax, series, figure)
        Draws one axes.  Every series holds the rows of one method (and, in the
        combined layout, one sweep value) with its legend label and line style.
plot.py does everything else: options, loading, figure size and layout, axis
labels and limits, legend and saving.  figure holds the options of the figure
(e.g. figure.metric), its name, the sweep parameter and its values.

A new plot type is a class here and its entry in PLOT_TYPES; a config
selects it with type: <name>.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

from plot_style import format_value
from runs import AGGREGATED_FILE_NAME, AVAILABILITY_FILE_NAME, COVERAGE_FILE_NAME, FILTERED_EVENTS_FILE_NAME


class PlotType:
    """Base class of the plot types (see the module docstring)."""

    options: dict = {}
    axes = ("sweep", "metric")
    # Option layout: all sweep values in one axes (combined), one axes per
    # value (facets) or one figure per value (separate).  Without it, the sweep
    # values are the x axis.
    layouts = False
    # One axes per method, next to each other.
    facet_by_method = False
    # Markers on the lines of every method (otherwise only on method parts).
    markers = False

    def prepare(self, data, figure) -> pd.DataFrame:
        """Table of everything draw needs, with the columns method and sweep."""
        raise NotImplementedError

    def draw(self, ax, series, figure) -> None:
        """Draw the series of one axes."""
        raise NotImplementedError

    def legend(self, fig) -> tuple[list, list] | None:
        """Legend entries (handles, labels); None: the labeled artists of all axes (see plot_style.axes_legend)."""
        return None


def split_method(method: str) -> tuple[str, str]:
    """Method and part of an entry of the option methods, e.g. ("CENTRALIZED", "t_twr")."""
    base, _, part = method.partition(".")
    return base, part


def metric_table(data, metric: str, methods: list[str]) -> pd.DataFrame:
    """Statistics of a metric over the repetitions per method and sweep value.

    Columns method, sweep, n_runs and the aggregates of extract_statistics.py
    (mean, std, sem, min, q25, median, q75, max).  <METHOD>.<part> uses the
    metric <metric>_<part>, e.g. mean_error_t_twr.
    """
    aggregated = data.read(AGGREGATED_FILE_NAME)
    tables = []
    for method in methods:
        base, part = split_method(method)
        rows = aggregated.loc[
            (aggregated["Method"] == base) & (aggregated["metric"] == (f"{metric}_{part}" if part else metric))
        ]
        if rows.empty:
            continue
        # One row per sweep value, one column per aggregate.
        table = rows.pivot_table(
            index=["sweep", "n_runs"], columns="aggregate", values="value", aggfunc="first"
        ).reset_index()
        table.columns.name = None
        table.insert(0, "method", method)
        # Without a mean, the metric has no value at this sweep value.
        tables.append(table.dropna(subset=["mean"]))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def sample_table(data, file_name: str, column: str, methods: list[str]) -> pd.DataFrame:
    """All samples of a column per method and sweep value (column x), e.g. for an ECDF.

    <METHOD>.<part> keeps the rows with localization_method <part>; files
    without this column have no parts.
    """
    # Only the needed columns: the event files are large.
    samples = data.read(file_name, columns=[column, "localization_method"])
    tables = []
    for method in methods:
        base, part = split_method(method)
        rows = samples.loc[samples["Method"] == base]
        if part:
            if "localization_method" not in rows:
                continue
            rows = rows.loc[rows["localization_method"] == part]
        tables.append(pd.DataFrame({"method": method, "sweep": rows["sweep"], "x": rows[column]}).dropna())
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


class Line(PlotType):
    """Statistic of a metric over the repetitions versus the sweep parameter.

    Options:
        metric    metric of statistics_run_aggregated.csv, e.g. mean_coverage
        errorbar  sem (standard error), std (standard deviation), ci95 (95 %
                  confidence interval of the mean, Student's t) or none
    """

    options = {"metric": None, "errorbar": "sem"}
    markers = True
    ERRORBARS = ("sem", "std", "ci95", "none")

    def prepare(self, data, figure) -> pd.DataFrame:
        """Mean and error bar of the metric per method and sweep value."""
        if figure.errorbar not in self.ERRORBARS:
            raise SystemExit(f"{figure.name}: errorbar must be one of {', '.join(self.ERRORBARS)}")
        table = metric_table(data, figure.metric, figure.methods)
        if table.empty:
            return table
        table["y"] = table["mean"]
        # With a single repetition, std and sem are empty and missing in the table: no error bars.
        if figure.errorbar == "ci95" and "sem" in table:
            # Half width of the interval: Student's t quantile with n_runs - 1 degrees of freedom times sem.
            table["yerr"] = table["sem"] * scipy.stats.t.ppf(0.975, table["n_runs"] - 1)
        elif figure.errorbar != "none" and figure.errorbar in table:
            table["yerr"] = table[figure.errorbar]
        return table.sort_values("sweep")

    def draw(self, ax, series, figure) -> None:
        """One line per method with error bars; one x tick per sweep value."""
        for s in series:
            yerr = s.data["yerr"] if "yerr" in s.data else None
            ax.errorbar(s.data["sweep"], s.data["y"], yerr=yerr, label=s.label, **s.style)
        ax.set_xticks(sorted(set().union(*(s.data["sweep"] for s in series))))


class Ecdf(PlotType):
    """Empirical CDF of all samples of a quantity (all nodes, windows and repetitions).

    Options:
        metric  error (localization error of every event), availability (A_i
                of every node) or coverage (C(t) of every window)
        layout  combined, facets or separate (see PlotType.layouts)
    """

    options = {"metric": None, "layout": "facets"}
    axes = ("metric", "ECDF")
    layouts = True
    # metric: (file, column, x limits)
    METRICS = {
        "error": (FILTERED_EVENTS_FILE_NAME, "est_position_error", (0, 1)),
        # Starts below 0 so that the step of nodes with A_i = 0 is visible.
        "availability": (AVAILABILITY_FILE_NAME, "availability", (-0.02, 1)),
        "coverage": (COVERAGE_FILE_NAME, "coverage", (0, 1.01)),
    }

    def prepare(self, data, figure) -> pd.DataFrame:
        """Samples per method and sweep value."""
        if figure.metric not in self.METRICS:
            raise SystemExit(f"{figure.name}: metric must be one of {', '.join(self.METRICS)}")
        file_name, column, _ = self.METRICS[figure.metric]
        return sample_table(data, file_name, column, figure.methods)

    def draw(self, ax, series, figure) -> None:
        """One ECDF per series."""
        for s in series:
            style = s.style
            # An ECDF has a vertex per sample: markers only every 10 % of the line length.
            if "marker" in style:
                style = {**style, "markevery": 0.1}
            sns.ecdfplot(x=s.data["x"], label=s.label, ax=ax, **style)
        ax.set_xlim(self.METRICS[figure.metric][2])
        ax.set_ylim(0, 1.02)


class Timeseries(PlotType):
    """Coverage C(t) over time, averaged over the repetitions and smoothed.

    With a single sweep value, the legend shows the mean coverage of every
    method (time average of the unsmoothed C(t), averaged over repetitions).
    Options:
        metric     coverage (the only time series of extract_statistics.py)
        smoothing  moving-average window [s]
        layout     combined, facets or separate (see PlotType.layouts)
    """

    # smoothing 0.25 s: about 6 coverage windows and 5 % of the shortest simulation.
    options = {"metric": "coverage", "smoothing": 0.25, "layout": "facets"}
    axes = ("Time [s]", "metric")
    layouts = True

    def prepare(self, data, figure) -> pd.DataFrame:
        """Smoothed curve (x: time, y) and mean value per method and sweep value."""
        if figure.metric != "coverage":
            raise SystemExit(f"{figure.name}: metric must be coverage")
        coverage = data.read(COVERAGE_FILE_NAME)
        if coverage.empty:
            return coverage
        # Smoothing window as a number of coverage windows.
        window = round(figure.smoothing / float(coverage["coverage_window_size"].iloc[0]))
        tables = []
        for method in figure.methods:
            for value, rows in coverage.loc[coverage["Method"] == method].groupby("sweep"):
                # Mean over the repetitions per coverage window.
                curve = rows.groupby("time_bin_start")["coverage"].mean()
                tables.append(pd.DataFrame({
                    "method": method,
                    "sweep": value,
                    "x": curve.index,
                    # Centered moving average; min_periods=1 keeps the curve up to both ends.
                    "y": curve.rolling(max(1, window), center=True, min_periods=1).mean().to_numpy(),
                    # Time average per repetition, then the mean over the repetitions.
                    "mean": rows.groupby("Rep")["coverage"].mean().mean(),
                }))
        return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    def draw(self, ax, series, figure) -> None:
        """One curve per series."""
        for s in series:
            label = s.label
            if label is not None and len(figure.values) == 1:
                label += f" ($\\bar{{C}}$={s.data['mean'].iloc[0]:.2f})"
            ax.plot(s.data["x"], s.data["y"], label=label, **s.style)
        ax.set_ylim(0, 1.02)


class Opportunities(PlotType):
    """Outcome of the t-anchor opportunities (opportunities.py), one axes per method.

    Bars: mean fraction of every outcome over the repetitions (sums up to 1),
    stacked; outcomes that never occur are left out (e.g. stale seqNo for
    Centralized).  Black line: mean fraction of eligible nodes (heard all
    anchors, not busy); a random selection hits an eligible node with this probability,
    so for Centralized the green bar stays below it (not every eligible node
    gets a t-TWR position).  One bar per sweep value at equal distances.
    """

    axes = ("sweep", "Fraction of\nRanging Rounds")
    facet_by_method = True
    # Outcomes in stacking order (bottom to top) with legend label and color.
    OUTCOMES = {
        "used": ("Localized t-Anchor", "#4c9a2a"),
        "missed_no_sender": ("Not sent", "#f0a030"),
        "missed_failed": ("Failed t-TWR", "#d0473c"),
        "missed_stale": ("Stale seqNo", "#8c6bb1"),
        "no_eligible_node": ("No eligible node", "#a0a0a0"),
    }
    ELIGIBLE_LABEL = "Eligible nodes"

    def prepare(self, data, figure) -> pd.DataFrame:
        """Mean fraction per outcome (or "eligible"), method and sweep value."""
        tables = []
        for outcome in [*self.OUTCOMES, "eligible"]:
            metric = "mean_eligible_fraction" if outcome == "eligible" else f"opportunity_{outcome}"
            table = metric_table(data, metric, figure.methods)
            # None of the methods has opportunities: no figure.
            if table.empty:
                return table
            tables.append(table[["method", "sweep", "mean"]].assign(outcome=outcome))
        table = pd.concat(tables, ignore_index=True)
        # Outcomes without a nonzero mean in any method and sweep value.
        never = [
            outcome for outcome in self.OUTCOMES
            if not (table.loc[table["outcome"] == outcome, "mean"] > 0).any()
        ]
        return table.loc[~table["outcome"].isin(never)]

    def draw(self, ax, series, figure) -> None:
        """Stacked outcome bars and the eligible fraction of the method of this axes."""
        # One axes per method: a single series.
        rows = series[0].data
        values = sorted(rows["sweep"].unique())
        # Equidistant bar positions, labeled with the sweep values below.
        positions = np.arange(len(values))
        shown = [outcome for outcome in self.OUTCOMES if outcome in set(rows["outcome"])]
        # Top of the stack so far, per bar.
        bottom = np.zeros(len(values))
        for outcome in shown:
            label, color = self.OUTCOMES[outcome]
            fractions = self._means(rows, outcome, values).fillna(0.0).to_numpy()
            ax.bar(positions, fractions, bottom=bottom, width=0.7, color=color, label=label)
            bottom += fractions
        ax.plot(
            positions, self._means(rows, "eligible", values).to_numpy(),
            color="black", marker="o", label=self.ELIGIBLE_LABEL,
        )
        ax.set_ylim(0, 1.02)
        ax.set_xticks(positions, [format_value(value) for value in values])

    def legend(self, fig) -> tuple[list, list]:
        """Outcomes in stacking order, then the eligible fraction."""
        handles, labels = fig.axes[0].get_legend_handles_labels()
        # Stable sort: the eligible line moves to the end, the outcomes keep their order.
        order = sorted(range(len(labels)), key=lambda i: labels[i] == self.ELIGIBLE_LABEL)
        return [handles[i] for i in order], [labels[i] for i in order]

    @staticmethod
    def _means(rows: pd.DataFrame, outcome: str, values: list) -> pd.Series:
        """Mean fraction of one outcome per sweep value (NaN where missing)."""
        return rows.loc[rows["outcome"] == outcome].set_index("sweep")["mean"].reindex(values)


# Plot types by the name used in the configs (type: <name>).
PLOT_TYPES = {
    "line": Line,
    "ecdf": Ecdf,
    "timeseries": Timeseries,
    "opportunities": Opportunities,
}
