"""Figures of the statistics written by extract_statistics.py.

Reads one evaluation directory and writes the figures to its figures/ folder.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


# -----------------------------------------------------------------------------
# Settings (set from the command line in main)
# -----------------------------------------------------------------------------

RUNS_DIR_NAME = "runs"
SHOW_FIGURES = False
SAVE_FIGURES = True
FIGURE_EXTENSION = None
# Stable file names (overwritten on every run) unless --timestamp is given.
TIMESTAMP_FIGURES = False

# All plots that can be selected with --plots.
PLOT_NAMES = (
    "error",
    "rmse",
    "interval",
    "coverage",
    # Position updates per simulated node and second (log axis): combines
    # update interval and coverage, e.g. for the scaling with the number of nodes.
    "update_rate",
    # Median / 10 % / 90 % quantile of A_i, one figure each.
    "availability",
    "availability_q10",
    "availability_q90",
    "availability_quantiles",
    # 10 % quantile, median and 90 % quantile of A_i in three panels.
    "availability_deciles",
    "error_boxplot",
    "node_error_boxplot",
    "error_ecdf",
    "coverage_time",
    "availability_cdf",
    "availability_time",
    "error_vs_coverage",
    "interval_vs_coverage",
    "coverage_error_facet",
    # Error / localization interval of HyRASeL and of its v-TWR part alone
    # (metrics <statistic>_v_twr of extract_statistics.py).
    "error_by_technique",
    "interval_by_technique",
    # v-anchor opportunities of Centralized/Distributed (opportunities.py).
    "opportunities",
    "opportunities_time",
    # Table (CSV + Markdown) of the opportunity percentages for the text.
    "opportunity_table",
)
# Plots that need the (large) filtered event files.
EVENT_PLOTS = {"error_ecdf", "availability_time", "coverage_error_facet"}
EVENT_COLUMNS = {
    "Config", "Rep", "Method", "NodeId", "event_time", "est_position_error", "localization_method"
}


# -----------------------------------------------------------------------------
# Plot labels and global style
# -----------------------------------------------------------------------------

# y labels longer than a single-panel axes (ROW_HEIGHT) are split into two lines.
ACCURACY_DESCRIPTOR = "Localization Error [m]"
COVERAGE_DESCRIPTOR = "Coverage Fraction"
MEAN_COVERAGE_DESCRIPTOR = "Mean Coverage"
AVAILABILITY_DESCRIPTOR = "Node Availability"
MEDIAN_AVAILABILITY_DESCRIPTOR = "Median Node Availability"
Q10_AVAILABILITY_DESCRIPTOR = "10 % Quantile of\nNode Availability"
Q90_AVAILABILITY_DESCRIPTOR = "90 % Quantile of\nNode Availability"
TIME_DESCRIPTOR = "Time [s]"
INTER_UPDATE_TIME_DESCRIPTOR = "Inter-Update Interval [s]"
NUM_NODES_DESCRIPTOR = "Number of Nodes"
NUM_SLOTS_DESCRIPTOR = "Number of v-Anchor Slots"
DENSITY_DESCRIPTOR = "Inter-Anchor Distance [m]"
OPPORTUNITY_DESCRIPTOR = "Fraction of\nv-Anchor Opportunities"
UPDATE_RATE_DESCRIPTOR = "Updates per Node\nand Second [1/s]"
CDF_DESCRIPTOR = "CDF"

# IEEE two-column layout: body text 10 pt, captions 8 pt.  Figures are saved
# with exactly these widths (no bbox_inches="tight"), so the font sizes are
# true sizes when the figure is included with width=\columnwidth / \textwidth.
FONT_SIZE = 9        # axis labels, legend, titles
TICK_FONT_SIZE = 8   # tick labels, annotations (never below the caption size)
COL_WIDTH = 3.5      # \columnwidth [in]
COL_WIDTH_2 = 7.16   # \textwidth [in]
ROW_HEIGHT = 1.9     # single-panel figure including the legend above [in]
FACET_HEIGHT = 0.95  # per facet of the stacked facet figures [in]

# Mean ± SEM over repetitions versus the sweep parameter.
# plot name: (metric, y label, logarithmic y axis)
LINE_PLOTS = {
    "error": ("mean_error", ACCURACY_DESCRIPTOR, False),
    "rmse": ("rmse_error", "RMSE [m]", False),
    "interval": ("mean_localization_interval", INTER_UPDATE_TIME_DESCRIPTOR, False),
    "coverage": ("mean_coverage", MEAN_COVERAGE_DESCRIPTOR, False),
    "availability": ("median_availability", MEDIAN_AVAILABILITY_DESCRIPTOR, False),
    "availability_q10": ("q10_availability", Q10_AVAILABILITY_DESCRIPTOR, False),
    "availability_q90": ("q90_availability", Q90_AVAILABILITY_DESCRIPTOR, False),
    "update_rate": ("update_rate", UPDATE_RATE_DESCRIPTOR, True),
}

# HyRASeL and its v-TWR part as separate lines (previous Figure 04a/b).
# plot name: (metric, y label, logarithmic y axis)
TECHNIQUE_PLOTS = {
    "error_by_technique": ("mean_error", ACCURACY_DESCRIPTOR, False),
    "interval_by_technique": ("mean_localization_interval", INTER_UPDATE_TIME_DESCRIPTOR, True),
}

# Plots whose figure width can be set with --width.
WIDTH_PLOTS = [*LINE_PLOTS, *TECHNIQUE_PLOTS]

# set_theme resets the rcParams, so it comes first.
sns.set_theme(style="whitegrid")
mpl.rcParams.update({
    "font.family": "sans-serif",
    # Liberation Sans has the metrics of Arial; the fallback DejaVu Sans is
    # considerably wider.
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    # Math text (e.g. $\\bar{C}$) in the same font.
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "figure.labelsize": FONT_SIZE,
    "xtick.labelsize": TICK_FONT_SIZE,
    "ytick.labelsize": TICK_FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "legend.title_fontsize": FONT_SIZE,
    # Compact spacing.
    "axes.titlepad": 2,
    "axes.labelpad": 2,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "errorbar.capsize": 2,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0,
    "legend.labelspacing": 0.2,
    "legend.borderpad": 0.1,
    "legend.borderaxespad": 0.2,
    # Layout fills the figure up to a 1 pt margin (no dead white border).
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 1 / 72,
    "figure.constrained_layout.w_pad": 1 / 72,
    "figure.constrained_layout.hspace": 0.02,
    "figure.constrained_layout.wspace": 0.02,
    # Embedded TrueType instead of Type 3 fonts (IEEE PDF eXpress).
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Outcomes of a v-anchor opportunity (see opportunities.py) with legend label
# and color.  The order is the stacking order of the bars, bottom to top.
OPPORTUNITY_OUTCOMES = {
    "used": ("Used", "#4c9a2a"),
    "missed_no_sender": ("Missed: no transmission", "#f0a030"),
    "missed_failed": ("Missed: failed contention", "#d0473c"),
    "missed_stale": ("Missed: stale seqNo", "#8c6bb1"),
    "no_eligible_node": ("No eligible node", "#a0a0a0"),
}
# Label of missed_no_sender if it is the only missed outcome shown (outcomes
# that never occur are not plotted, e.g. failed contention for Centralized).
MISSED_LABEL = "Missed"

# Suffix of the v-TWR part of a method in --methods / --method-labels, e.g.
# CENTRALIZED.v_twr: the <statistic>_v_twr metrics of extract_statistics.py
# (error_by_technique, interval_by_technique) or the v_twr localization events
# (error_ecdf).
VTWR_SUFFIX = ".v_twr"

# Fixed style (index into color palette, line styles and markers) per method,
# so a method looks the same in all figures, whatever its label and whichever
# methods are selected.  The v-TWR part of a method takes its color.
STYLE_INDEX = {"HYPERBOLIC_SELF_LOCALIZATION": 0, "CENTRALIZED": 1, "DISTRIBUTED": 2, "TWR": 4}
STYLE_ALIASES = {"SELF_LOCALIZATION": "HYPERBOLIC_SELF_LOCALIZATION"}

# Plotted methods in legend order and their labels (default: the method name);
# set from --methods and --method-labels.
PLOT_METHODS = []
METHOD_LABELS = {}

SWEEP_PARAMETERS = {
    "numSlots": {
        "label": NUM_SLOTS_DESCRIPTOR,
        "figure_prefix": "",
        # None: one tick per simulated value (the batches sweep 2-8 or 2-6).
        "xticks": None,
    },
    "numNodes": {
        "label": NUM_NODES_DESCRIPTOR,
        "figure_prefix": "",
        "xticks": [4, 9, 16, 25, 36, 49],
    },
    "interAnchorDist": {
        "label": DENSITY_DESCRIPTOR,
        "figure_prefix": "",
        "xticks": [130, 165, 200, 235, 270],
    },
    "speed": {
        "label": "Speed [m/s]",
        "figure_prefix": "",
        "xticks": [2, 5, 10, 25, 50],
    },
    "loss": {
        "label": "Message Loss Probability",
        "figure_prefix": "",
        "xticks": [0, 0.1, 0.2, 0.3, 0.5],
    },
    "anchorNodeLoss": {
        "label": "Anchor-Node Message Loss Probability",
        "figure_prefix": "",
        "xticks": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    },
    "nodeNodeLoss": {
        "label": "Node-Node Message Loss Probability",
        "figure_prefix": "",
        "xticks": [0, 0.2, 0.4, 0.6, 0.8],
    },
}

# Evaluations without sweep parameter (e.g. the baseline) use the Config
# column as their only "parameter value": every plot then works unchanged with
# one facet / one x position, and the facet title names the config.
NO_SWEEP_PARAMETER = "Config"
NO_SWEEP_PARAMETER_CONFIG = {
    "label": "Configuration",
    "figure_prefix": "",
    "xticks": None,
}


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "evaluation_dir",
        type=Path,
        help="Evaluation directory <config>/<estimator>/<filter_profile>",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Figure directory (default: <evaluation_dir>/figures)",
    )
    # The values of the following options are set in configs/defaults.yml
    # and configs/<Config>.yml.
    parser.add_argument(
        "--format",
        choices=("pdf", "png", "svg"),
        required=True,
        help="Figure format",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=PLOT_NAMES,
        required=True,
        metavar="PLOT",
        help="Plots to generate. Choices: " + ", ".join(PLOT_NAMES),
    )
    parser.add_argument(
        "--width",
        nargs="+",
        default=[],
        metavar="PLOT=FRACTION",
        help=(
            "Figure width of single plots as a fraction of the column width, "
            "e.g. coverage=0.5 availability_q10=0.5 for two plots next to each "
            "other in one column (include them with width=0.5\\columnwidth). "
            "Only for the line plots: " + ", ".join(WIDTH_PLOTS)
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        metavar="METHOD",
        help=(
            "Methods to plot, in legend order; <METHOD>" + VTWR_SUFFIX + " is the "
            "v-TWR part of a method (only error_by_technique, interval_by_technique, error_ecdf)"
        ),
    )
    parser.add_argument(
        "--method-labels",
        nargs="+",
        default=[],
        metavar="METHOD=LABEL",
        help="Display names of the methods, e.g. CENTRALIZED=HyRASeL (default: the method name)",
    )
    parser.add_argument(
        "--smoothing-window",
        type=float,
        required=True,
        help="Moving-average window in seconds for coverage-over-time plots",
    )
    parser.add_argument(
        "--ecdf-mode",
        choices=("combined", "separate", "both"),
        required=True,
        help="Put parameter values in one ECDF or separate figures",
    )
    parser.add_argument(
        "--show-fliers",
        action="store_true",
        help="Show outliers in run-level and node-level boxplots",
    )
    parser.add_argument(
        "--no-errorbars",
        action="store_true",
        help="Do not show SEM error bars in aggregated line plots",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Prefix figure names with YYMMDD_HHMM_ instead of overwriting them",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after saving them",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save figures (normally used together with --show)",
    )
    args = parser.parse_args()

    if not args.evaluation_dir.is_dir():
        parser.error(f"evaluation_dir is not a directory: {args.evaluation_dir}")
    if args.no_save and not args.show:
        parser.error("--no-save requires --show; otherwise no output is produced")
    if args.smoothing_window <= 0:
        parser.error("--smoothing-window must be greater than zero")
    method_labels = {}
    for entry in args.method_labels:
        method, separator, label = entry.partition("=")
        if not separator or not method:
            parser.error(f"--method-labels expects METHOD=LABEL, got: {entry}")
        method_labels[method] = label
    args.method_labels = method_labels
    widths = {}
    for entry in args.width:
        plot_name, separator, fraction = entry.partition("=")
        if not separator or plot_name not in WIDTH_PLOTS:
            parser.error(
                f"--width expects PLOT=FRACTION with PLOT one of {', '.join(WIDTH_PLOTS)}, got: {entry}"
            )
        try:
            widths[plot_name] = float(fraction)
        except ValueError:
            parser.error(f"--width: FRACTION must be a number, got: {entry}")
        if widths[plot_name] <= 0:
            parser.error(f"--width: FRACTION must be greater than zero, got: {entry}")
    args.width = widths
    return args


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def build_figure_dir(evaluation_dir: Path) -> Path:
    """Return the output directory for all generated figures."""
    return evaluation_dir / "figures"


def build_output_path(figure_dir: Path, figure_name: str) -> Path:
    """Build the figure output path; with --timestamp prefixed by YYMMDD_HHMM_."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    if TIMESTAMP_FIGURES:
        figure_name = f"{datetime.now().strftime('%y%m%d_%H%M')}_{figure_name}"
    return figure_dir / f"{figure_name}.{FIGURE_EXTENSION}"


def fit_to_figure_width(fig: plt.Figure, text: mpl.text.Text) -> None:
    """Keep a label that exceeds the figure (e.g. narrow figures, --width) inside it.

    The label is split into two lines (at the space nearest the middle); an
    axis label that still exceeds (it is centered on its axes) is shifted
    inward.  Constrained layout does not widen the margins for this.
    """
    renderer = fig.canvas.get_renderer()

    def overflow() -> tuple[float, float]:
        fig.canvas.draw()
        extent = text.get_window_extent(renderer)
        return fig.bbox.x0 - extent.x0, extent.x1 - fig.bbox.x1

    label = text.get_text()
    if not text.get_visible() or not label or max(overflow()) <= 0:
        return
    if "\n" not in label and " " in label:
        split = min(
            (index for index, char in enumerate(label) if char == " "),
            key=lambda index: abs(index - len(label) / 2),
        )
        text.set_text(label[:split] + "\n" + label[split + 1:])
    left, right = overflow()
    axes = next((ax for ax in fig.axes if ax.xaxis.label is text), None)
    if axes is not None and (left > 0) != (right > 0):
        # x of an axis label is in axes coordinates.
        shift = left if left > 0 else -right
        text.set_x(text.get_position()[0] + shift / axes.bbox.width)


def save_or_show(fig: plt.Figure, figure_dir: Path, figure_name: str) -> None:
    """Save and/or show a figure based on global script settings."""
    for text in [fig._supxlabel] + [ax.xaxis.label for ax in fig.axes]:
        if text is not None:
            fit_to_figure_width(fig, text)

    if SAVE_FIGURES:
        # No bbox_inches="tight": the file keeps the exact figure width (see
        # COL_WIDTH); constrained layout already removes the white border.
        output_path = build_output_path(figure_dir, figure_name)
        fig.savefig(output_path, dpi=300)
        # Labels longer than their axes (e.g. a y label of a flat panel) are
        # cut off at the figure border instead.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = [fig._supxlabel, fig._supylabel] + [
            text for ax in fig.axes for text in (ax.xaxis.label, ax.yaxis.label, ax.title)
        ]
        for text in texts:
            if text is None or not text.get_visible() or not text.get_text():
                continue
            extent = text.get_window_extent(renderer)
            overflow = max(-extent.x0, -extent.y0, extent.x1 - fig.bbox.x1,
                           extent.y1 - fig.bbox.y1) * 72 / fig.dpi
            if overflow > 0.5:
                print(f"Warning: {output_path.name}: \"{text.get_text()}\" exceeds "
                      f"the figure by {overflow:.0f} pt.")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def apply_axis_options(
    ax: plt.Axes,
    xlabel: str = "",
    ylabel: str = "",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
    xlog: bool = False,
    ylog: bool = False,
) -> None:
    """Apply shared axis formatting."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xticks is not None:
        # Only ticks within the plotted values; set_xticks would extend the axis.
        x_min, x_max = ax.dataLim.intervalx
        tolerance = 1e-9 * max(1.0, abs(x_min), abs(x_max))
        ax.set_xticks([tick for tick in xticks if x_min - tolerance <= tick <= x_max + tolerance])
    if yticks is not None:
        ax.set_yticks(yticks)


def legend_row_order(items: list, ncol: int) -> list:
    """Reorder items so that a legend (filled column by column) reads row by row."""
    nrows, num_large_columns = divmod(len(items), ncol)
    rows_per_column = [nrows + 1] * num_large_columns + [nrows] * (ncol - num_large_columns)
    return [
        items[row * ncol + column]
        for column, column_rows in enumerate(rows_per_column)
        for row in range(column_rows)
    ]


def add_figure_legend(
    fig: plt.Figure,
    handles: list | None = None,
    labels: list[str] | None = None,
) -> None:
    """One legend above all axes, the same in every figure.

    Without handles, the labeled artists of the first axes are used; legends
    of the axes are removed.  The legend gets as few rows as fit into the
    figure width, the entries read row by row (first entry top left).
    """
    if handles is None:
        handles, labels = fig.axes[0].get_legend_handles_labels()
        axes_legend = fig.axes[0].get_legend()
        if not handles and axes_legend is not None:
            # e.g. seaborn boxplot: the hue entries only exist in its legend.
            handles = list(axes_legend.legend_handles)
            labels = [text.get_text() for text in axes_legend.get_texts()]
    for ax in fig.axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    if not handles:
        return

    def make_legend(ncol: int):
        return fig.legend(
            legend_row_order(list(handles), ncol),
            legend_row_order(list(labels), ncol),
            loc="outside upper center",
            ncol=ncol,
        )

    renderer = fig.canvas.get_renderer()
    for max_ncol in range(len(handles), 0, -1):
        legend = make_legend(max_ncol)
        fits = legend.get_window_extent(renderer).width <= fig.bbox.width
        legend.remove()
        if fits or max_ncol == 1:
            break
    # Same number of rows, entries spread evenly over the columns.
    nrows = -(-len(handles) // max_ncol)
    make_legend(-(-len(handles) // nrows))


def set_facet_ylabel(fig: plt.Figure, axes, ylabel: str) -> None:
    """y label of stacked facets: once for the figure, it is longer than one facet."""
    if len(axes) > 1:
        fig.supylabel(ylabel)
    else:
        axes[0].set_ylabel(ylabel)


def format_parameter_value(value) -> str:
    """Short label of a parameter value: numbers without trailing zeros, text as is (Config)."""
    return f"{value:g}" if isinstance(value, (int, float, np.number)) else str(value)


def facet_title(parameter_column: str, parameter_value) -> str:
    """Title of one parameter facet, e.g. "Inter-Anchor Distance: 130 m"; none without sweep."""
    if parameter_column == NO_SWEEP_PARAMETER:
        return ""
    label = SWEEP_PARAMETERS.get(parameter_column, {}).get("label", parameter_column)
    name, _, unit = label.partition(" [")
    unit = unit.removesuffix("]")
    return f"{name}: {format_parameter_value(parameter_value)}" + (f" {unit}" if unit else "")


def display_label(method: str) -> str:
    """Display label of a method (entry of --methods)."""
    return METHOD_LABELS.get(method, method)


def add_method_labels(dataframe: pd.DataFrame, method_suffix: str = "") -> pd.DataFrame:
    """Keep the plotted methods (PLOT_METHODS) and add their display labels.

    method_suffix selects a part of the methods, e.g. VTWR_SUFFIX for the
    <statistic>_v_twr metrics.
    """
    if dataframe.empty or "Method" not in dataframe.columns:
        return dataframe

    methods = dataframe["Method"].astype(str) + method_suffix
    selected = methods.isin(PLOT_METHODS)
    dataframe = dataframe.loc[selected].copy()
    dataframe["MethodLabel"] = methods[selected].map(display_label)
    return dataframe


def label_method(label: str) -> str:
    """Method (entry of --methods) of a display label; unknown labels are their own method."""
    for method in PLOT_METHODS:
        if display_label(method) == label:
            return method
    return label


def ordered_labels(labels) -> list[str]:
    """Labels in the order of PLOT_METHODS (= legend order), unknown ones last."""
    def sort_key(label: str) -> tuple[int, str]:
        method = label_method(label)
        return (PLOT_METHODS.index(method) if method in PLOT_METHODS else len(PLOT_METHODS), label)
    return sorted(set(labels), key=sort_key)


def method_groups(dataframe: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Rows of every method (label, rows) in legend order."""
    groups = dict(tuple(dataframe.groupby("MethodLabel")))
    return [(label, groups[label]) for label in ordered_labels(groups)]


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------

def load_run_csvs(
    evaluation_dir: Path,
    file_name: str,
    usecols: set[str] | None = None,
) -> pd.DataFrame:
    """Load one CSV type from all runs (<evaluation_dir>/runs/<Parameters>/<repX>/).

    usecols optionally restricts the loaded columns; sweep-parameter columns
    are always kept.
    """
    dataframes = []
    columns_filter = (
        None
        if usecols is None
        else lambda column: column in usecols or column in SWEEP_PARAMETERS
    )

    for file_path in sorted((evaluation_dir / RUNS_DIR_NAME).rglob(file_name)):
        try:
            dataframe = pd.read_csv(file_path, usecols=columns_filter)
        except pd.errors.EmptyDataError:
            print(f"Warning: skipping empty CSV: {file_path}")
            continue
        dataframe["source_file"] = str(file_path)
        dataframes.append(dataframe)

    if not dataframes:
        print(f"Warning: no {file_name} files found below {evaluation_dir}.")
        return pd.DataFrame()

    return pd.concat(dataframes, ignore_index=True)


def load_data(
    evaluation_dir: Path,
    plots: set[str] = set(PLOT_NAMES),
) -> dict[str, pd.DataFrame]:
    """Load the exported CSVs needed for the selected plots."""
    aggregated_path = evaluation_dir / "statistics_run_aggregated.csv"

    def load_if(needed: bool, file_name: str, usecols: set[str] | None = None) -> pd.DataFrame:
        if not needed:
            return pd.DataFrame()
        return load_run_csvs(evaluation_dir, file_name, usecols)

    data = {
        # Only the columns used by the plots; the full files are very large.
        "events": load_if(
            bool(plots & EVENT_PLOTS), "localization_events_filtered.csv", EVENT_COLUMNS
        ),
        "coverage": load_if(
            bool(plots & {"coverage_time", "availability_time", "coverage_error_facet"}),
            "coverage.csv",
        ),
        "availability": load_if("availability_cdf" in plots, "availability.csv"),
        "opportunities": load_if("opportunities_time" in plots, "opportunities.csv"),
        "statistics_run": load_if("error_boxplot" in plots, "statistics_run.csv"),
        "statistics_node": load_if("node_error_boxplot" in plots, "statistics_node.csv"),
        "statistics_run_aggregated": pd.DataFrame(),
    }

    if aggregated_path.exists():
        data["statistics_run_aggregated"] = pd.read_csv(aggregated_path)
    else:
        print(f"Warning: {aggregated_path} not found. Line plots with SEM will be skipped.")

    return data


# -----------------------------------------------------------------------------
# Statistics selection helpers
# -----------------------------------------------------------------------------

def metric_with_mean_and_sem(
    aggregated_statistics_df: pd.DataFrame,
    metric: str,
    parameter_column: str,
    method_suffix: str = "",
) -> pd.DataFrame:
    """Return one metric of the plotted methods with mean and SEM in wide form.

    method_suffix: see add_method_labels.
    """
    if aggregated_statistics_df.empty or parameter_column not in aggregated_statistics_df.columns:
        return pd.DataFrame()

    metric_df = add_method_labels(
        aggregated_statistics_df.loc[aggregated_statistics_df["metric"] == metric],
        method_suffix,
    )

    if metric_df.empty:
        return pd.DataFrame()

    # dict.fromkeys removes the duplicate if the parameter column is Config
    # (evaluations without sweep parameter).
    index_columns = [
        column
        for column in dict.fromkeys(
            ["Config", "Method", "MethodLabel", parameter_column, "unit", "n_runs"]
        )
        if column in metric_df.columns
    ]

    plot_df = (
        metric_df
        .pivot_table(
            index=index_columns,
            columns="aggregate",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    plot_df.columns.name = None

    return plot_df


def run_metric(statistics_run_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Select one run-level metric from statistics_run.csv."""
    if statistics_run_df.empty:
        return pd.DataFrame()

    return add_method_labels(
        statistics_run_df.loc[statistics_run_df["metric"] == metric].copy()
    )


# -----------------------------------------------------------------------------
# Plot functions
# -----------------------------------------------------------------------------

def plot_metric_line(
    aggregated_statistics_df: pd.DataFrame,
    parameter_column: str,
    metric: str,
    xlabel: str,
    ylabel: str,
    figure_dir: Path,
    figure_name: str,
    xticks: list[float] | None = None,
    show_errorbars: bool = True,
    fig_width: float = COL_WIDTH,
    fig_height: float = ROW_HEIGHT,
    ylog: bool = False,
) -> None:
    """Plot aggregated run metrics with SEM error bars."""
    plot_df = metric_with_mean_and_sem(aggregated_statistics_df, metric, parameter_column)

    if plot_df.empty or "mean" not in plot_df.columns:
        print(f"Skipping {figure_name}: no data for metric {metric}.")
        return

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    styles = method_styles(list(plot_df["MethodLabel"].unique()), with_marker=True)

    for method_label, method_df in method_groups(plot_df):
        method_df = method_df.sort_values(parameter_column)
        y_error = (
            method_df["sem"]
            if show_errorbars and "sem" in method_df.columns
            else None
        )

        ax.errorbar(
            method_df[parameter_column],
            method_df["mean"],
            yerr=y_error,
            label=method_label,
            **styles[method_label],
        )

    if xticks is None:
        xticks = sorted(plot_df[parameter_column].unique())
    apply_axis_options(ax, xlabel=xlabel, ylabel=ylabel, xticks=xticks, ylog=ylog)
    add_figure_legend(fig)
    save_or_show(fig, figure_dir, figure_name)


def plot_metric_boxplot(
    statistics_df: pd.DataFrame,
    parameter_column: str,
    metric: str,
    xlabel: str,
    ylabel: str,
    figure_dir: Path,
    figure_name: str,
    showfliers: bool = False,
    fig_width: float = COL_WIDTH,
    fig_height: float = ROW_HEIGHT,
) -> None:
    """Plot run-level or node-level metric distributions."""
    plot_df = run_metric(statistics_df, metric)

    if plot_df.empty or parameter_column not in plot_df.columns:
        print(f"Skipping {figure_name}: no data for metric {metric}.")
        return

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.boxplot(
        data=plot_df,
        x=parameter_column,
        y="value",
        hue="MethodLabel",
        hue_order=ordered_labels(plot_df["MethodLabel"]),
        showfliers=showfliers,
        ax=ax,
    )

    apply_axis_options(ax, xlabel=xlabel, ylabel=ylabel)
    add_figure_legend(fig)
    save_or_show(fig, figure_dir, figure_name)


def plot_error_ecdf(
    events_df: pd.DataFrame,
    parameter_column: str,
    figure_dir: Path,
    figure_name: str,
    mode: str = "separate",
    error_column: str = "est_position_error",
    fig_width: float = COL_WIDTH,
    fig_height: float = ROW_HEIGHT,
) -> None:
    """Plot localization-error ECDFs combined and/or per parameter value.

    The v-TWR part of a method (<METHOD>.v_twr in --methods) is the ECDF of
    its events with localization_method v_twr.
    """
    if events_df.empty or parameter_column not in events_df.columns:
        return

    plot_df = add_method_labels(events_df)
    if "localization_method" in events_df.columns:
        vtwr_df = add_method_labels(
            events_df.loc[events_df["localization_method"] == "v_twr"], VTWR_SUFFIX
        )
        plot_df = pd.concat([plot_df, vtwr_df], ignore_index=True)
    plot_df = plot_df.loc[plot_df[error_column].notna()].copy()

    if plot_df.empty:
        return

    if mode not in {"combined", "separate", "both"}:
        raise ValueError(f"Unsupported ECDF mode: {mode}")

    modes = ("combined", "separate") if mode == "both" else (mode,)
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    for current_mode in modes:
        if current_mode == "combined":
            subsets = [(None, plot_df)]
        else:
            subsets = [
                (parameter_value, subset_df)
                for parameter_value, subset_df in plot_df.groupby(
                    parameter_column, sort=True
                )
            ]

        for parameter_value, parameter_df in subsets:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            if current_mode == "combined":
                groups = parameter_df.groupby(
                    [parameter_column, "MethodLabel"], sort=True
                )
                for index, (group_key, subset_df) in enumerate(groups):
                    value, method_label = group_key
                    sns.ecdfplot(
                        data=subset_df,
                        x=error_column,
                        label=f"{method_label}, {value}",
                        linestyle=linestyles[index % len(linestyles)],
                        ax=ax,
                    )
                output_name = f"{figure_name}_combined"
            else:
                styles = method_styles(list(parameter_df["MethodLabel"].unique()))
                for style in styles.values():
                    # An ECDF has a vertex per event: markers (v-TWR) only
                    # every 10 % of the line length.
                    if "marker" in style:
                        style["markevery"] = 0.1
                for method_label, subset_df in method_groups(parameter_df):
                    sns.ecdfplot(
                        data=subset_df,
                        x=error_column,
                        label=method_label,
                        ax=ax,
                        **styles[method_label],
                    )
                ax.set_title(facet_title(parameter_column, parameter_value))
                output_name = (
                    f"{figure_name}_{parameter_column}_{parameter_value}"
                )

            apply_axis_options(
                ax,
                xlabel=ACCURACY_DESCRIPTOR,
                ylabel=CDF_DESCRIPTOR,
                xlim=(0, 1),
                ylim=(0, 1.01),
            )
            add_figure_legend(fig)
            save_or_show(fig, figure_dir, output_name)


def plot_coverage_error_facet(
    coverage_df: pd.DataFrame,
    events_df: pd.DataFrame,
    figure_dir: Path,
    parameter_column: str = "interAnchorDist",
    figure_name: str = "Coverage_Facet",
    error_column: str = "est_position_error",
    descriptors: tuple[str, ...] = (),
) -> None:
    """Plot coverage ECDF and localization-error ECDF as two-row facets."""
    coverage_df = add_method_labels(coverage_df)
    events_df = add_method_labels(events_df)
    if coverage_df.empty or events_df.empty:
        return
    if parameter_column not in coverage_df.columns or parameter_column not in events_df.columns:
        return

    parameter_values = sorted(coverage_df[parameter_column].dropna().unique())
    method_labels = ordered_labels(
        set(coverage_df["MethodLabel"].dropna()) | set(events_df["MethodLabel"].dropna())
    )

    if not parameter_values:
        return

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(parameter_values),
        figsize=(COL_WIDTH_2, 2 * ROW_HEIGHT),
        sharey="row",
        sharex=False,
    )

    if len(parameter_values) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    styles = method_styles(method_labels)
    ecdf_ticks = np.arange(0, 1.01, 0.25)

    for parameter_index, parameter_value in enumerate(parameter_values):
        ax_cov = axes[0, parameter_index]
        ax_err = axes[1, parameter_index]
        title = (
            f"{descriptors[parameter_index]} ({parameter_value} m)"
            if parameter_index < len(descriptors)
            else f"{parameter_value} m"
        )

        for method_label in method_labels:
            line_kwargs = styles[method_label]

            coverage_subset = coverage_df.loc[
                (coverage_df[parameter_column] == parameter_value)
                & (coverage_df["MethodLabel"] == method_label)
            ]
            error_subset = events_df.loc[
                (events_df[parameter_column] == parameter_value)
                & (events_df["MethodLabel"] == method_label)
                & events_df[error_column].notna()
            ]

            if not coverage_subset.empty:
                sns.ecdfplot(data=coverage_subset, x="coverage", ax=ax_cov, label=method_label, **line_kwargs)
            if not error_subset.empty:
                sns.ecdfplot(data=error_subset, x=error_column, ax=ax_err, **line_kwargs)

        ax_cov.set_title(title)
        apply_axis_options(
            ax_cov,
            xlabel=COVERAGE_DESCRIPTOR,
            ylabel=CDF_DESCRIPTOR if parameter_index == 0 else "",
            xlim=(0, 1.01),
            ylim=(0, 1.01),
            xticks=list(ecdf_ticks),
            yticks=list(ecdf_ticks),
        )
        apply_axis_options(
            ax_err,
            xlabel=ACCURACY_DESCRIPTOR,
            ylabel=CDF_DESCRIPTOR if parameter_index == 0 else "",
            xlim=(0, 1),
            ylim=(0, 1.01),
            yticks=list(ecdf_ticks),
        )

    add_figure_legend(fig)
    save_or_show(fig, figure_dir, figure_name)


def method_styles(method_labels: list[str], with_marker: bool = False) -> dict[str, dict]:
    """Assign one fixed color, line style (and marker) per method.

    The style depends on the method (STYLE_INDEX), not on its label or on the
    other methods in a figure, so styles are consistent across all figures.
    The v-TWR part of a method gets the color of its method, a dotted line and
    hollow diamonds (not used by any method style).
    """
    def base_method(method_label: str) -> str:
        method = label_method(method_label).removesuffix(VTWR_SUFFIX)
        return STYLE_ALIASES.get(method, method)

    unknown_methods = sorted({base_method(label) for label in method_labels} - set(STYLE_INDEX))
    style_index = {
        **STYLE_INDEX,
        **{method: max(STYLE_INDEX.values()) + 1 + i for i, method in enumerate(unknown_methods)},
    }
    palette = sns.color_palette("tab10")
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "x"]
    styles = {}
    for method_label in method_labels:
        index = style_index[base_method(method_label)]
        styles[method_label] = {
            "color": palette[index % len(palette)],
            "linestyle": linestyles[index % len(linestyles)],
        }
        if with_marker:
            styles[method_label]["marker"] = markers[index % len(markers)]
        if label_method(method_label).endswith(VTWR_SUFFIX):
            styles[method_label].update(linestyle=":", marker="D", markerfacecolor="white")
    return styles


def make_parameter_facets(
    parameter_values: list,
) -> tuple[plt.Figure, np.ndarray]:
    """One facet per parameter value, all under one another (a single one: ROW_HEIGHT)."""
    fig, axes = plt.subplots(
        nrows=len(parameter_values),
        ncols=1,
        sharex=True,
        sharey=True,
        # + legend above and x axis below.
        figsize=(COL_WIDTH, max(ROW_HEIGHT, FACET_HEIGHT * len(parameter_values) + 0.6)),
    )
    return fig, np.atleast_1d(axes)


def plot_coverage_over_time(
    coverage_df: pd.DataFrame,
    parameter_column: str,
    figure_dir: Path,
    figure_name: str,
    smoothing_window: float,
) -> None:
    """Instantaneous coverage C(t), averaged over repetitions and smoothed.

    With a single facet (e.g. baseline), the legend shows the mean coverage
    (time average of the unsmoothed C(t), averaged over repetitions) of every
    method; with several facets these are the values of the coverage plot.
    """
    plot_df = add_method_labels(coverage_df)
    if plot_df.empty or parameter_column not in plot_df.columns:
        print(f"Skipping {figure_name}: no coverage data.")
        return

    window_size = float(plot_df["coverage_window_size"].iloc[0])
    smoothing_windows = max(1, round(smoothing_window / window_size))
    parameter_values = sorted(plot_df[parameter_column].dropna().unique())
    styles = method_styles(list(plot_df["MethodLabel"].unique()))

    fig, axes = make_parameter_facets(parameter_values)
    for ax, parameter_value in zip(axes, parameter_values):
        parameter_df = plot_df.loc[plot_df[parameter_column] == parameter_value]
        for method_label, method_df in method_groups(parameter_df):
            label = method_label
            if len(parameter_values) == 1:
                mean_coverage = method_df.groupby("Rep")["coverage"].mean().mean()
                label += f" ($\\bar{{C}}$={mean_coverage:.2f})"
            coverage_curve = method_df.groupby("time_bin_start")["coverage"].mean()
            smoothed_curve = coverage_curve.rolling(
                smoothing_windows, center=True, min_periods=1
            ).mean()
            ax.plot(
                smoothed_curve.index,
                smoothed_curve.to_numpy(),
                label=label,
                **styles[method_label],
            )

        ax.set_title(facet_title(parameter_column, parameter_value))
        apply_axis_options(ax, ylim=(0, 1.02))

    set_facet_ylabel(fig, axes, COVERAGE_DESCRIPTOR)
    axes[-1].set_xlabel(TIME_DESCRIPTOR)
    add_figure_legend(fig)
    save_or_show(fig, figure_dir, figure_name)


def plot_availability_cdf(
    availability_df: pd.DataFrame,
    parameter_column: str,
    figure_dir: Path,
    figure_name: str,
) -> None:
    """ECDF of the node availability A_i over all nodes and repetitions."""
    plot_df = add_method_labels(availability_df)
    if plot_df.empty or parameter_column not in plot_df.columns:
        print(f"Skipping {figure_name}: no availability data.")
        return

    parameter_values = sorted(plot_df[parameter_column].dropna().unique())
    styles = method_styles(list(plot_df["MethodLabel"].unique()))

    fig, axes = make_parameter_facets(parameter_values)
    for ax, parameter_value in zip(axes, parameter_values):
        parameter_df = plot_df.loc[plot_df[parameter_column] == parameter_value]
        for method_label, method_df in method_groups(parameter_df):
            sns.ecdfplot(
                data=method_df,
                x="availability",
                label=method_label,
                ax=ax,
                **styles[method_label],
            )

        ax.set_title(facet_title(parameter_column, parameter_value))
        # Start slightly below 0 so that the step of nodes with A_i = 0 is visible.
        apply_axis_options(ax, xlim=(-0.02, 1), ylim=(0, 1.02))

    set_facet_ylabel(fig, axes, CDF_DESCRIPTOR)
    axes[-1].set_xlabel(AVAILABILITY_DESCRIPTOR)
    add_figure_legend(fig)
    save_or_show(fig, figure_dir, figure_name)


def running_median_availability(
    run_events_df: pd.DataFrame,
    num_nodes: int,
    num_windows: int,
    window_size: float,
) -> np.ndarray:
    """Median over all nodes of A_i(t) = localized windows up to t / windows up to t."""
    localized = np.zeros((num_nodes, num_windows), dtype=bool)
    window_index = np.floor(run_events_df["event_time"].to_numpy() / window_size).astype(int)
    node_index = pd.factorize(run_events_df["NodeId"])[0]
    in_range = window_index < num_windows
    localized[node_index[in_range], window_index[in_range]] = True

    running_availability = localized.cumsum(axis=1) / np.arange(1, num_windows + 1)
    return np.median(running_availability, axis=0)


def plot_availability_over_time(
    events_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    parameter_column: str,
    figure_dir: Path,
    figure_name: str,
) -> None:
    """Running median node availability over time, averaged over repetitions.

    Number of nodes and time axis of every run are taken from coverage.csv, so
    runs without any event are included as zero.
    """
    coverage_df = add_method_labels(coverage_df)
    if coverage_df.empty or parameter_column not in coverage_df.columns:
        print(f"Skipping {figure_name}: no coverage data.")
        return

    window_size = float(coverage_df["coverage_window_size"].iloc[0])
    run_keys = [parameter_column, "Method", "Rep"]
    events_by_run = (
        {key: run_df for key, run_df in events_df.groupby(run_keys)}
        if not events_df.empty
        else {}
    )
    empty_events = pd.DataFrame({"event_time": [], "NodeId": []})

    curves = []
    for (parameter_value, method, rep, method_label), run_coverage_df in coverage_df.groupby(
        [*run_keys, "MethodLabel"]
    ):
        median_curve = running_median_availability(
            events_by_run.get((parameter_value, method, rep), empty_events),
            num_nodes=int(run_coverage_df["total_nodes"].iloc[0]),
            num_windows=len(run_coverage_df),
            window_size=window_size,
        )
        curves.append(pd.DataFrame({
            parameter_column: parameter_value,
            "MethodLabel": method_label,
            "time": np.arange(len(median_curve)) * window_size,
            "median_availability": median_curve,
        }))
    curves_df = pd.concat(curves, ignore_index=True)

    parameter_values = sorted(curves_df[parameter_column].unique())
    styles = method_styles(list(curves_df["MethodLabel"].unique()))
    fig, axes = make_parameter_facets(parameter_values)
    for ax, parameter_value in zip(axes, parameter_values):
        parameter_df = curves_df.loc[curves_df[parameter_column] == parameter_value]
        for method_label, method_df in method_groups(parameter_df):
            mean_curve = method_df.groupby("time")["median_availability"].mean()
            ax.plot(
                mean_curve.index,
                mean_curve.to_numpy(),
                label=method_label,
                **styles[method_label],
            )

        ax.set_title(facet_title(parameter_column, parameter_value))
        apply_axis_options(ax, ylim=(0, 1.02))

    set_facet_ylabel(fig, axes, MEDIAN_AVAILABILITY_DESCRIPTOR)
    axes[-1].set_xlabel(TIME_DESCRIPTOR)
    add_figure_legend(fig)
    save_or_show(fig, figure_dir, figure_name)


def plot_availability_deciles(
    aggregated_statistics_df: pd.DataFrame,
    parameter_column: str,
    xlabel: str,
    figure_dir: Path,
    figure_name: str,
    xticks: list[float] | None = None,
    show_errorbars: bool = True,
) -> None:
    """10 % quantile, median and 90 % quantile of the node availability, one panel each.

    In every run the quantiles are taken over the availabilities A_i of all
    simulated nodes; the panels show their mean ± SEM over the repetitions.
        q10:    the worst-served nodes (10 % of the nodes are below this value)
        median: the typical node
        q90:    the best-served nodes (10 % of the nodes are above this value)
    Separate panels instead of one band (availability_quantiles), because the
    bands of several methods overlap and hide each other.
    """
    panels = (
        ("q10_availability", "10 % Quantile"),
        ("median_availability", "Median"),
        ("q90_availability", "90 % Quantile"),
    )
    panel_dfs = [
        metric_with_mean_and_sem(aggregated_statistics_df, metric, parameter_column)
        for metric, _ in panels
    ]
    if any(panel_df.empty or "mean" not in panel_df.columns for panel_df in panel_dfs):
        print(f"Skipping {figure_name}: no availability quantiles.")
        return

    styles = method_styles(
        ordered_labels(set().union(*(panel_df["MethodLabel"] for panel_df in panel_dfs))),
        with_marker=True,
    )
    if xticks is None:
        xticks = sorted(panel_dfs[0][parameter_column].unique())

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(panels),
        sharey=True,
        figsize=(COL_WIDTH_2, ROW_HEIGHT),
    )
    for ax, (_, title), panel_df in zip(axes, panels, panel_dfs):
        for method_label, method_df in method_groups(panel_df):
            method_df = method_df.sort_values(parameter_column)
            ax.errorbar(
                method_df[parameter_column],
                method_df["mean"],
                yerr=method_df["sem"] if show_errorbars and "sem" in method_df.columns else None,
                label=method_label,
                **styles[method_label],
            )
        ax.set_title(title)
        apply_axis_options(ax, ylim=(0, 1.02), xticks=xticks)
    axes[0].set_ylabel(AVAILABILITY_DESCRIPTOR)
    # One x label for the three panels.
    fig.supxlabel(xlabel)
    add_figure_legend(fig)
    save_or_show(fig, figure_dir, figure_name)


def plot_availability_quantiles(
    aggregated_statistics_df: pd.DataFrame,
    parameter_column: str,
    xlabel: str,
    figure_dir: Path,
    figure_name: str,
    xticks: list[float] | None = None,
) -> None:
    """Median node availability with a band from the 10 % to the 90 % quantile.

    All three values are means over the repetitions.
    """
    quantile_dfs = {
        quantile: metric_with_mean_and_sem(
            aggregated_statistics_df, f"{quantile}_availability", parameter_column
        )
        for quantile in ("q10", "median", "q90")
    }
    if any(quantile_df.empty for quantile_df in quantile_dfs.values()):
        print(f"Skipping {figure_name}: no availability quantiles.")
        return

    plot_df = quantile_dfs["median"][["MethodLabel", parameter_column, "mean"]]
    for quantile in ("q10", "q90"):
        plot_df = plot_df.merge(
            quantile_dfs[quantile][["MethodLabel", parameter_column, "mean"]],
            on=["MethodLabel", parameter_column],
            suffixes=("", f"_{quantile}"),
        )

    styles = method_styles(list(plot_df["MethodLabel"].unique()))
    fig, ax = plt.subplots(figsize=(COL_WIDTH, ROW_HEIGHT))
    for method_label, method_df in method_groups(plot_df):
        method_df = method_df.sort_values(parameter_column)
        ax.plot(
            method_df[parameter_column],
            method_df["mean"],
            marker="o",
            label=method_label,
            **styles[method_label],
        )
        ax.fill_between(
            method_df[parameter_column],
            method_df["mean_q10"],
            method_df["mean_q90"],
            color=styles[method_label]["color"],
            alpha=0.2,
        )

    apply_axis_options(
        ax, xlabel=xlabel, ylabel=AVAILABILITY_DESCRIPTOR, ylim=(0, 1.02), xticks=xticks
    )
    add_figure_legend(fig)
    save_or_show(fig, figure_dir, figure_name)


def plot_metric_vs_metric(
    aggregated_statistics_df: pd.DataFrame,
    parameter_column: str,
    x_metric: str,
    y_metric: str,
    xlabel: str,
    ylabel: str,
    figure_dir: Path,
    figure_name: str,
) -> None:
    """One point per method and parameter value: mean ± SEM of two metrics.

    Points of one method are connected in parameter order and annotated with
    the parameter value.
    """
    x_df = metric_with_mean_and_sem(aggregated_statistics_df, x_metric, parameter_column)
    y_df = metric_with_mean_and_sem(aggregated_statistics_df, y_metric, parameter_column)
    if x_df.empty or y_df.empty:
        print(f"Skipping {figure_name}: no data for {x_metric} / {y_metric}.")
        return

    keys = ["MethodLabel", parameter_column]
    plot_df = x_df[[*keys, "mean", "sem"]].merge(
        y_df[[*keys, "mean", "sem"]], on=keys, suffixes=("_x", "_y")
    ).dropna(subset=["mean_x", "mean_y"])

    styles = method_styles(list(plot_df["MethodLabel"].unique()))
    fig, ax = plt.subplots(figsize=(COL_WIDTH, ROW_HEIGHT))
    for method_label, method_df in method_groups(plot_df):
        method_df = method_df.sort_values(parameter_column)
        ax.errorbar(
            method_df["mean_x"],
            method_df["mean_y"],
            xerr=method_df["sem_x"],
            yerr=method_df["sem_y"],
            marker="o",
            label=method_label,
            **styles[method_label],
        )
        for _, row in method_df.iterrows():
            ax.annotate(
                format_parameter_value(row[parameter_column]),
                (row["mean_x"], row["mean_y"]),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=TICK_FONT_SIZE,
            )

    apply_axis_options(ax, xlabel=xlabel, ylabel=ylabel)
    add_figure_legend(fig)
    save_or_show(fig, figure_dir, figure_name)


def plot_metric_by_technique(
    aggregated_statistics_df: pd.DataFrame,
    parameter_column: str,
    metric: str,
    xlabel: str,
    ylabel: str,
    figure_dir: Path,
    figure_name: str,
    xticks: list[float] | None = None,
    show_errorbars: bool = True,
    ylog: bool = False,
    fig_width: float = COL_WIDTH,
) -> None:
    """Metric of every method and, separately, of its v-TWR part (mean ± SEM over repetitions).

    Two lines per method:
        HyRASeL (Centr./Distr.):  <metric> of the combined event stream
                                  (self-localization + v-TWR), solid line
        v-TWR (Centr./Distr.):    <metric>_v_twr, i.e. the same statistic on
                                  the v-TWR events only, dotted line with
                                  hollow diamonds in the color of its method
    The _v_twr metrics are written by extract_statistics.py
    (_make_localization_method_statistics_df).  The v-TWR line of a method is
    shown if <METHOD>.v_twr is selected (--methods).  ylog: logarithmic y
    axis, e.g. for the interval (HyRASeL updates every few ms, v-TWR every
    few 100 ms).
    """
    combined_df = metric_with_mean_and_sem(aggregated_statistics_df, metric, parameter_column)
    vtwr_df = metric_with_mean_and_sem(
        aggregated_statistics_df, f"{metric}_v_twr", parameter_column, VTWR_SUFFIX
    )
    if "mean" not in vtwr_df.columns and any(method.endswith(VTWR_SUFFIX) for method in PLOT_METHODS):
        print(f"{figure_name}: no {metric}_v_twr statistics; re-run the extraction for the v-TWR lines.")
    plot_dfs = [df for df in (combined_df, vtwr_df) if "mean" in df.columns]
    if not plot_dfs:
        print(f"Skipping {figure_name}: no data for metric {metric}.")
        return
    plot_df = pd.concat(plot_dfs, ignore_index=True).dropna(subset=["mean"])

    styles = method_styles(list(plot_df["MethodLabel"].unique()), with_marker=True)

    fig, ax = plt.subplots(figsize=(fig_width, ROW_HEIGHT))
    for method_label, method_df in method_groups(plot_df):
        method_df = method_df.sort_values(parameter_column)
        ax.errorbar(
            method_df[parameter_column],
            method_df["mean"],
            yerr=method_df["sem"] if show_errorbars and "sem" in method_df.columns else None,
            label=method_label,
            **styles[method_label],
        )

    if xticks is None:
        xticks = sorted(plot_df[parameter_column].unique())
    apply_axis_options(ax, xlabel=xlabel, ylabel=ylabel, xticks=xticks, ylog=ylog)
    add_figure_legend(fig)
    save_or_show(fig, figure_dir, figure_name)


# -----------------------------------------------------------------------------
# v-anchor opportunities (HyRASeL Centralized / Distributed)
# -----------------------------------------------------------------------------
#
# One opportunity = the first v-anchor slot of one anchor sequence.  Every
# sequence has exactly one outcome (used / missed / no eligible node); the
# definitions are in opportunities.py.  Two figures:
#   opportunities       stacked outcome fractions versus the sweep parameter
#                       (input: opportunity_* metrics of statistics_run_aggregated.csv)
#   opportunities_time  used fraction and eligible fraction over time
#                       (input: opportunities.csv of every run)

def plot_opportunity_outcomes(
    aggregated_statistics_df: pd.DataFrame,
    parameter_column: str,
    xlabel: str,
    figure_dir: Path,
    figure_name: str,
) -> None:
    """Outcome of the v-anchor opportunities, stacked, one facet per method.

    Bars: mean fraction of every outcome over the repetitions (sums up to 1),
          colors and order from OPPORTUNITY_OUTCOMES.  Outcomes that never
          occur for the plotted methods are left out (e.g. failed contention
          and stale seqNo for Centralized).
    Black line: mean fraction of eligible nodes (heard all anchors).  For
          Centralized the green bar follows this line, because a random
          selection hits an eligible node with exactly this probability.
    The x positions are categorical (one bar per simulated parameter value).
    A single method gets no facet title.
    """
    outcome_dfs = {
        outcome: metric_with_mean_and_sem(
            aggregated_statistics_df, f"opportunity_{outcome}", parameter_column
        )
        for outcome in OPPORTUNITY_OUTCOMES
    }
    eligible_df = metric_with_mean_and_sem(
        aggregated_statistics_df, "mean_eligible_fraction", parameter_column
    )
    if any(outcome_df.empty for outcome_df in outcome_dfs.values()):
        print(f"Skipping {figure_name}: no opportunity statistics.")
        return

    method_labels = ordered_labels(outcome_dfs["used"]["MethodLabel"])
    parameter_values = sorted(outcome_dfs["used"][parameter_column].unique())
    positions = np.arange(len(parameter_values))

    # Mean fraction per outcome, method and parameter value.
    outcome_values = {
        outcome: {
            method_label: (
                outcome_df.loc[outcome_df["MethodLabel"] == method_label]
                .set_index(parameter_column)["mean"]
                .reindex(parameter_values)
                .fillna(0.0)
                .to_numpy()
            )
            for method_label in method_labels
        }
        for outcome, outcome_df in outcome_dfs.items()
    }
    shown_outcomes = {
        outcome: label
        for outcome, (label, _) in OPPORTUNITY_OUTCOMES.items()
        if any(np.any(values > 0) for values in outcome_values[outcome].values())
    }
    if [outcome for outcome in shown_outcomes if outcome.startswith("missed_")] == ["missed_no_sender"]:
        shown_outcomes["missed_no_sender"] = MISSED_LABEL

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(method_labels),
        sharey=True,
        figsize=(COL_WIDTH_2 if len(method_labels) > 1 else COL_WIDTH, ROW_HEIGHT),
    )
    axes = np.atleast_1d(axes)
    for ax, method_label in zip(axes, method_labels):
        # Stack the outcome fractions on top of each other.
        bottom = np.zeros(len(parameter_values))
        for outcome, outcome_label in shown_outcomes.items():
            values = outcome_values[outcome][method_label]
            ax.bar(
                positions, values, bottom=bottom, width=0.7,
                color=OPPORTUNITY_OUTCOMES[outcome][1], label=outcome_label,
            )
            bottom += values

        method_eligible_df = (
            eligible_df.loc[eligible_df["MethodLabel"] == method_label]
            .set_index(parameter_column)
            .reindex(parameter_values)
        )
        ax.plot(
            positions,
            method_eligible_df["mean"].to_numpy(),
            color="black",
            marker="o",
            label="Eligible nodes",
        )

        if len(method_labels) > 1:
            ax.set_title(method_label)
        apply_axis_options(ax, ylim=(0, 1.02))
        ax.set_xticks(positions, [format_parameter_value(value) for value in parameter_values])
    axes[0].set_ylabel(OPPORTUNITY_DESCRIPTOR)
    if len(axes) > 1:
        fig.supxlabel(xlabel)
    else:
        axes[0].set_xlabel(xlabel)

    # One legend for all facets (identical entries), in stacking order.
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(label) for label in [*shown_outcomes.values(), "Eligible nodes"]]
    add_figure_legend(fig, [handles[i] for i in order], [labels[i] for i in order])
    save_or_show(fig, figure_dir, figure_name)


def plot_opportunities_over_time(
    opportunity_df: pd.DataFrame,
    parameter_column: str,
    figure_dir: Path,
    figure_name: str,
    window: float,
) -> None:
    """Fraction of used opportunities (solid) and of eligible nodes (dotted) over time.

    One facet per parameter value, one color per method.  Sequences are
    grouped into time windows of the given length (--smoothing-window, about
    20 sequences per 0.25 s), averaged per run and then over the repetitions.
    With a single facet, the legend shows the fraction of used opportunities
    over the whole run (mean over repetitions; with several facets see the
    opportunities plot).  The eligible fraction shows how many nodes could
    have answered at that time, e.g. while the nodes move through the anchor
    field.  The legend explains the colors of the methods (only with several
    methods) and the two line styles.
    """
    plot_df = add_method_labels(opportunity_df)
    if plot_df.empty or parameter_column not in plot_df.columns:
        print(f"Skipping {figure_name}: no opportunity data.")
        return

    # Center of the time window of every sequence.
    plot_df["time"] = (np.floor(plot_df["seq_start_time"] / window) + 0.5) * window
    plot_df["eligible_fraction"] = plot_df["num_eligible_nodes"] / plot_df["num_nodes"]
    parameter_values = sorted(plot_df[parameter_column].dropna().unique())
    styles = method_styles(list(plot_df["MethodLabel"].unique()))

    fig, axes = make_parameter_facets(parameter_values)
    used_fractions = {}
    for ax, parameter_value in zip(axes, parameter_values):
        parameter_df = plot_df.loc[plot_df[parameter_column] == parameter_value]
        for method_label, method_df in method_groups(parameter_df):
            # Mean per run and window first, then over the repetitions, so
            # every repetition has the same weight.
            curves = (
                method_df.groupby(["Rep", "time"])[["used", "eligible_fraction"]].mean()
                .groupby("time").mean()
            )
            used_fractions[method_label] = method_df.groupby("Rep")["used"].mean().mean()
            ax.plot(
                curves.index,
                curves["used"].to_numpy(),
                **{**styles[method_label], "linestyle": "-"},
            )
            ax.plot(
                curves.index,
                curves["eligible_fraction"].to_numpy(),
                **{**styles[method_label], "linestyle": ":"},
            )

        ax.set_title(facet_title(parameter_column, parameter_value))
        apply_axis_options(ax, ylim=(0, 1.02))

    set_facet_ylabel(fig, axes, "Fraction")
    axes[-1].set_xlabel(TIME_DESCRIPTOR)

    method_labels = ordered_labels(used_fractions)
    used_label = "Used opportunities"
    if len(parameter_values) == 1 and len(method_labels) == 1:
        used_label += f" ({used_fractions[method_labels[0]]:.2f})"
    # Line styles in the method color if there is only one method.
    style_color = styles[method_labels[0]]["color"] if len(method_labels) == 1 else "gray"
    handles = [
        Line2D([], [], color=style_color, linestyle="-"),
        Line2D([], [], color=style_color, linestyle=":"),
    ]
    labels = [used_label, "Eligible nodes"]
    if len(method_labels) > 1:
        handles = [
            Line2D([], [], color=styles[method_label]["color"], linestyle="-")
            for method_label in method_labels
        ] + handles
        labels = [
            method_label
            + (f" (used: {used_fractions[method_label]:.2f})" if len(parameter_values) == 1 else "")
            for method_label in method_labels
        ] + labels
    add_figure_legend(fig, handles, labels)
    save_or_show(fig, figure_dir, figure_name)


# Rows of the opportunity table: (column title, metric, transform).  Values
# are fractions of all opportunities (= anchor sequences) unless noted.
OPPORTUNITY_TABLE_ROWS = (
    ("Used [%]", "opportunity_used", False),
    ("Missed [%]", "opportunity_missed", False),
    ("No eligible node [%]", "opportunity_no_eligible_node", False),
    # 1 - efficiency: missed share of the sequences that had an eligible node.
    ("Missed of usable [%]", "opportunity_efficiency", True),
    ("Eligible nodes [%]", "mean_eligible_fraction", False),
)


def write_opportunity_table(
    aggregated_statistics_df: pd.DataFrame,
    parameter_column: str,
    figure_dir: Path,
    figure_name: str,
) -> None:
    """Opportunity percentages per method and parameter value as table (for the text).

    Columns (mean ± SEM over repetitions, in percent):
        Used                  sequences with a new v-anchor
        Missed                sequences without a new v-anchor although at
                              least one node was eligible (all missed_* outcomes)
        No eligible node      sequences in which no node heard all anchors
                              (Used + Missed + No eligible node = 100 %)
        Missed of usable      Missed / (Used + Missed) = 1 - opportunity_efficiency
        Eligible nodes        mean fraction of eligible nodes per sequence
    Written as <figure_name>.csv (numbers) and <figure_name>.md (formatted)
    next to the figures and printed to the console.
    """
    rows = []
    for title, metric, complement in OPPORTUNITY_TABLE_ROWS:
        metric_df = metric_with_mean_and_sem(aggregated_statistics_df, metric, parameter_column)
        if metric_df.empty and metric == "opportunity_missed":
            # Extractions before opportunity_missed existed: sum of the missed
            # outcomes (mean only, the SEM of a sum is not available).
            parts = [
                metric_with_mean_and_sem(aggregated_statistics_df, f"opportunity_{outcome}", parameter_column)
                for outcome in OPPORTUNITY_OUTCOMES
                if outcome.startswith("missed_")
            ]
            parts = [part for part in parts if not part.empty]
            if parts:
                metric_df = parts[0].copy()
                metric_df["mean"] = sum(part["mean"].to_numpy() for part in parts)
                metric_df["sem"] = np.nan
        if metric_df.empty:
            continue
        metric_df = metric_df[["Method", "MethodLabel", parameter_column, "mean", "sem", "n_runs"]].copy()
        if complement:
            metric_df["mean"] = 1.0 - metric_df["mean"]
        metric_df["column"] = title
        rows.append(metric_df)

    if not rows:
        print(f"Skipping {figure_name}: no opportunity statistics.")
        return

    table_df = pd.concat(rows, ignore_index=True)
    table_df["mean"] *= 100
    table_df["sem"] *= 100
    column_order = [title for title, _, _ in OPPORTUNITY_TABLE_ROWS]
    index_columns = ["MethodLabel", parameter_column, "n_runs"]

    numbers_df = table_df.pivot_table(
        index=index_columns, columns="column", values=["mean", "sem"], aggfunc="first"
    )
    numbers_df.columns = [f"{title} {statistic}" for statistic, title in numbers_df.columns]
    numbers_df = numbers_df[
        [f"{title} {statistic}" for title in column_order for statistic in ("mean", "sem")
         if f"{title} {statistic}" in numbers_df.columns]
    ].reset_index()

    table_df["text"] = [
        f"{mean:.1f}" + ("" if np.isnan(sem) else f" ± {sem:.1f}")
        for mean, sem in zip(table_df["mean"], table_df["sem"])
    ]
    text_df = table_df.pivot_table(
        index=index_columns, columns="column", values="text", aggfunc="first"
    )
    text_df = text_df[[title for title in column_order if title in text_df.columns]].reset_index()
    text_df = text_df.rename(columns={"MethodLabel": "Method", "n_runs": "Runs"})
    # Plain Markdown table (pandas.to_markdown would need the tabulate package).
    header = [str(column) for column in text_df.columns]
    markdown = "\n".join(
        [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
            *(
                "| " + " | ".join(str(value) for value in row) + " |"
                for row in text_df.itertuples(index=False)
            ),
        ]
    )

    print(f"\n{figure_name} (mean ± SEM over repetitions):\n{markdown}\n")
    if SAVE_FIGURES:
        figure_dir.mkdir(parents=True, exist_ok=True)
        numbers_df.to_csv(figure_dir / f"{figure_name}.csv", index=False)
        (figure_dir / f"{figure_name}.md").write_text(markdown + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Plot orchestration
# -----------------------------------------------------------------------------

def plot_sweep_parameter(
    data: dict[str, pd.DataFrame],
    parameter_column: str,
    parameter_config: dict,
    figure_dir: Path,
    plots: set[str],
    ecdf_mode: str,
    show_errorbars: bool,
    showfliers: bool,
    smoothing_window: float,
    widths: dict[str, float] | None = None,
) -> None:
    """Generate the selected plots for one available sweep parameter.

    widths: figure width of single plots as a fraction of COL_WIDTH (--width,
    only LINE_PLOTS and TECHNIQUE_PLOTS).
    """
    label = parameter_config["label"]
    prefix = parameter_config["figure_prefix"]
    filename_prefix = f"{prefix}_" if prefix else ""
    xticks = parameter_config["xticks"]
    aggregated_df = data["statistics_run_aggregated"]

    widths = widths or {}

    # Mean ± SEM over repetitions versus the sweep parameter.
    for plot_name, (metric, ylabel, ylog) in LINE_PLOTS.items():
        if plot_name not in plots:
            continue
        plot_metric_line(
            aggregated_statistics_df=aggregated_df,
            parameter_column=parameter_column,
            metric=metric,
            xlabel=label,
            ylabel=ylabel,
            figure_dir=figure_dir,
            figure_name=f"{filename_prefix}{plot_name}_vs_{parameter_column}",
            xticks=xticks,
            show_errorbars=show_errorbars,
            fig_width=COL_WIDTH * widths.get(plot_name, 1.0),
            ylog=ylog,
        )

    if "availability_deciles" in plots:
        plot_availability_deciles(
            aggregated_statistics_df=aggregated_df,
            parameter_column=parameter_column,
            xlabel=label,
            figure_dir=figure_dir,
            figure_name=f"{filename_prefix}availability_deciles_vs_{parameter_column}",
            xticks=xticks,
            show_errorbars=show_errorbars,
        )

    if "availability_quantiles" in plots:
        plot_availability_quantiles(
            aggregated_statistics_df=aggregated_df,
            parameter_column=parameter_column,
            xlabel=label,
            figure_dir=figure_dir,
            figure_name=f"{filename_prefix}availability_quantiles_vs_{parameter_column}",
            xticks=xticks,
        )

    if "error_boxplot" in plots:
        plot_metric_boxplot(
            statistics_df=data["statistics_run"],
            parameter_column=parameter_column,
            metric="mean_error",
            xlabel=label,
            ylabel=ACCURACY_DESCRIPTOR,
            figure_dir=figure_dir,
            figure_name=f"{filename_prefix}error_boxplot_vs_{parameter_column}",
            showfliers=showfliers,
        )

    if "node_error_boxplot" in plots:
        plot_metric_boxplot(
            statistics_df=data["statistics_node"],
            parameter_column=parameter_column,
            metric="mean_error",
            xlabel=label,
            ylabel=ACCURACY_DESCRIPTOR,
            figure_dir=figure_dir,
            figure_name=(
                f"{filename_prefix}node_error_boxplot_vs_{parameter_column}"
            ),
            showfliers=showfliers,
        )

    if "error_ecdf" in plots:
        plot_error_ecdf(
            events_df=data["events"],
            parameter_column=parameter_column,
            figure_dir=figure_dir,
            figure_name=f"{filename_prefix}error_ecdf_vs_{parameter_column}",
            mode=ecdf_mode,
        )

    # One facet per parameter value, all under one another.
    if "coverage_time" in plots:
        plot_coverage_over_time(
            coverage_df=data["coverage"],
            parameter_column=parameter_column,
            figure_dir=figure_dir,
            figure_name=f"{filename_prefix}coverage_time_by_{parameter_column}",
            smoothing_window=smoothing_window,
        )

    if "availability_cdf" in plots:
        plot_availability_cdf(
            availability_df=data["availability"],
            parameter_column=parameter_column,
            figure_dir=figure_dir,
            figure_name=f"{filename_prefix}availability_cdf_by_{parameter_column}",
        )

    if "availability_time" in plots:
        plot_availability_over_time(
            events_df=data["events"],
            coverage_df=data["coverage"],
            parameter_column=parameter_column,
            figure_dir=figure_dir,
            figure_name=f"{filename_prefix}availability_time_by_{parameter_column}",
        )

    # Mean coverage against another metric; one point per parameter value.
    scatter_plots = {
        "error_vs_coverage": ("mean_error", ACCURACY_DESCRIPTOR),
        "interval_vs_coverage": ("mean_localization_interval", INTER_UPDATE_TIME_DESCRIPTOR),
    }
    for plot_name, (x_metric, xlabel) in scatter_plots.items():
        if plot_name not in plots:
            continue
        plot_metric_vs_metric(
            aggregated_statistics_df=aggregated_df,
            parameter_column=parameter_column,
            x_metric=x_metric,
            y_metric="mean_coverage",
            xlabel=xlabel,
            ylabel=MEAN_COVERAGE_DESCRIPTOR,
            figure_dir=figure_dir,
            figure_name=f"{filename_prefix}{plot_name}_by_{parameter_column}",
        )

    # HyRASeL and its v-TWR part as separate lines (previous Figure 04a/b).
    for plot_name, (metric, ylabel, ylog) in TECHNIQUE_PLOTS.items():
        if plot_name not in plots:
            continue
        plot_metric_by_technique(
            aggregated_statistics_df=aggregated_df,
            parameter_column=parameter_column,
            metric=metric,
            xlabel=label,
            ylabel=ylabel,
            figure_dir=figure_dir,
            figure_name=f"{filename_prefix}{plot_name}_vs_{parameter_column}",
            xticks=xticks,
            show_errorbars=show_errorbars,
            fig_width=COL_WIDTH * widths.get(plot_name, 1.0),
            ylog=ylog,
        )

    # v-anchor opportunities; only Centralized/Distributed have the data.
    if "opportunities" in plots:
        plot_opportunity_outcomes(
            aggregated_statistics_df=aggregated_df,
            parameter_column=parameter_column,
            xlabel=label,
            figure_dir=figure_dir,
            figure_name=f"{filename_prefix}opportunities_vs_{parameter_column}",
        )

    if "opportunity_table" in plots:
        write_opportunity_table(
            aggregated_statistics_df=aggregated_df,
            parameter_column=parameter_column,
            figure_dir=figure_dir,
            figure_name=f"{filename_prefix}opportunity_table_by_{parameter_column}",
        )

    if "opportunities_time" in plots:
        plot_opportunities_over_time(
            opportunity_df=data["opportunities"],
            parameter_column=parameter_column,
            figure_dir=figure_dir,
            figure_name=f"{filename_prefix}opportunities_time_by_{parameter_column}",
            window=smoothing_window,
        )

    if "coverage_error_facet" in plots and parameter_column == "interAnchorDist":
        plot_coverage_error_facet(
            coverage_df=data["coverage"],
            events_df=data["events"],
            figure_dir=figure_dir,
        )


def main() -> None:
    """Load exported data and generate all available plots."""
    global FIGURE_EXTENSION, SAVE_FIGURES, SHOW_FIGURES, TIMESTAMP_FIGURES
    global PLOT_METHODS, METHOD_LABELS

    args = parse_args()
    evaluation_dir = args.evaluation_dir.expanduser().resolve()
    figure_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else build_figure_dir(evaluation_dir)
    )
    FIGURE_EXTENSION = args.format
    SAVE_FIGURES = not args.no_save
    SHOW_FIGURES = args.show
    TIMESTAMP_FIGURES = args.timestamp
    PLOT_METHODS = list(dict.fromkeys(args.methods))
    METHOD_LABELS = args.method_labels

    plots = set(args.plots)
    data = load_data(evaluation_dir, plots)

    available_parameters = [
        parameter
        for parameter in SWEEP_PARAMETERS
        if any(parameter in dataframe.columns for dataframe in data.values() if not dataframe.empty)
    ]

    parameter_configs = {
        parameter: SWEEP_PARAMETERS[parameter] for parameter in available_parameters
    }
    if not parameter_configs:
        parameter_configs = {NO_SWEEP_PARAMETER: NO_SWEEP_PARAMETER_CONFIG}

    print(f"Available sweep parameters: {list(parameter_configs)}")
    print(f"Figure output directory: {figure_dir}")

    for parameter_column, parameter_config in parameter_configs.items():
        plot_sweep_parameter(
            data=data,
            parameter_column=parameter_column,
            parameter_config=parameter_config,
            figure_dir=figure_dir,
            plots=plots,
            ecdf_mode=args.ecdf_mode,
            show_errorbars=not args.no_errorbars,
            showfliers=args.show_fliers,
            smoothing_window=args.smoothing_window,
            widths=args.width,
        )


if __name__ == "__main__":
    main()
