"""Look of all figures: fonts, method styles, labels, figure layout, legend and saving.

Used by plot.py; the values (sizes, labels, styles) are set in the plots
section of configs/defaults.yml.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

# Cycles of the method styles (index from plots.styles in configs/defaults.yml).
PALETTE = sns.color_palette("tab10")
LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "x"]
# Color of the sweep-value shades in the legend of combined figures.
VALUE_LEGEND_COLOR = "gray"

# Stacked facets: each facet gets half the height of a single-axes figure,
# plus this space for the legend above and the x axis below [in].
FACETS_EXTRA_HEIGHT = 0.6

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


def font_rc(figure) -> dict:
    """rcParams of the font sizes of one figure (options font_size, tick_font_size)."""
    return {
        **dict.fromkeys(
            ("font.size", "axes.titlesize", "axes.labelsize", "figure.labelsize",
             "legend.fontsize", "legend.title_fontsize"),
            figure.font_size,
        ),
        "xtick.labelsize": figure.tick_font_size,
        "ytick.labelsize": figure.tick_font_size,
    }


class Style:
    """Display labels and line styles of methods, metrics and sweep parameters.

    plots: plots section of the config (labels, styles, part_styles).
    methods: all methods of the figures; those without an entry in styles get
    the next free index (in alphabetical order), so a method looks the same in
    all figures of a config.
    """

    def __init__(self, plots: dict, methods: list[str]):
        self.labels = plots["labels"]
        self.part_styles = plots["part_styles"]
        self.index = dict(plots["styles"])
        for method in sorted({method.partition(".")[0] for method in methods} - set(self.index)):
            self.index[method] = max(self.index.values(), default=-1) + 1

    def label(self, key: str) -> str:
        """Display label of a method, metric or sweep parameter (default: the key)."""
        return self.labels.get(key, key)

    def method(self, method: str, markers: bool) -> dict:
        """Line style of a method; <METHOD>.<part> keeps the color of its method."""
        base, _, part = method.partition(".")
        index = self.index[base]
        style = {"color": PALETTE[index % len(PALETTE)], "linestyle": LINESTYLES[index % len(LINESTYLES)]}
        if markers:
            style["marker"] = MARKERS[index % len(MARKERS)]
        if part:
            style.update(self.part_styles.get(part, {}))
        return style

    def unit(self, parameter: str | None) -> str:
        """Unit of a sweep parameter from its label, e.g. "m" for "Inter-Anchor Distance [m]"."""
        return self.label(parameter).partition(" [")[2].removesuffix("]") if parameter else ""

    def value_label(self, parameter: str | None, value) -> str:
        """Sweep value with its unit, e.g. "130 m"."""
        unit = self.unit(parameter)
        return format_value(value) + (f" {unit}" if unit else "")

    def facet_title(self, parameter: str | None, value=None) -> str:
        """Title of one facet, e.g. "Inter-Anchor Distance: 130 m"; the parameter name without value.

        Without sweep parameter, facets have no title.
        """
        if parameter is None:
            return ""
        name = self.label(parameter).partition(" [")[0]
        return name if value is None else f"{name}: {self.value_label(parameter, value)}"


def format_value(value) -> str:
    """Short label of a sweep value: numbers without trailing zeros, text as is."""
    return f"{value:g}" if isinstance(value, (int, float, np.number)) else str(value)


def value_shades(color, num_values: int) -> list:
    """One shade of color per sweep value, light to dark; the color itself for one value."""
    if num_values == 1:
        return [color]
    rgb = np.array(mpl.colors.to_rgb(color))
    # From halfway to white to 40 % darker.
    return sns.blend_palette([0.5 + 0.5 * rgb, rgb, 0.6 * rgb], num_values)


def create_figure(figure, num_axes: int, columns: bool) -> tuple[plt.Figure, np.ndarray]:
    """Figure with num_axes axes under one another, or next to each other (columns).

    Width: option width times column_width. Height: option height; stacked
    facets get half of it each plus FACETS_EXTRA_HEIGHT.
    """
    width = figure.width * figure.column_width
    if columns:
        fig, axes = plt.subplots(nrows=1, ncols=num_axes, sharey=True, figsize=(width, figure.height))
    else:
        height = max(figure.height, figure.height / 2 * num_axes + FACETS_EXTRA_HEIGHT)
        fig, axes = plt.subplots(nrows=num_axes, ncols=1, sharex=True, sharey=True, figsize=(width, height))
    return fig, np.atleast_1d(axes)


def apply_axis_options(ax: plt.Axes, figure) -> None:
    """Grid and the axis options of the config (log scales, limits, x ticks)."""
    ax.grid(True, alpha=0.3)
    if figure.xlog:
        ax.set_xscale("log")
    if figure.ylog:
        ax.set_yscale("log")
    if figure.xlim is not None:
        ax.set_xlim(figure.xlim)
    if figure.ylim is not None:
        ax.set_ylim(figure.ylim)
    if figure.xticks is not None:
        # Only ticks within the plotted values; set_xticks would extend the axis.
        x_min, x_max = ax.dataLim.intervalx
        tolerance = 1e-9 * max(1.0, abs(x_min), abs(x_max))
        ax.set_xticks([tick for tick in figure.xticks if x_min - tolerance <= tick <= x_max + tolerance])


def set_axis_labels(fig: plt.Figure, axes: np.ndarray, xlabel: str, ylabel: str, columns: bool) -> None:
    """Axis labels: once per figure; labels shared by several axes span all of them."""
    # Labels set while drawing (e.g. by seaborn) are replaced.
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
    if columns:
        axes[0].set_ylabel(ylabel)
        if len(axes) > 1:
            fig.supxlabel(xlabel)
        else:
            axes[0].set_xlabel(xlabel)
    else:
        # The y label is longer than one facet.
        if len(axes) > 1:
            fig.supylabel(ylabel)
        else:
            axes[0].set_ylabel(ylabel)
        axes[-1].set_xlabel(xlabel)


def axes_legend(axes: np.ndarray, label_order: list[str]) -> tuple[list, list]:
    """Labeled artists of all axes, one per label, in label_order (other labels after them, as drawn).

    An axes lacks a method without data at its sweep value, so the first axes is not enough.
    """
    entries = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(label, handle)
    labels = sorted(entries, key=lambda label: label_order.index(label) if label in label_order else len(label_order))
    return [entries[label] for label in labels], labels


def combined_legend(styles: dict[str, dict], value_labels: list[str]) -> tuple[list, list]:
    """Legend of a combined figure: every method in its style, then the sweep values as shades.

    styles: line style per method label, in legend order.
    """
    shades = value_shades(VALUE_LEGEND_COLOR, len(value_labels))
    handles = [Line2D([], [], **style) for style in styles.values()]
    handles += [Line2D([], [], color=shade) for shade in shades]
    return handles, [*styles, *value_labels]


def legend_row_order(items: list, ncol: int) -> list:
    """Reorder items so that a legend (filled column by column) reads row by row."""
    # The first num_large_columns columns hold one entry more than the others.
    nrows, num_large_columns = divmod(len(items), ncol)
    rows_per_column = [nrows + 1] * num_large_columns + [nrows] * (ncol - num_large_columns)
    return [
        items[row * ncol + column]
        for column, column_rows in enumerate(rows_per_column)
        for row in range(column_rows)
    ]


def add_figure_legend(fig: plt.Figure, handles: list, labels: list[str]) -> None:
    """One legend above all axes, the same in every figure.

    The legend gets as few rows as fit into the figure width, the entries read
    row by row (first entry top left).
    """
    if not handles:
        return

    def make_legend(ncol: int):
        """Figure legend with ncol columns, read row by row."""
        return fig.legend(
            legend_row_order(list(handles), ncol),
            legend_row_order(list(labels), ncol),
            loc="outside upper center",
            ncol=ncol,
        )

    renderer = fig.canvas.get_renderer()
    # Most columns first: the first legend that fits the width sets the number of rows.
    for max_ncol in range(len(handles), 0, -1):
        legend = make_legend(max_ncol)
        fits = legend.get_window_extent(renderer).width <= fig.bbox.width
        legend.remove()
        if fits or max_ncol == 1:
            break
    # Same number of rows, entries spread evenly over the columns.
    # -(-a // b) is the ceiling division.
    nrows = -(-len(handles) // max_ncol)
    make_legend(-(-len(handles) // nrows))


def fit_to_figure_width(fig: plt.Figure, text: mpl.text.Text) -> None:
    """Keep a label that exceeds the figure (e.g. narrow figures) inside it.

    The label is split into two lines (at the space nearest the middle); an
    axis label that still exceeds (it is centered on its axes) is shifted
    inward.  Constrained layout does not widen the margins for this.
    """
    renderer = fig.canvas.get_renderer()

    def overflow() -> tuple[float, float]:
        """Pixels by which the label sticks out on the left and on the right (negative: inside)."""
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
    # Shift only a label that sticks out on one side; on both, shifting cannot help.
    if axes is not None and (left > 0) != (right > 0):
        # x of an axis label is in axes coordinates.
        shift = left if left > 0 else -right
        text.set_x(text.get_position()[0] + shift / axes.bbox.width)


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save and close a figure; warn about labels that exceed it."""
    # fig._supxlabel: the text of fig.supxlabel, None if unset (Matplotlib has no public getter).
    for text in [fig._supxlabel] + [ax.xaxis.label for ax in fig.axes]:
        if text is not None:
            fit_to_figure_width(fig, text)

    # No bbox_inches="tight": the file keeps the exact figure size (options
    # width, height); constrained layout already removes the white border.
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
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
        # Largest overflow over the four figure borders, converted from pixels to points.
        overflow = max(-extent.x0, -extent.y0, extent.x1 - fig.bbox.x1,
                       extent.y1 - fig.bbox.y1) * 72 / fig.dpi
        if overflow > 0.5:
            print(f"Warning: {path.name}: \"{text.get_text()}\" exceeds the figure by {overflow:.0f} pt.")
    plt.close(fig)
