"""Figures of the statistics written by extract_statistics.py.

Reads one evaluation directory (or, for a plot-only config with sources, those
of other configs) and draws every figure in plots.figures of the config (see
README.md) to <evaluation_dir>/figures/<Config>_<figure>.<format>.
The look of all figures is set in plot_style.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import pandas as pd

import plot_style
from config import load_config
from plot_types import PLOT_TYPES, PlotType
from runs import RUNS_DIR_NAME

LAYOUTS = ("combined", "facets", "separate")


class Data:
    """CSV files of one evaluation directory, each read once.

    sources: {evaluation directory: {method: name in the figures}} to read
    instead, each restricted to its methods (plot-only config); default: all
    methods of evaluation_dir.  Every table gets the column sweep: the value of
    the sweep parameter, or the config name without sweep parameter.
    """

    def __init__(
        self, evaluation_dir: Path, parameter: str | None, config_name: str,
        sources: dict[Path, dict[str, str]] | None = None,
    ):
        self.evaluation_dir = evaluation_dir
        self.parameter = parameter
        self.config_name = config_name
        self.sources = sources or {evaluation_dir: None}
        self.tables = {}

    def read(self, file_name: str, columns: list[str] | None = None) -> pd.DataFrame:
        """A file of the evaluation directory, or the files of all runs (runs/<parameters>/repN/).

        columns: only these columns (if present) and Method, Rep and the sweep
        parameter, e.g. for the large event files.
        """
        key = (file_name, tuple(columns) if columns else None)
        if key not in self.tables:
            self.tables[key] = self._load(file_name, columns)
        return self.tables[key]

    def _load(self, file_name: str, columns: list[str] | None) -> pd.DataFrame:
        """Read and concatenate the files of one name of all sources; exit if a file or method is missing."""
        keep = None if columns is None else {*columns, "Method", "Rep", self.parameter}
        # A callable instead of a list: requested columns that a file lacks are no error.
        usecols = None if keep is None else (lambda column: column in keep)
        tables = []
        for directory, methods in self.sources.items():
            table = self._read_files(directory, file_name, usecols)
            if methods is not None:
                missing = set(methods) - set(table["Method"])
                if missing:
                    raise SystemExit(f"No method {', '.join(sorted(missing))} in {file_name} below {directory}")
                table = table.loc[table["Method"].isin(methods)].assign(Method=lambda t: t["Method"].map(methods))
            tables.append(table)
        table = pd.concat(tables, ignore_index=True)
        if self.parameter and self.parameter not in table:
            raise SystemExit(f"{file_name} has no column {self.parameter} (sweep parameter of the config)")
        table["sweep"] = table[self.parameter] if self.parameter else self.config_name
        return table

    @staticmethod
    def _read_files(directory: Path, file_name: str, usecols) -> pd.DataFrame:
        """All files of one name of an evaluation directory, concatenated; exit if there is none or one is empty."""
        # Aggregated files lie in the evaluation directory, per-run files below runs/.
        path = directory / file_name
        paths = [path] if path.exists() else sorted((directory / RUNS_DIR_NAME).rglob(file_name))
        if not paths:
            raise SystemExit(f"No {file_name} below {directory} (run the extraction first)")
        tables = []
        for path in paths:
            try:
                tables.append(pd.read_csv(path, usecols=usecols))
            except pd.errors.EmptyDataError:
                # The extraction writes at least the header.
                raise SystemExit(f"Empty file (aborted extraction?): {path}") from None
        return pd.concat(tables, ignore_index=True)


@dataclass
class Series:
    """Rows of one method (and sweep value) with its legend label and line style."""

    method: str
    label: str | None
    style: dict
    data: pd.DataFrame


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("evaluation_dir", type=Path, help="Evaluation directory <Config>/<estimator>/<filter_profile>")
    parser.add_argument("--config", type=Path, required=True, help="Config file, e.g. configs/Speed.yml")
    parser.add_argument("--output-dir", type=Path, help="Figure directory (default: <evaluation_dir>/figures)")
    parser.add_argument("--format", choices=("pdf", "png", "svg"), help="Format of all figures (overrides the config)")
    parser.add_argument("--figures", nargs="+", metavar="FIGURE", help="Only these figures of plots.figures")
    return parser.parse_args()


def sweep_parameter(config: dict) -> str | None:
    """Sweep parameter of the config (plot-only config: of its sources); None without sweep."""
    sweeps = [config["sweep"]]
    if config["sources"]:
        # The source configs lie next to the config.
        sweeps = [load_config(Path(config["path"]).with_name(f"{source}.yml"))["sweep"] for source in config["sources"]]
    if len({tuple(sweep) for sweep in sweeps}) > 1:
        raise SystemExit(f"{config['name']}: the sources sweep different parameters: {[list(sweep) for sweep in sweeps]}")
    if len(sweeps[0]) > 1:
        raise SystemExit(f"{config['name']}: figures support at most one sweep parameter, got {list(sweeps[0])}")
    return next(iter(sweeps[0]), None)


def source_dirs(config: dict, evaluation_dir: Path) -> dict[Path, dict[str, str]] | None:
    """Evaluation directory and methods ({method: name in the figures}) of every source of a plot-only config.

    A source is read with the estimator and filter profile of evaluation_dir:
    <results>/<source>/<estimator>/<filter_profile>.  None without sources.
    """
    if not config["sources"]:
        return None
    names = [name for methods in config["sources"].values() for name in methods.values()]
    if len(names) != len(set(names)):
        raise SystemExit(f"{config['name']}: a method name occurs in several sources, rename it: {names}")
    results, profile = evaluation_dir.parents[2], evaluation_dir.relative_to(evaluation_dir.parents[1])
    return {results / source / profile: methods for source, methods in config["sources"].items()}


def figure_options(name: str, entry: dict, plot_type: PlotType, defaults: dict) -> dict:
    """Options of one figure: plot type defaults < plots.defaults < the figure entry."""
    unknown = set(entry) - {"type", *plot_type.options, *defaults}
    if unknown:
        raise SystemExit(f"{name}: unknown option(s) {', '.join(sorted(unknown))}")
    options = {**plot_type.options, **defaults, **entry}
    missing = [key for key in plot_type.options if options[key] is None]
    if missing:
        raise SystemExit(f"{name}: missing option(s) {', '.join(missing)}")
    if plot_type.layouts and options["layout"] not in LAYOUTS:
        raise SystemExit(f"{name}: layout must be one of {', '.join(LAYOUTS)}")
    return options


def make_series(rows: pd.DataFrame, figure, style: plot_style.Style, markers: bool, combined: bool) -> list[Series]:
    """Series of one axes in legend order (option methods).

    combined: one series per method and sweep value, in shades of the method
    color (smallest value lightest) and without label (see combined_legend).
    """
    series = []
    for method in figure.methods:
        method_rows = rows.loc[rows["method"] == method]
        if method_rows.empty:
            continue
        method_style = style.method(method, markers)
        if not combined:
            series.append(Series(method, style.label(method), method_style, method_rows))
            continue
        # Shades of all sweep values of the figure, as in the legend.
        shades = dict(zip(figure.values, plot_style.value_shades(method_style["color"], len(figure.values))))
        for value, value_rows in method_rows.groupby("sweep", sort=True):
            series.append(Series(method, None, {**method_style, "color": shades[value]}, value_rows))
    return series


def axis_label(spec: str, figure, style: plot_style.Style) -> str:
    """Axis label of a plot type: "sweep", "metric" or a fixed text (see PlotType.axes)."""
    if spec == "sweep":
        return style.label(figure.parameter) if figure.parameter else ""
    if spec == "metric":
        return style.label(figure.metric)
    return spec


def draw_figure(
    path: Path,
    panels: list[tuple[str, pd.DataFrame]],
    plot_type: PlotType,
    figure,
    style: plot_style.Style,
    columns: bool = False,
    combined: bool = False,
) -> None:
    """Draw one axes per panel (title, rows) and save the figure.

    columns: axes next to each other instead of under one another; combined:
    all sweep values in one axes (see make_series).
    """
    # With a single sweep value, there is nothing to combine.
    combined = combined and len(figure.values) > 1
    # The font sizes of this figure apply only within the context.
    with mpl.rc_context(plot_style.font_rc(figure)):
        fig, axes = plot_style.create_figure(figure, len(panels), columns)
        for ax, (title, rows) in zip(axes, panels):
            plot_type.draw(ax, make_series(rows, figure, style, plot_type.markers, combined), figure)
            ax.set_title(title)
            plot_style.apply_axis_options(ax, figure)
        # Options xlabel and ylabel override the labels of the plot type.
        plot_style.set_axis_labels(
            fig, axes,
            figure.xlabel if figure.xlabel is not None else axis_label(plot_type.axes[0], figure, style),
            figure.ylabel if figure.ylabel is not None else axis_label(plot_type.axes[1], figure, style),
            columns,
        )
        # Legend of the plot type, else methods and value shades (combined), else the lines of all axes.
        legend = plot_type.legend(fig)
        if legend is None and combined:
            methods = [method for method in figure.methods if method in set(panels[0][1]["method"])]
            legend = plot_style.combined_legend(
                {style.label(method): style.method(method, plot_type.markers) for method in methods},
                [style.value_label(figure.parameter, value) for value in figure.values],
            )
        elif legend is None:
            legend = plot_style.axes_legend(axes, [style.label(method) for method in figure.methods])
        plot_style.add_figure_legend(fig, *legend)
        plot_style.save_figure(fig, path)


def plot_figure(name: str, entry: dict, plots: dict, data: Data, style: plot_style.Style, output_dir: Path,
                file_format: str | None) -> None:
    """Prepare and draw one figure of plots.figures."""
    if not isinstance(entry, dict) or entry.get("type") not in PLOT_TYPES:
        raise SystemExit(f"{name}: type must be one of {', '.join(PLOT_TYPES)}")
    plot_type = PLOT_TYPES[entry["type"]]()
    # The options as attributes (figure.metric, figure.width, ...), with name and sweep parameter.
    figure = SimpleNamespace(
        **figure_options(name, entry, plot_type, plots["defaults"]), name=name, parameter=data.parameter
    )
    if file_format:
        figure.format = file_format

    table = plot_type.prepare(data, figure)
    if table.empty:
        print(f"Skipping {name}: no data.")
        return
    # Sweep values with data, for facets, shades and legend.
    figure.values = sorted(table["sweep"].unique())

    def path(suffix: str = "") -> Path:
        """File of the figure: <Config>_<figure><suffix>.<format>."""
        return output_dir / f"{data.config_name}_{name}{suffix}.{figure.format}"

    def panel(value) -> tuple[str, pd.DataFrame]:
        """Facet title and rows of one sweep value."""
        return style.facet_title(figure.parameter, value), table.loc[table["sweep"] == value]

    # One axes per method, next to each other; the facet title is left out for a single method.
    if plot_type.facet_by_method:
        methods = [method for method in figure.methods if method in set(table["method"])]
        panels = [
            (style.label(method) if len(methods) > 1 else "", table.loc[table["method"] == method])
            for method in methods
        ]
        draw_figure(path(), panels, plot_type, figure, style, columns=True)
    # Sweep values on the x axis: a single axes.
    elif not plot_type.layouts:
        draw_figure(path(), [("", table)], plot_type, figure, style)
    elif figure.layout == "combined":
        draw_figure(path(), [(style.facet_title(figure.parameter), table)], plot_type, figure, style, combined=True)
    elif figure.layout == "facets":
        draw_figure(path(), [panel(value) for value in figure.values], plot_type, figure, style)
    # Layout separate: one file per sweep value, named <Config>_<figure>_<value>.
    else:
        for value in figure.values:
            draw_figure(path(f"_{plot_style.format_value(value)}"), [panel(value)], plot_type, figure, style)


def main() -> None:
    """Draw all figures of the config."""
    args = parse_args()
    config = load_config(args.config)
    plots = config["plots"]
    figures = plots.get("figures") or {}
    # Styles of the methods of all figures, so that they do not depend on --figures.
    methods = [
        *plots["defaults"]["methods"],
        *(method for entry in figures.values() for method in (entry or {}).get("methods", [])),
    ]
    style = plot_style.Style(plots, methods)
    if args.figures:
        unknown = set(args.figures) - set(figures)
        if unknown:
            raise SystemExit(f"Not in plots.figures: {', '.join(sorted(unknown))}")
        figures = {name: entry for name, entry in figures.items() if name in args.figures}

    evaluation_dir = args.evaluation_dir.expanduser().resolve()
    # A plot-only config has no extraction of its own; its directory only receives the figures.
    if not config["sources"] and not evaluation_dir.is_dir():
        raise SystemExit(f"evaluation_dir is not a directory: {evaluation_dir}")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else evaluation_dir / "figures"
    data = Data(evaluation_dir, sweep_parameter(config), config["name"], source_dirs(config, evaluation_dir))
    print(f"Figure output directory: {output_dir}")
    for name, entry in figures.items():
        plot_figure(name, entry, plots, data, style, output_dir, args.format)


if __name__ == "__main__":
    main()
