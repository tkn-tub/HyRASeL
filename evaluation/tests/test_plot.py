import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import plot
import plot_style
from config import DEFAULTS_PATH, load_config
from plot_types import Ecdf, Line, metric_table, sample_table


def write_evaluation_dir(directory: Path) -> None:
    """Evaluation directory with two speeds, CENTRALIZED (with t-TWR metrics) and TWR."""
    rows = [
        {"Method": method, "speed": speed, "metric": metric, "aggregate": aggregate,
         "value": value * speed, "n_runs": 3}
        for method in ("CENTRALIZED", "TWR")
        for speed in (5, 10)
        for metric in ("mean_error", "mean_error_t_twr")
        if method == "CENTRALIZED" or metric == "mean_error"
        for aggregate, value in (("mean", 0.1), ("std", 0.02), ("sem", 0.01))
    ]
    pd.DataFrame(rows).to_csv(directory / "statistics_run_aggregated.csv", index=False)
    for speed in (5, 10):
        run_dir = directory / "runs" / f"speed={speed},Method=CENTRALIZED" / "rep0"
        run_dir.mkdir(parents=True)
        pd.DataFrame({
            "Method": "CENTRALIZED", "Rep": 0, "speed": speed,
            "est_position_error": [0.1, 0.2, 0.3], "localization_method": ["self_loc", "t_twr", "self_loc"],
        }).to_csv(run_dir / "localization_events_filtered.csv", index=False)
        pd.DataFrame({"Method": "CENTRALIZED", "Rep": 0, "speed": speed, "availability": [0.5, 1.0]}).to_csv(
            run_dir / "availability.csv", index=False
        )


class PlotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.path = Path(self.directory.name)
        write_evaluation_dir(self.path)
        self.data = plot.Data(self.path, "speed", "Speed")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_figure_options_override_type_defaults_and_plots_defaults(self) -> None:
        options = plot.figure_options(
            "error", {"type": "line", "metric": "mean_error"}, Line(), {"errorbar": "std", "width": 0.5}
        )
        self.assertEqual((options["metric"], options["errorbar"], options["width"]), ("mean_error", "std", 0.5))

    def test_unknown_and_missing_options_are_errors(self) -> None:
        with self.assertRaisesRegex(SystemExit, "unknown option"):
            plot.figure_options("error", {"type": "line", "metric": "mean_error", "colour": "red"}, Line(), {})
        with self.assertRaisesRegex(SystemExit, "missing option.*metric"):
            plot.figure_options("error", {"type": "line"}, Line(), {})

    def test_method_part_uses_metric_with_part_suffix(self) -> None:
        table = metric_table(self.data, "mean_error", ["CENTRALIZED.t_twr", "TWR.t_twr", "TWR"])
        self.assertEqual(sorted(set(table["method"])), ["CENTRALIZED.t_twr", "TWR"])
        self.assertEqual(list(table["sweep"].unique()), [5, 10])

    def test_method_part_keeps_events_of_its_localization_method(self) -> None:
        events = sample_table(
            self.data, "localization_events_filtered.csv", "est_position_error", ["CENTRALIZED", "CENTRALIZED.t_twr"]
        )
        self.assertEqual(events.groupby("method").size().to_dict(), {"CENTRALIZED": 6, "CENTRALIZED.t_twr": 2})
        availability = sample_table(self.data, "availability.csv", "availability", ["CENTRALIZED.t_twr"])
        self.assertTrue(availability.empty)

    def test_ci95_uses_student_t_quantile(self) -> None:
        figure = SimpleNamespace(name="error", metric="mean_error", errorbar="ci95", methods=["TWR"])
        table = Line().prepare(self.data, figure)
        # t quantile 0.975 for 2 degrees of freedom
        self.assertAlmostEqual(table["yerr"].iloc[0], 0.01 * 5 * 4.302653, places=5)

    def test_single_repetition_has_no_error_bars(self) -> None:
        # std and sem of one repetition are empty in the aggregated file.
        rows = [
            {"Method": "TWR", "speed": speed, "metric": "mean_error", "aggregate": aggregate, "value": value,
             "n_runs": 1}
            for speed in (5, 10)
            for aggregate, value in (("mean", 0.1), ("std", None), ("sem", None))
        ]
        pd.DataFrame(rows).to_csv(self.path / "statistics_run_aggregated.csv", index=False)
        data = plot.Data(self.path, "speed", "Speed")
        for errorbar in ("sem", "std", "ci95"):
            figure = SimpleNamespace(name="error", metric="mean_error", errorbar=errorbar, methods=["TWR"])
            table = Line().prepare(data, figure)
            self.assertNotIn("yerr", table)
            self.assertEqual(table["y"].tolist(), [0.1, 0.1])

    def test_layouts_write_one_file_per_figure_or_value(self) -> None:
        plots = load_config(DEFAULTS_PATH)["plots"]
        style = plot_style.Style(plots, ["CENTRALIZED"])
        for name, layout in (("ecdf", "facets"), ("ecdf_values", "separate")):
            entry = {"type": "ecdf", "metric": "error", "layout": layout, "methods": ["CENTRALIZED"]}
            plot.plot_figure(name, entry, plots, self.data, style, self.path / "figures", "png")
        files = sorted(path.name for path in (self.path / "figures").iterdir())
        self.assertEqual(files, ["Speed_ecdf.png", "Speed_ecdf_values_10.png", "Speed_ecdf_values_5.png"])

    def test_combined_shades_follow_the_sweep_value(self) -> None:
        plots = load_config(DEFAULTS_PATH)["plots"]
        style = plot_style.Style(plots, ["CENTRALIZED", "TWR"])
        rows = pd.DataFrame({"method": ["CENTRALIZED"] * 3 + ["TWR"] * 2, "sweep": [1, 2, 3, 2, 3]})
        figure = SimpleNamespace(methods=["CENTRALIZED", "TWR"], values=[1, 2, 3])
        series = plot.make_series(rows, figure, style, markers=False, combined=True)
        shades = plot_style.value_shades(style.method("TWR", False)["color"], 3)
        self.assertEqual([s.style["color"] for s in series if s.method == "TWR"], shades[1:])

    def test_missing_or_empty_file_is_an_error(self) -> None:
        with self.assertRaisesRegex(SystemExit, "No statistics_run_aggregated.csv"):
            metric_table(plot.Data(self.path / "missing", "speed", "Speed"), "mean_error", ["TWR"])
        (self.path / "runs" / "speed=5,Method=CENTRALIZED" / "rep0" / "coverage.csv").write_text("")
        with self.assertRaisesRegex(SystemExit, "Empty file"):
            self.data.read("coverage.csv")

    def test_legend_lists_the_methods_of_all_axes(self) -> None:
        fig, axes = plt.subplots(2)
        axes[0].plot([0], label="TWR")
        axes[1].plot([0], label="TWR")
        axes[1].plot([0], label="HyRASeL")
        self.assertEqual(plot_style.axes_legend(axes, ["HyRASeL", "TWR"])[1], ["HyRASeL", "TWR"])
        plt.close(fig)

    def test_ecdf_rejects_unknown_metric(self) -> None:
        figure = SimpleNamespace(name="x", metric="speed", methods=["CENTRALIZED"])
        with self.assertRaisesRegex(SystemExit, "metric must be one of"):
            Ecdf().prepare(self.data, figure)


class SourcesTest(unittest.TestCase):
    """Plot-only config that reads two source configs A and B from results/<Config>/huber/profile."""

    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        root = Path(self.directory.name)
        for source in ("A", "B"):
            (root / "results" / source / "huber" / "profile").mkdir(parents=True)
            write_evaluation_dir(root / "results" / source / "huber" / "profile")
            (root / "configs" / f"{source}.yml").parent.mkdir(exist_ok=True)
            (root / "configs" / f"{source}.yml").write_text("sweep:\n  speed: [5, 10]\n")
        self.config_path = root / "configs" / "Comparison.yml"
        self.config_path.write_text("sources:\n  A: [TWR]\n  B: {CENTRALIZED: CENTRALIZED_VARIANT}\n")
        self.evaluation_dir = root / "results" / "Comparison" / "huber" / "profile"

    def tearDown(self) -> None:
        self.directory.cleanup()

    def data(self) -> plot.Data:
        """Data of the plot-only config."""
        config = load_config(self.config_path)
        sources = plot.source_dirs(config, self.evaluation_dir)
        return plot.Data(self.evaluation_dir, plot.sweep_parameter(config), config["name"], sources)

    def test_sources_keep_their_methods_under_the_given_names(self) -> None:
        table = metric_table(self.data(), "mean_error", ["TWR", "CENTRALIZED_VARIANT.t_twr", "CENTRALIZED"])
        self.assertEqual(sorted(set(table["method"])), ["CENTRALIZED_VARIANT.t_twr", "TWR"])
        self.assertEqual(sorted(table["sweep"].unique()), [5, 10])

    def test_missing_method_or_different_sweep_is_an_error(self) -> None:
        self.config_path.write_text("sources:\n  A: [DISTRIBUTED]\n")
        with self.assertRaisesRegex(SystemExit, "No method DISTRIBUTED"):
            self.data().read("statistics_run_aggregated.csv")
        self.config_path.with_name("B.yml").write_text("sweep:\n  numNodes: [4, 9]\n")
        self.config_path.write_text("sources:\n  A: [TWR]\n  B: [CENTRALIZED]\n")
        with self.assertRaisesRegex(SystemExit, "different parameters"):
            self.data()

    def test_a_name_in_several_sources_is_an_error(self) -> None:
        self.config_path.write_text("sources:\n  A: [CENTRALIZED]\n  B: [CENTRALIZED]\n")
        with self.assertRaisesRegex(SystemExit, "several sources"):
            self.data()


if __name__ == "__main__":
    unittest.main()
