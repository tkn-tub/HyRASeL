import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

# The modules of the evaluation chain live in src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import extract_statistics
import metrics
from runs import EVENTS_FILE_NAME, POSTPROCESSED_DIR_NAME, RUN_METADATA_FILE_NAME
from config_defaults import default_args


def _filter(methods=metrics.LOCALIZATION_METHODS, error_dimension="3d") -> extract_statistics.EventFilter:
    """Filter with the default limits."""
    return extract_statistics.EventFilter(error_dimension, 2.0, 3.0, True, tuple(methods))


class FilterProfileTest(unittest.TestCase):
    def test_all_methods_in_any_order_have_no_suffix(self) -> None:
        self.assertEqual(_filter(("self_loc", "t_twr", "twr")).profile, "pdop-2_mad-3_stitch")
        self.assertEqual(_filter(("twr", "self_loc", "t_twr")).profile, "pdop-2_mad-3_stitch")

    def test_method_subsets_have_a_suffix_in_canonical_order(self) -> None:
        self.assertEqual(_filter(("t_twr",)).profile, "pdop-2_mad-3_stitch_loc-t_twr")
        self.assertEqual(_filter(("twr", "self_loc")).profile, "pdop-2_mad-3_stitch_loc-self_loc-twr")
        self.assertEqual(_filter(("self_loc", "t_twr")).profile, "pdop-2_mad-3_stitch_loc-self_loc-t_twr")

    def test_profile_names_the_filtered_dop(self) -> None:
        self.assertEqual(_filter(error_dimension="2d").profile, "hdop-2_mad-3_stitch")
        self.assertEqual(_filter(error_dimension="3d").profile, "pdop-2_mad-3_stitch")

    def test_filter_profile_of_the_arguments(self) -> None:
        arguments = [*default_args("extract"), "--config-path", "x"]
        self.assertEqual(extract_statistics.filter_profile(arguments), "hdop-2_mad-3_stitch")


class ArgumentsTest(unittest.TestCase):
    def test_localization_methods(self) -> None:
        arguments = [*default_args("extract"), "--config-path", "x", "--localization-methods", "self_loc", "t_twr"]
        self.assertEqual(
            extract_statistics._argument_parser().parse_args(arguments).localization_methods, ["self_loc", "t_twr"]
        )

    def test_abbreviated_options_are_rejected(self) -> None:
        parser = extract_statistics._argument_parser()
        arguments = [*default_args("extract"), "--config-path", "x"]
        self.assertEqual(parser.parse_args([*arguments, "--pdop-max", "5"]).pdop_max, 5.0)
        with self.assertRaises(SystemExit):
            parser.parse_args([*arguments, "--pdop", "5"])


class SelectTest(unittest.TestCase):
    @staticmethod
    def _events() -> pd.DataFrame:
        return pd.DataFrame({
            "localization_method": ["self_loc", "t_twr", "twr"],
            "est_position_error": [1.0, 2.0, 3.0],
            "est_position_error_2d": [0.5, np.nan, 2.5],
            "PDOP": [1.0, 1.0, 1.0],
            "HDOP": [0.4, 0.4, 5.0],
        })

    def test_selection_combines_the_evaluated_methods(self) -> None:
        selected = _filter(("self_loc", "t_twr")).select(self._events())
        self.assertEqual(selected["localization_method"].tolist(), ["self_loc", "t_twr"])

    def test_2d_selection_uses_2d_error_and_hdop(self) -> None:
        selected = _filter(error_dimension="2d").select(self._events())
        self.assertEqual(selected["est_position_error"].tolist(), [0.5, 2.5])
        self.assertEqual(selected["PDOP"].tolist(), [0.4, 5.0])
        self.assertEqual(selected["est_position_error_3d"].tolist(), [1.0, 3.0])

    def test_2d_selection_requires_2d_columns(self) -> None:
        with self.assertRaises(ValueError):
            _filter(error_dimension="2d").select(self._events().drop(columns=["HDOP"]))


class RunStatisticsTest(unittest.TestCase):
    @staticmethod
    def _statistics(events: pd.DataFrame, num_nodes: int = 4, end_time: float = 0.5) -> pd.DataFrame:
        """Metrics of a run whose events all pass the filter, by metric."""
        events = metrics.add_localization_interval(events)
        coverage = metrics.coverage_table(events, 0.1, num_nodes, end_time)
        availability = metrics.availability_table(events, 0.1, num_nodes, end_time)
        statistics = metrics.run_statistics(events, events, coverage, availability, num_nodes, end_time, None)
        return pd.DataFrame(statistics).set_index("metric")

    def test_error_statistics_use_every_evaluated_event(self) -> None:
        events = pd.DataFrame({
            "NodeId": [5, 5],
            "event_time": [0.0, 0.1],
            "localization_method": ["self_loc", "t_twr"],
            "est_position_error": [1.0, 3.0],
        })
        statistics = self._statistics(events)
        self.assertEqual(statistics.loc["mean_error", "value"], 2.0)
        self.assertEqual(statistics.loc["mean_error", "n"], 2)
        # 2 updates / (4 simulated nodes * 0.5 s)
        self.assertEqual(statistics.loc["update_rate", "value"], 1.0)
        self.assertEqual(statistics.loc["update_rate", "unit"], "1/s")

    def test_statistics_and_interval_per_localization_method(self) -> None:
        events = pd.DataFrame({
            "NodeId": [5, 5, 5, 5],
            "event_time": [0.0, 1.0, 1.5, 3.0],
            "localization_method": ["t_twr", "self_loc", "self_loc", "t_twr"],
            "est_position_error": [1.0, 0.2, 0.4, 3.0],
        })
        statistics = self._statistics(events, end_time=3.0)
        self.assertAlmostEqual(statistics.loc["mean_error_t_twr", "value"], 2.0)
        self.assertAlmostEqual(statistics.loc["mean_error_self_loc", "value"], 0.3)
        # Interval between consecutive events of the same node and method only.
        self.assertAlmostEqual(statistics.loc["mean_localization_interval_t_twr", "value"], 3.0)
        self.assertAlmostEqual(statistics.loc["mean_localization_interval_self_loc", "value"], 0.5)
        self.assertEqual(statistics.loc["mean_error_t_twr", "unit"], "m")
        self.assertEqual(statistics.loc["mean_localization_interval_t_twr", "unit"], "s")
        self.assertNotIn("mean_error_twr", statistics.index)


class CoverageAvailabilityTest(unittest.TestCase):
    def test_unlocalized_nodes_and_full_time_axis_count_as_uncovered(self) -> None:
        # 4 simulated nodes, 10 windows of 0.1 s; node 5 is localized in two
        # windows, node 6 in one, nodes without events never.
        events = pd.DataFrame({"NodeId": [5, 5, 5, 6], "event_time": [0.01, 0.02, 0.55, 0.15]})
        coverage = metrics.coverage_table(events, 0.1, 4, 0.95)
        availability = metrics.availability_table(events, 0.1, 4, 0.95)

        self.assertEqual(len(coverage), 10)
        self.assertEqual(sorted(availability["availability"]), [0.0, 0.0, 0.1, 0.2])
        self.assertAlmostEqual(coverage["coverage"].mean(), availability["availability"].mean())

    def test_coverage_spans_zero_to_end_time_with_empty_windows(self) -> None:
        events = pd.DataFrame({"event_time": [0.05, 0.06, 0.31], "NodeId": [1, 2, 1]})
        coverage = metrics.coverage_table(events, 0.1, 4, 0.5)

        self.assertEqual(len(coverage), 6)
        self.assertAlmostEqual(coverage["time_bin_start"].iloc[0], 0.0)
        self.assertEqual(coverage["localized_nodes"].tolist(), [2, 0, 0, 1, 0, 0])
        self.assertEqual(coverage["coverage"].tolist(), [0.5, 0, 0, 0.25, 0, 0])

    def test_run_without_events_has_zero_availability(self) -> None:
        events = pd.DataFrame({"NodeId": [], "event_time": []})
        availability = metrics.availability_table(events, 0.1, 3, 0.95)
        self.assertEqual(list(availability["availability"]), [0.0, 0.0, 0.0])


class MainOutputTest(unittest.TestCase):
    @staticmethod
    def _write_run(config_path: Path, parameters: str, tdoa_loss: str = "huber") -> None:
        """Post-processed run rep0 with 4 self-localizations of 2 nodes."""
        run_path = config_path / POSTPROCESSED_DIR_NAME / parameters / "rep0"
        run_path.mkdir(parents=True)
        events = pd.DataFrame({
            "event_time": [0.01, 0.05, 0.01, 0.05],
            "NodeId": [5, 5, 6, 6],
            "srcId": 1,
            "localization_method": "self_loc",
            "using_anchor_stitching": False,
            "est_position_error": [0.1, 0.11, 0.12, 0.13],
            "residual_mad_score": 1.0,
            "PDOP": 1.0,
        })
        events.to_csv(run_path / EVENTS_FILE_NAME, index=False)
        metadata = {"simulation": {"end_time": 0.1, "num_nodes": 2}, "processing_settings": {"tdoa_loss": tdoa_loss}}
        (run_path / RUN_METADATA_FILE_NAME).write_text(json.dumps(metadata))

    def test_main_writes_the_evaluation_dir_of_the_estimator(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "example"
            self._write_run(config_path, "numSlots=2,Method=TWR")
            extract_statistics.main([
                *default_args("extract"), "--config-path", str(config_path),
                "--error-dimension", "3d", "--localization-methods", "self_loc",
            ])

            evaluation_dir = config_path / "huber" / "pdop-2_mad-3_stitch_loc-self_loc"
            self.assertTrue((evaluation_dir / "statistics_run_all.csv").exists())
            self.assertTrue((evaluation_dir / "statistics_run_aggregated.csv").exists())
            run_dir = evaluation_dir / "runs" / "numSlots=2,Method=TWR" / "rep0"
            filtered = pd.read_csv(run_dir / "localization_events_filtered.csv")
            self.assertEqual(len(filtered), 4)
            self.assertEqual(set(filtered["tdoa_estimator"]), {"huber"})
            # The postprocessed level is not part of the sweep metadata.
            self.assertEqual(set(filtered["Config"]), {"example"})
            self.assertEqual(set(filtered["numSlots"]), {2})

    def test_runs_of_different_estimators_are_an_error(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "example"
            self._write_run(config_path, "Method=TWR", "huber")
            self._write_run(config_path, "Method=CENTRALIZED", "linear")
            with self.assertRaises(SystemExit):
                extract_statistics.main([*default_args("extract"), "--config-path", str(config_path)])


class MainBehaviorTest(unittest.TestCase):
    """Behavior of main() seen in its output files: filter rules, aborted and empty runs."""

    ARGUMENTS = [
        "--coverage-window-size", "0.1", "--error-dimension", "2d", "--pdop-max", "2",
        "--residual-mad-max", "3", "--localization-methods", "self_loc", "t_twr", "twr",
    ]

    @staticmethod
    def _events(**columns) -> pd.DataFrame:
        """Events of node 5 with one column per keyword (lists of equal length), seqNo numbers the rows."""
        num_events = len(next(iter(columns.values())))
        events = pd.DataFrame({
            "event_time": np.arange(num_events) * 0.01,
            "NodeId": 5,
            "seqNo": np.arange(num_events),
            "srcId": 1,
            "localization_method": "self_loc",
            "using_anchor_stitching": False,
            "est_position_error": 0.5,
            "est_position_error_2d": 0.4,
            "PDOP": 1.0,
            "HDOP": 1.0,
            "residual_mad_score": 1.0,
        })
        for name, values in columns.items():
            events[name] = values
        return events

    @staticmethod
    def _write_run(config_path: Path, rep: int, events: pd.DataFrame, end_time: float = 1.0) -> None:
        """Post-processed TWR run with 2 simulated nodes."""
        run_path = config_path / "postprocessed" / "Method=TWR" / f"rep{rep}"
        run_path.mkdir(parents=True)
        events.to_csv(run_path / "localization_events.csv", index=False)
        metadata = {
            "simulation": {"end_time": end_time, "num_nodes": 2},
            "processing_settings": {"tdoa_loss": "huber"},
        }
        (run_path / "run_metadata.json").write_text(json.dumps(metadata))

    @staticmethod
    def _evaluation_dir(config_path: Path) -> Path:
        """The only evaluation directory of config_path."""
        [evaluation_dir] = (config_path / "huber").iterdir()
        return evaluation_dir

    def _kept_rows(self, events: pd.DataFrame, *options: str) -> list[int]:
        """seqNo of the filtered events of one run with these events."""
        with TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "example"
            self._write_run(config_path, 0, events)
            extract_statistics.main([*self.ARGUMENTS, *options, "--config-path", str(config_path)])
            run_dir = self._evaluation_dir(config_path) / "runs" / "Method=TWR" / "rep0"
            return sorted(pd.read_csv(run_dir / "localization_events_filtered.csv")["seqNo"])

    def test_filter_keeps_values_at_the_limits_and_missing_values(self) -> None:
        events = self._events(
            HDOP=[2.0, 2.0 + 1e-9, np.nan, 1.0, 1.0, 1.0],
            residual_mad_score=[1.0, 1.0, 1.0, 3.0, 3.0 + 1e-9, np.nan],
        )
        self.assertEqual(self._kept_rows(events, "--use-anchor-stitching"), [0, 2, 3, 5])

    def test_filter_of_stitched_events(self) -> None:
        events = self._events(using_anchor_stitching=[False, True, False])
        self.assertEqual(self._kept_rows(events, "--use-anchor-stitching"), [0, 1, 2])
        self.assertEqual(self._kept_rows(events), [0, 2])

    def test_aborted_and_empty_runs(self) -> None:
        events = self._events(HDOP=[1.0, 1.0])
        for skip_empty_runs in (False, True):
            with self.subTest(skip_empty_runs=skip_empty_runs), TemporaryDirectory() as temporary_directory:
                config_path = Path(temporary_directory) / "example"
                self._write_run(config_path, 0, events)
                self._write_run(config_path, 1, events.iloc[:0])
                # Ends before 0.9 x the median end time of the repetitions: aborted simulation.
                self._write_run(config_path, 2, events, end_time=0.85)
                options = ["--skip-empty-runs"] if skip_empty_runs else []
                extract_statistics.main([*self.ARGUMENTS, *options, "--config-path", str(config_path)])

                runs = pd.read_csv(self._evaluation_dir(config_path) / "statistics_run_all.csv")
                runs = runs.set_index(["Rep", "metric"])
                reps = sorted(set(runs.index.get_level_values("Rep")))
                self.assertEqual(reps, [0] if skip_empty_runs else [0, 1])
                # Node 5 of 2 nodes is localized in the first of 11 windows.
                self.assertAlmostEqual(runs.loc[(0, "mean_coverage"), "value"], 0.5 / 11)
                if not skip_empty_runs:
                    self.assertEqual(runs.loc[(1, "mean_coverage"), "value"], 0.0)
                    self.assertTrue(np.isnan(runs.loc[(1, "mean_error"), "value"]))


if __name__ == "__main__":
    unittest.main()
