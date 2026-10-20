import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import sys
from pathlib import Path

# The modules of the evaluation chain live in src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import extract_statistics as statistics
from config_defaults import default_args


class FilterProfileLocalizationMethodsTest(unittest.TestCase):
    def _profile(self, methods: tuple[str, ...]) -> str:
        return statistics._build_filter_profile(
            pdop_max=2.0,
            residual_mad_max=3.0,
            use_anchor_stitching=True,
            num_real_anchors=4,
            only_use_real_anchors_as_reference=False,
            localization_methods=methods,
            error_dimension="3d",
        )

    def test_default_all_methods_has_no_suffix(self) -> None:
        self.assertEqual(
            self._profile(("self_loc", "v_twr", "twr")),
            "pdop-2_mad-3_stitch_all-ref",
        )

    def test_reordered_all_methods_has_no_suffix(self) -> None:
        self.assertEqual(
            self._profile(("twr", "self_loc", "v_twr")),
            "pdop-2_mad-3_stitch_all-ref",
        )

    def test_single_method_has_method_suffix(self) -> None:
        self.assertEqual(
            self._profile(("v_twr",)),
            "pdop-2_mad-3_stitch_all-ref_loc-v_twr",
        )

    def test_method_subset_has_canonical_suffix_order(self) -> None:
        self.assertEqual(
            self._profile(("twr", "self_loc")),
            "pdop-2_mad-3_stitch_all-ref_loc-self_loc-twr",
        )

    def test_self_loc_and_v_twr_have_one_combined_profile(self) -> None:
        self.assertEqual(
            self._profile(("self_loc", "v_twr")),
            "pdop-2_mad-3_stitch_all-ref_loc-self_loc-v_twr",
        )

    def test_self_loc_event_stream_has_own_profile(self) -> None:
        profile = statistics._build_filter_profile(
            pdop_max=2.0,
            residual_mad_max=3.0,
            use_anchor_stitching=True,
            num_real_anchors=4,
            only_use_real_anchors_as_reference=False,
            localization_methods=("self_loc",),
            error_dimension="3d",
        )
        self.assertEqual(
            profile,
            "pdop-2_mad-3_stitch_all-ref_loc-self_loc",
        )


class LocalizationMethodsCliTest(unittest.TestCase):
    def test_one_option_controls_all_statistic_methods(self) -> None:
        parser = statistics._build_argument_parser()

        arguments = parser.parse_args(
            [
                *default_args("extract"), "--config-path", "x",
                "--localization-methods", "self_loc", "v_twr",
            ]
        )

        self.assertEqual(arguments.localization_methods, ["self_loc", "v_twr"])
        self.assertNotIn(
            "--error-localization-methods",
            parser._option_string_actions,
        )


class EstimatorViewTest(unittest.TestCase):
    @staticmethod
    def _results() -> pd.DataFrame:
        return pd.DataFrame({
            "localization_method": ["self_loc", "v_twr", "twr"],
            "est_position_error": [1.0, 2.0, 3.0],
        })

    def test_view_uses_only_the_selected_methods(self) -> None:
        view = statistics._make_estimator_view(self._results(), ("self_loc",), "3d")
        self.assertEqual(view["localization_method"].tolist(), ["self_loc"])

    def test_view_combines_selected_position_streams(self) -> None:
        view = statistics._make_estimator_view(
            self._results(), ("self_loc", "v_twr"), "3d"
        )
        self.assertEqual(
            set(view["localization_method"]), {"self_loc", "v_twr"}
        )

    def test_estimator_name_follows_the_post_processing_settings(self) -> None:
        def name(profile: str, loss: str) -> str:
            return statistics._get_estimator_name(
                {"processing_settings": {
                    "tdoa_solver_profile": profile, "tdoa_loss": loss,
                }}
            )

        self.assertEqual(name("robust", "huber"), "huber")
        self.assertEqual(name("robust", "linear"), "linear")
        self.assertEqual(name("legacy", "huber"), "legacy")


class UnifiedStatisticsStreamTest(unittest.TestCase):
    def test_error_statistics_use_every_selected_event(self) -> None:
        selected_events = pd.DataFrame(
            {
                "NodeId": [5, 5],
                "localization_method": ["self_loc", "v_twr"],
                "est_position_error": [1.0, 3.0],
                "localization_interval": [np.nan, 0.1],
            }
        )
        coverage = pd.DataFrame({"coverage": [1.0]})
        availability = pd.DataFrame({"availability": [1.0]})

        result = statistics._make_run_statistics_df(
            filtered_results_df=selected_events,
            coverage_df=coverage,
            availability_df=availability,
            metadata={},
            len_before_filtering=2,
            len_after_filtering=2,
            total_nodes=4,
            end_time=0.5,
        )
        mean_error = result.loc[result["metric"].eq("mean_error")].iloc[0]

        self.assertEqual(mean_error["value"], 2.0)
        self.assertEqual(mean_error["n"], 2)
        # 2 updates / (4 simulated nodes * 0.5 s)
        update_rate = result.loc[result["metric"].eq("update_rate")].iloc[0]
        self.assertEqual(update_rate["value"], 1.0)
        self.assertEqual(update_rate["unit"], "1/s")


class CoverageAvailabilityTest(unittest.TestCase):
    def test_unlocalized_nodes_and_full_time_axis_count_as_uncovered(self) -> None:
        # 4 simulated nodes, 10 windows of 0.1 s; node 5 is localized in two
        # windows, node 6 in one, nodes without events never.
        events = pd.DataFrame(
            {"NodeId": [5, 5, 5, 6], "event_time": [0.01, 0.02, 0.55, 0.15]}
        )
        kwargs = dict(
            metadata={},
            coverage_window_size=0.1,
            total_nodes=4,
            end_time=0.95,
        )

        coverage = statistics._make_coverage_df(
            filtered_results_df=events, time_column="event_time", **kwargs
        )
        availability = statistics._make_availability_df(
            filtered_results_df=events, **kwargs
        )

        self.assertEqual(len(coverage), 10)
        self.assertEqual(
            sorted(availability["availability"]), [0.0, 0.0, 0.1, 0.2]
        )
        self.assertAlmostEqual(
            coverage["coverage"].mean(), availability["availability"].mean()
        )

    def test_run_without_events_has_zero_coverage(self) -> None:
        events = pd.DataFrame({"NodeId": [], "event_time": []})
        availability = statistics._make_availability_df(
            filtered_results_df=events,
            metadata={},
            coverage_window_size=0.1,
            total_nodes=3,
            end_time=0.95,
        )

        self.assertEqual(list(availability["availability"]), [0.0, 0.0, 0.0])


class EstimatorStatisticsOutputTest(unittest.TestCase):
    @staticmethod
    def _event_rows() -> pd.DataFrame:
        rows = []
        for index, (node_id, event_time) in enumerate(
            ((5, 0.01), (5, 0.05), (6, 0.01), (6, 0.05))
        ):
            row = {
                "event_time": event_time,
                "NodeId": node_id,
                "srcId": 1,
                "localization_method": "self_loc",
                "using_anchor_stitching": False,
                "est_position_x": 1.0,
                "est_position_y": 2.0,
                "est_position_z": 0.0,
                "est_position_error": 0.1 + index * 0.01,
                "residuals": "[0.1 0.2 0.3]",
                "residual_mad_score": 1.0,
                "PDOP": 1.0,
                "cond": 2.0,
                "tdoa_solver_profile": "robust",
                "tdoa_optimizer_method": "trf",
                "tdoa_dimension": 2,
                "tdoa_fixed_z": 0.0,
                "tdoa_loss": "huber",
                "tdoa_f_scale": 1.0,
                "tdoa_initial_PDOP": 1.0,
                "tdoa_initial_cond": 2.0,
                "tdoa_initial_rank": 2,
                "tdoa_rank": 2,
                "localization_bounds": "([-1, -1], [11, 11])",
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def test_main_writes_the_evaluation_dir_of_the_estimator(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "example"
            run_path = (
                config_path
                / statistics.POSTPROCESSED_DIR_NAME
                / "numSlots=2,Method=CENTRALIZED"
                / "rep0"
            )
            run_path.mkdir(parents=True)
            self._event_rows().to_csv(
                run_path / statistics.DATA_FILE_NAME,
                index=False,
            )
            (run_path / statistics.RUN_METADATA_FILE_NAME).write_text(
                json.dumps({
                    "simulation": {"end_time": 0.1, "num_nodes": 2},
                    "processing_settings": {
                        "tdoa_solver_profile": "robust", "tdoa_loss": "huber",
                    },
                })
            )

            statistics.main(
                [
                    *default_args("extract"),
                    "--config-path", str(config_path),
                    "--error-dimension", "3d",
                    "--localization-methods", "self_loc",
                ]
            )

            profile = "pdop-2_mad-3_stitch_all-ref_loc-self_loc"
            evaluation_dir = config_path / "huber" / profile
            self.assertTrue((evaluation_dir / "statistics_run_all.csv").exists())
            self.assertTrue(
                (evaluation_dir / "statistics_run_aggregated.csv").exists()
            )

            run_output = Path("runs") / "numSlots=2,Method=CENTRALIZED" / "rep0"
            selected_events = pd.read_csv(
                evaluation_dir / run_output / "localization_events_filtered.csv"
            )
            self.assertEqual(len(selected_events), 4)
            self.assertEqual(
                set(selected_events["tdoa_estimator"]), {"huber"}
            )
            # The postprocessed level is not part of the sweep metadata.
            self.assertEqual(set(selected_events["Config"]), {"example"})
            self.assertEqual(set(selected_events["numSlots"]), {2})


class ErrorDimensionTest(unittest.TestCase):
    def _results(self) -> pd.DataFrame:
        return pd.DataFrame({
            "localization_method": ["self_loc"] * 3,
            "est_position_error": [1.0, 2.0, 3.0],
            "est_position_error_2d": [0.5, np.nan, 2.5],
            "PDOP": [1.0, 1.0, 1.0],
            "HDOP": [0.4, 0.4, 5.0],
        })

    def test_2d_view_exposes_2d_error_and_hdop(self) -> None:
        view = statistics._make_estimator_view(
            self._results(), ("self_loc",), "2d"
        )

        self.assertEqual(view["est_position_error"].tolist(), [0.5, 2.5])
        self.assertEqual(view["PDOP"].tolist(), [0.4, 5.0])
        self.assertEqual(view["est_position_error_3d"].tolist(), [1.0, 3.0])

    def test_2d_view_requires_2d_columns(self) -> None:
        results = self._results().drop(columns=["HDOP"])
        with self.assertRaises(ValueError):
            statistics._make_estimator_view(results, ("self_loc",), "2d")

    def test_filter_profile_names_the_filtered_dop(self) -> None:
        kwargs = dict(
            pdop_max=2.0,
            residual_mad_max=3.0,
            use_anchor_stitching=True,
            num_real_anchors=4,
            only_use_real_anchors_as_reference=False,
            localization_methods=statistics.ALL_LOCALIZATION_METHODS,
        )
        self.assertEqual(
            statistics._build_filter_profile(**kwargs, error_dimension="2d"),
            "hdop-2_mad-3_stitch_all-ref",
        )
        self.assertEqual(
            statistics._build_filter_profile(**kwargs, error_dimension="3d"),
            "pdop-2_mad-3_stitch_all-ref",
        )

    def test_abbreviated_options_are_rejected(self) -> None:
        parser = statistics._build_argument_parser()
        arguments = [*default_args("extract"), "--config-path", "x"]
        self.assertEqual(parser.parse_args([*arguments, "--pdop-max", "5"]).pdop_max, 5.0)
        with self.assertRaises(SystemExit):
            parser.parse_args([*arguments, "--pdop", "5"])


class CoverageTimeAxisTest(unittest.TestCase):
    def test_coverage_spans_zero_to_end_time_with_empty_windows(self) -> None:
        events = pd.DataFrame({
            "event_time": [0.05, 0.06, 0.31],
            "NodeId": [1, 2, 1],
        })

        coverage = statistics._make_coverage_df(
            events, {}, coverage_window_size=0.1, total_nodes=4, end_time=0.5
        )

        self.assertEqual(len(coverage), 6)
        self.assertAlmostEqual(coverage["time_bin_start"].iloc[0], 0.0)
        self.assertEqual(
            coverage["localized_nodes"].tolist(), [2, 0, 0, 1, 0, 0]
        )
        self.assertEqual(
            coverage["coverage"].tolist(), [0.5, 0, 0, 0.25, 0, 0]
        )



class LocalizationMethodStatisticsTest(unittest.TestCase):
    def test_statistics_and_interval_are_computed_per_method(self) -> None:
        events = pd.DataFrame({
            "NodeId": [5, 5, 5, 5],
            "event_time": [0.0, 1.0, 1.5, 3.0],
            "localization_method": ["v_twr", "self_loc", "self_loc", "v_twr"],
            "est_position_error": [1.0, 0.2, 0.4, 3.0],
        })
        statistics_df = statistics._make_localization_method_statistics_df(
            events, {"Config": "test"}, ["self_loc", "v_twr"]
        ).set_index("metric")

        self.assertAlmostEqual(statistics_df.loc["mean_error_v_twr", "value"], 2.0)
        self.assertAlmostEqual(statistics_df.loc["mean_error_self_loc", "value"], 0.3)
        # Interval between consecutive events of the same node and method only.
        self.assertAlmostEqual(statistics_df.loc["mean_localization_interval_v_twr", "value"], 3.0)
        self.assertAlmostEqual(statistics_df.loc["mean_localization_interval_self_loc", "value"], 0.5)
        self.assertEqual(statistics_df.loc["mean_error_v_twr", "unit"], "m")
        self.assertEqual(statistics_df.loc["mean_localization_interval_v_twr", "unit"], "s")


if __name__ == "__main__":
    unittest.main()
