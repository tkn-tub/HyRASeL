import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pipeline


class ConfigTest(unittest.TestCase):
    def test_config_is_merged_into_defaults(self) -> None:
        with TemporaryDirectory() as directory:
            defaults = Path(directory) / "defaults.yml"
            defaults.write_text(
                "jobs: 8\npostprocess:\n  tdoa_loss: huber\n  use_vtwr_pos: true\n"
            )
            config_path = Path(directory) / "speed.yml"
            config_path.write_text(
                "data: data/x/Speed\njobs: 4\npostprocess:\n  tdoa_loss: linear\n"
            )

            config = pipeline.load_config(config_path, defaults)

        self.assertEqual(config["name"], "speed")
        self.assertEqual(config["jobs"], 4)
        self.assertEqual(
            config["postprocess"], {"tdoa_loss": "linear", "use_vtwr_pos": True}
        )
        self.assertEqual(config["sweep"], {})

    def test_data_defaults_to_simulation_config_of_same_name(self) -> None:
        with TemporaryDirectory() as directory:
            defaults = Path(directory) / "defaults.yml"
            defaults.write_text("data_root: ../traces\n")
            config_path = Path(directory) / "Speed.yml"
            config_path.write_text("sweep:\n  speed: [5]\n")

            config = pipeline.load_config(config_path, defaults)

        self.assertEqual(config["data"], str(Path("../traces/Speed")))

    def test_options_become_cli_arguments(self) -> None:
        args = pipeline.to_cli_args(
            {"format": "pdf", "plots": ["error", "coverage"], "show": True, "no-save": False, "x": None}
        )
        self.assertEqual(args, ["--format", "pdf", "--plots", "error", "coverage", "--show"])

    def test_dict_options_become_key_value_arguments(self) -> None:
        args = pipeline.to_cli_args(
            {"method-labels": {"CENTRALIZED": "HyRASeL (Centr.)", "DISTRIBUTED": None}, "empty": {}}
        )
        self.assertEqual(args, ["--method-labels", "CENTRALIZED=HyRASeL (Centr.)"])

    def test_one_job_per_method_and_sweep_value(self) -> None:
        config = {
            "name": "loss-all",
            "data": "data/x/PacketLoss-All",
            "methods": ["TWR", "CENTRALIZED"],
            "reps": "0-10",
            "sweep": {"loss": [0, 0.05]},
            "postprocess": {"use_vtwr_pos": True},
        }

        jobs, missing_raw = pipeline.postprocess_jobs(config, extra_args=[])

        self.assertEqual(missing_raw, [])
        self.assertEqual(
            [job.name for job in jobs],
            ["TWR_loss_0", "TWR_loss_0_05", "CENTRALIZED_loss_0", "CENTRALIZED_loss_0_05"],
        )
        self.assertIn("loss:0.05", jobs[1].command)
        self.assertEqual(jobs[1].command[-1], "--use_vtwr_pos")
        postprocessed = pipeline.ROOT / "results" / "loss-all" / "postprocessed"
        results_dir_index = jobs[1].command.index("--results_dir")
        self.assertEqual(jobs[1].command[results_dir_index + 1], str(postprocessed))
        self.assertEqual(
            jobs[1].run_dirs[-1], postprocessed / "loss=0.05,Method=TWR" / "rep10"
        )

    def test_split_reps_creates_one_job_per_rep(self) -> None:
        self.assertEqual(pipeline.expand_reps("0-2,5"), [0, 1, 2, 5])
        self.assertEqual(pipeline.expand_reps(["3", 1]), [1, 3])

        config = {
            "name": "speed",
            "data": "data/x/Speed",
            "methods": ["TWR"],
            "reps": "0-1",
            "split_reps": True,
            "sweep": {"speed": [2]},
            "postprocess": {},
        }
        jobs, _ = pipeline.postprocess_jobs(config, extra_args=[])

        self.assertEqual([job.name for job in jobs], ["TWR_speed_2_rep0", "TWR_speed_2_rep1"])
        rep_index = jobs[1].command.index("--rep_limit")
        self.assertEqual(jobs[1].command[rep_index + 1 : rep_index + 3], ["1", "--sweep_parameters"])

        # Reps without raw traces get no job and are reported.
        jobs, missing_raw = pipeline.postprocess_jobs(
            config, extra_args=[], raw_reps={"speed=2,Method=TWR": {1}}
        )
        self.assertEqual([job.name for job in jobs], ["TWR_speed_2_rep1"])
        self.assertEqual(missing_raw, ["speed=2,Method=TWR/rep0"])

    def test_raw_reps_and_resume(self) -> None:
        with TemporaryDirectory() as directory:
            raw_dir = Path(directory) / "Method=HYPERBOLIC#_SELF#_LOCALIZATION-"
            raw_dir.mkdir()
            for name in ("node[0]_rep0_log.csv", "anchor[1]_rep3_log.csv"):
                (raw_dir / name).touch()
            self.assertEqual(
                pipeline.raw_reps_by_parameter_dir(Path(directory)),
                {"Method=HYPERBOLIC_SELF_LOCALIZATION": {0, 3}},
            )

            run_dir = Path(directory) / "run" / "rep0"
            job = pipeline.Job("TWR_rep0", ["python3", "post.py", "--method", "TWR"], [run_dir])
            self.assertFalse(pipeline.job_is_done(job))
            run_dir.mkdir(parents=True)
            (run_dir / "localization_events.csv").touch()
            (run_dir / "run_metadata.json").write_text('{"command": ["post.py", "--method", "TWR"]}')
            self.assertTrue(pipeline.job_is_done(job))
            # A different command (e.g. other settings) must be run again.
            other = job._replace(command=["python3", "post.py", "--method", "CENTRALIZED"])
            self.assertFalse(pipeline.job_is_done(other))

    def test_results_are_sorted_by_config(self) -> None:
        config = pipeline.load_config(pipeline.ROOT / "configs" / "Speed.yml")
        results = pipeline.ROOT / "results" / "Speed"
        self.assertEqual(pipeline.results_dir(config), results)
        self.assertEqual(
            pipeline.evaluation_dir(config, []),
            results / "huber" / "hdop-2_mad-3_stitch_all-ref",
        )
        config["postprocess"]["tdoa_solver_profile"] = "legacy"
        self.assertEqual(
            pipeline.evaluation_dir(config, ["--localization-methods", "self_loc"]),
            results / "legacy" / "hdop-2_mad-3_stitch_all-ref_loc-self_loc",
        )
        self.assertEqual(pipeline.log_dir(config, "20261004_1000"), results / "logs" / "20261004_1000")


if __name__ == "__main__":
    unittest.main()
