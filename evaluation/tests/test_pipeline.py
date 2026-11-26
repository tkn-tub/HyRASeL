import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pipeline
from config import ROOT, load_config
from runs import EVENTS_FILE_NAME, POSTPROCESSED_DIR_NAME, RUN_METADATA_FILE_NAME, parameter_key


class PipelineTest(unittest.TestCase):
    def test_one_job_per_method_and_sweep_value(self) -> None:
        config = {
            "name": "loss-all",
            "data": "data/x/PacketLoss-All",
            "methods": ["TWR", "CENTRALIZED"],
            "reps": "0-10",
            "sweep": {"loss": [0, 0.05]},
            "postprocess": {"use_t_twr_pos": True},
        }

        jobs, missing_raw = pipeline.postprocess_jobs(config, extra_args=[])

        self.assertEqual(missing_raw, [])
        self.assertEqual(
            [job.name for job in jobs],
            ["TWR_loss_0", "TWR_loss_0_05", "CENTRALIZED_loss_0", "CENTRALIZED_loss_0_05"],
        )
        self.assertIn("loss:0.05", jobs[1].command)
        self.assertEqual(jobs[1].command[-1], "--use_t_twr_pos")
        postprocessed = ROOT / "results" / "loss-all" / "postprocessed"
        results_dir_index = jobs[1].command.index("--results_dir")
        self.assertEqual(jobs[1].command[results_dir_index + 1], str(postprocessed))
        self.assertEqual(
            jobs[1].run_dirs[-1], postprocessed / "loss=0.05,Method=TWR" / "rep10"
        )

    def test_split_reps_creates_one_job_per_rep(self) -> None:
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
            config, extra_args=[], raw_reps={parameter_key({"speed": 2, "Method": "TWR"}): {1}}
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
                pipeline.raw_reps_by_parameters(Path(directory)),
                {parameter_key({"Method": "HYPERBOLIC_SELF_LOCALIZATION"}): {0, 3}},
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
        config = load_config(ROOT / "configs" / "Speed.yml")
        results = ROOT / "results" / "Speed"
        self.assertEqual(pipeline.results_dir(config), results)
        self.assertEqual(pipeline.log_dir(config, "20261004_1000"), results / "logs" / "20261004_1000")

    def test_evaluation_dir_follows_the_estimator_of_the_runs(self) -> None:
        config = load_config(ROOT / "configs" / "Speed.yml")
        with TemporaryDirectory() as directory, patch.object(pipeline, "RESULTS_ROOT", Path(directory)):
            results = Path(directory) / "Speed"
            # Without post-processed runs: the estimator of the config (huber).
            self.assertEqual(pipeline.evaluation_dir(config, []), results / "huber" / "hdop-2_mad-3_stitch")

            run_dir = results / POSTPROCESSED_DIR_NAME / "speed=5,Method=TWR" / "rep0"
            run_dir.mkdir(parents=True)
            (run_dir / EVENTS_FILE_NAME).touch()
            metadata = {"processing_settings": {"tdoa_loss": "linear"}, "simulation": {"num_nodes": 4, "end_time": 5.0}}
            (run_dir / RUN_METADATA_FILE_NAME).write_text(json.dumps(metadata))
            self.assertEqual(
                pipeline.evaluation_dir(config, ["--localization-methods", "self_loc"]),
                results / "linear" / "hdop-2_mad-3_stitch_loc-self_loc",
            )


if __name__ == "__main__":
    unittest.main()
