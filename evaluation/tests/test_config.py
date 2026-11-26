import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config, to_cli_args


class ConfigTest(unittest.TestCase):
    def test_config_is_merged_into_defaults(self) -> None:
        with TemporaryDirectory() as directory:
            defaults = Path(directory) / "defaults.yml"
            defaults.write_text(
                "jobs: 8\npostprocess:\n  tdoa_loss: huber\n  use_t_twr_pos: true\n"
            )
            config_path = Path(directory) / "speed.yml"
            config_path.write_text(
                "data: data/x/Speed\njobs: 4\npostprocess:\n  tdoa_loss: linear\n"
            )

            config = load_config(config_path, defaults)

        self.assertEqual(config["name"], "speed")
        self.assertEqual(config["jobs"], 4)
        self.assertEqual(
            config["postprocess"], {"tdoa_loss": "linear", "use_t_twr_pos": True}
        )
        self.assertEqual(config["sweep"], {})

    def test_data_defaults_to_simulation_config_of_same_name(self) -> None:
        with TemporaryDirectory() as directory:
            defaults = Path(directory) / "defaults.yml"
            defaults.write_text("data_root: ../traces\n")
            config_path = Path(directory) / "Speed.yml"
            config_path.write_text("sweep:\n  speed: [5]\n")

            config = load_config(config_path, defaults)

        self.assertEqual(config["data"], str(Path("../traces/Speed")))

    def test_options_become_cli_arguments(self) -> None:
        args = to_cli_args(
            {"pdop-max": 2.0, "localization-methods": ["t_twr", "twr"], "stitch": True, "no": False, "x": None}
        )
        self.assertEqual(args, ["--pdop-max", "2.0", "--localization-methods", "t_twr", "twr", "--stitch"])


if __name__ == "__main__":
    unittest.main()
