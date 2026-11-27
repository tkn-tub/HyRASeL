import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

# The modules of the evaluation chain live in src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runs import add_node_columns, find_run_traces, num_node_traces, parse_reps, parse_run_path, run_path

# Two anchors and one node, with their nodeId in the simulation.  Every anchor sends one message
# (srcId = its nodeId), which it logs first and the other two devices 1 and 2 ms later; the node only
# receives (as in the hyperbolic self-localization).
DEVICES = {"anchor[0]": 1, "anchor[1]": 2, "node[0]": 3}
SENDERS = ["anchor[0]", "anchor[1]"]


def _records(without: str | None = None) -> pd.DataFrame:
    """Trace records of all devices except without."""
    rows = []
    for sender in SENDERS:
        sender_id = DEVICES[sender]
        delay = 0.0
        for logger in [sender, *(device for device in DEVICES if device != sender)]:
            if logger != without:
                rows.append({
                    "Node_Name": logger, "srcId": sender_id, "destId": 255, "seqNo": sender_id,
                    "interfaceIdx": 0, "msgType": 0, "sim_timestamp": sender_id + delay,
                })
            delay += 0.001
    return pd.DataFrame(rows)


class NodeColumnsTest(unittest.TestCase):
    def test_node_ids_follow_the_trace_names(self) -> None:
        records = _records()
        add_node_columns(records)
        nodes = records.groupby("Node_Name")[["NodeId", "NodeType"]].first()
        self.assertEqual(nodes["NodeId"].to_dict(), DEVICES)
        self.assertEqual(nodes.loc["node[0]", "NodeType"], "node")
        self.assertTrue(records["src_is_anchor"].all())

    def test_missing_anchor_trace_is_an_error(self) -> None:
        # Without anchor[0], the anchor indices have a gap.
        with self.assertRaisesRegex(ValueError, "anchor traces are missing"):
            add_node_columns(_records(without="anchor[0]"))
        # Without anchor[1], node[0] would get its NodeId 2 and seem to send its message; only the
        # order of the records shows that node[0] received it.
        with self.assertRaisesRegex(ValueError, "not logged first by their sender"):
            add_node_columns(_records(without="anchor[1]"))

    def test_persistent_t_anchors_are_not_supported(self) -> None:
        records = _records()
        records.loc[records["Node_Name"] == "node[0]", "Node_Name"] = "persistentTAnchor[0]"
        with self.assertRaisesRegex(ValueError, "Unsupported node name"):
            add_node_columns(records)


class NamesTest(unittest.TestCase):
    def test_rep_specs(self) -> None:
        self.assertEqual(parse_reps("0-2,5"), [0, 1, 2, 5])
        self.assertEqual(parse_reps(["3", 1, "2-3"]), [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "start > end"):
            parse_reps("5-3")

    def test_run_path_is_parsed_back(self) -> None:
        path = run_path(Path("postprocessed"), {"speed": 5, "loss": 0.1, "Method": "TWR"}, 3)
        self.assertEqual(path, Path("postprocessed/speed=5,loss=0.1,Method=TWR/rep3"))
        self.assertEqual(parse_run_path(path), ({"speed": 5, "loss": 0.1, "Method": "TWR"}, 3))


class RawTraceLookupTest(unittest.TestCase):
    def test_finds_escaped_raw_directory_of_one_rep(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            experiment_path = Path(temporary_directory)
            raw_dir = experiment_path / "speed=2,Method=HYPERBOLIC#_SELF#_LOCALIZATION-"
            raw_dir.mkdir()
            for name in ("anchor[0]_rep1_log.csv", "node[0]_rep1_log.csv", "node[0]_rep10_log.csv"):
                (raw_dir / name).write_text("")

            parameters = {"speed": 2, "Method": "HYPERBOLIC_SELF_LOCALIZATION"}
            traces = find_run_traces(experiment_path, parameters, 1)
            self.assertEqual([trace.name for trace in traces], ["anchor[0]_rep1_log.csv", "node[0]_rep1_log.csv"])
            self.assertEqual(find_run_traces(experiment_path, {**parameters, "speed": 5}, 1), [])

    def test_lookup_ignores_order_and_number_format(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            experiment_path = Path(temporary_directory)
            raw_dir = experiment_path / "loss=0.10,speed=2,Method=TWR-"
            raw_dir.mkdir()
            (raw_dir / "node[0]_rep0_log.csv").write_text("")
            traces = find_run_traces(experiment_path, {"speed": 2.0, "loss": 0.1, "Method": "TWR"}, 0)
            self.assertEqual([trace.name for trace in traces], ["node[0]_rep0_log.csv"])

    def test_nodes_are_counted_by_their_traces(self) -> None:
        paths = [Path("anchor[0]_rep1_log.csv"), Path("node[0]_rep1_log.csv"), Path("node[3]_rep1_log.csv.gz")]
        self.assertEqual(num_node_traces(paths), 2)

    def test_other_files_in_a_raw_directory_are_an_error(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            experiment_path = Path(temporary_directory)
            raw_dir = experiment_path / "Method=TWR-"
            raw_dir.mkdir()
            (raw_dir / "node[0]_rep1.csv").write_text("")
            with self.assertRaisesRegex(ValueError, "Unexpected trace file name"):
                find_run_traces(experiment_path, {"Method": "TWR"}, 1)


if __name__ == "__main__":
    unittest.main()
