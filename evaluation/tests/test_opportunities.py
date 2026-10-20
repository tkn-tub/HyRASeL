import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import sys

# The modules of the evaluation chain live in src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import opportunities

NUM_ANCHORS = 4
SLOT = 0.002
SEQ_PERIOD = 7 * SLOT


def _raw_df(heard: dict, tx: dict, num_seqs: int = 4) -> pd.DataFrame:
    """Synthetic run with 4 anchors (ids 1-4) and nodes 5-7.

    heard: {seqNo: {node_id: number of anchors heard}}
    tx:    {seqNo: [(node_id, slot[, logged seqNo])]}, slot 4 = first v-anchor
           slot; a different logged seqNo models a stale transmission
    """
    rows = []

    def add(node_id, node_type, src_id, seq_no, slot, logged_seq_no=None):
        rows.append({
            "sim_timestamp": (seq_no - 1) * SEQ_PERIOD + slot * SLOT + 1e-6 * node_id,
            "seqNo": seq_no if logged_seq_no is None else logged_seq_no,
            "srcId": src_id,
            "NodeId": node_id,
            "NodeType": node_type,
        })

    for seq_no in range(1, num_seqs + 1):
        for anchor_id in range(1, NUM_ANCHORS + 1):
            add(anchor_id, "anchor", anchor_id, seq_no, anchor_id - 1)
        for node_id in (5, 6, 7):
            for anchor_id in range(1, heard.get(seq_no, {}).get(node_id, 0) + 1):
                add(node_id, "node", anchor_id, seq_no, anchor_id - 1)
        for node_id, slot, *logged_seq_no in tx.get(seq_no, []):
            add(node_id, "node", node_id, seq_no, slot, *logged_seq_no)
    return pd.DataFrame(rows)


class OpportunityOutcomeTest(unittest.TestCase):
    def test_centralized_outcomes(self) -> None:
        raw_df = _raw_df(
            heard={1: {5: 4, 6: 4}, 2: {5: 4, 6: 3}, 3: {5: 3}},
            # seq 1: selected node 5 answers; seq 2: only its follow-up.
            tx={1: [(5, 4)], 2: [(5, 5)]},
        )
        opportunity_df = opportunities.make_opportunity_df(raw_df, "CENTRALIZED", num_nodes=3)

        # The last sequence is dropped.
        self.assertEqual(opportunity_df["seqNo"].tolist(), [1, 2, 3])
        self.assertEqual(
            opportunity_df["outcome"].tolist(),
            ["used", "missed_no_sender", "no_eligible_node"],
        )
        self.assertEqual(opportunity_df["num_eligible_nodes"].tolist(), [2, 1, 0])

        metrics = opportunities.opportunity_metrics(opportunity_df)
        self.assertAlmostEqual(metrics["opportunity_used"], 1 / 3)
        self.assertAlmostEqual(metrics["opportunity_efficiency"], 1 / 2)
        self.assertAlmostEqual(metrics["opportunity_missed"], 1 / 3)
        self.assertAlmostEqual(metrics["mean_eligible_fraction"], 1 / 3)
        self.assertAlmostEqual(
            sum(metrics[f"opportunity_{outcome}"] for outcome in opportunities.OUTCOMES),
            1.0,
        )

    def test_distributed_success_needs_follow_up(self) -> None:
        raw_df = _raw_df(
            heard={1: {5: 4, 6: 4}, 2: {6: 4}, 3: {}},
            # seq 1: collision, only node 5 is confirmed (follow-up in seq 2);
            # seq 2: node 6 contends without follow-up.
            tx={1: [(5, 4), (6, 4)], 2: [(5, 5), (6, 4)]},
        )
        opportunity_df = opportunities.make_opportunity_df(raw_df, "DISTRIBUTED")

        self.assertEqual(
            opportunity_df["outcome"].tolist(),
            ["used", "missed_failed", "no_eligible_node"],
        )
        self.assertEqual(opportunity_df["num_first_slot_senders"].tolist(), [2, 1, 0])

    def test_stale_transmissions_belong_to_their_send_time_but_are_not_used(self) -> None:
        raw_df = _raw_df(
            heard={1: {5: 4, 6: 4}, 2: {6: 4}, 3: {6: 4}},
            # seq 2: node 5 heard no anchor and contends with the old seqNo 1,
            # followed by its follow-up in seq 3 (seqNo 2).
            tx={1: [(6, 4)], 2: [(6, 5), (5, 4, 1)], 3: [(5, 5, 2)]},
        )
        opportunity_df = opportunities.make_opportunity_df(raw_df, "DISTRIBUTED")

        self.assertEqual(
            opportunity_df["outcome"].tolist(),
            ["used", "missed_stale", "missed_no_sender"],
        )
        self.assertEqual(opportunity_df["num_first_slot_senders"].tolist(), [1, 1, 0])
        self.assertEqual(opportunity_df["num_stale_first_slot_senders"].tolist(), [0, 1, 0])

    def test_other_methods_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            opportunities.make_opportunity_df(_raw_df({}, {}), "TWR")


class RawTraceLookupTest(unittest.TestCase):
    def test_finds_escaped_raw_directory_of_one_rep(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            experiment_path = Path(temporary_directory)
            raw_dir = experiment_path / "speed=2,Method=CENTRALIZED-"
            raw_dir.mkdir()
            for name in ("anchor[0]_rep1_log.csv", "node[0]_rep1_log.csv", "node[0]_rep10_log.csv"):
                (raw_dir / name).write_text("")

            traces = opportunities.find_raw_traces(
                experiment_path, "speed=2,Method=CENTRALIZED", 1
            )
            self.assertEqual(
                [trace.name for trace in traces],
                ["anchor[0]_rep1_log.csv", "node[0]_rep1_log.csv"],
            )
            self.assertEqual(
                opportunities.find_raw_traces(experiment_path, "speed=5,Method=CENTRALIZED", 1),
                [],
            )


if __name__ == "__main__":
    unittest.main()
