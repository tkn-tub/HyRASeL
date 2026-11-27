import unittest
from pathlib import Path

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
    tx:    {seqNo: [(node_id, slot[, logged seqNo])]}, slot 4 = first t-anchor
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
            heard={1: {5: 4, 6: 4}, 2: {5: 4, 6: 4}, 3: {5: 3}},
            # seq 1: selected node 5 answers; seq 2: only its follow-up, which has a t-TWR position
            # (node 5 is busy, node 6 is eligible).
            tx={1: [(5, 4)], 2: [(5, 5)]},
        )
        opportunity_df = opportunities.make_opportunity_df(raw_df, "CENTRALIZED", {(5, 2)}, num_nodes=3)

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

    def test_a_busy_t_anchor_is_not_eligible(self) -> None:
        # seq 2: node 5 heard all anchors but sends the follow-up of seq 1.
        raw_df = _raw_df(heard={1: {5: 4}, 2: {5: 4}}, tx={1: [(5, 4)], 2: [(5, 5)]})
        opportunity_df = opportunities.make_opportunity_df(raw_df, "CENTRALIZED", {(5, 2)}, num_nodes=3)

        self.assertEqual(opportunity_df["num_eligible_nodes"].tolist(), [1, 0, 0])
        self.assertEqual(opportunity_df["outcome"].tolist(), ["used", "no_eligible_node", "no_eligible_node"])

    def test_a_selected_node_without_t_twr_position_misses_the_opportunity(self) -> None:
        raw_df = _raw_df(heard={1: {5: 4}}, tx={1: [(5, 4)], 2: [(5, 5)]})
        opportunity_df = opportunities.make_opportunity_df(raw_df, "CENTRALIZED", set(), num_nodes=3)

        self.assertEqual(opportunity_df["outcome"].tolist(), ["missed_failed", "no_eligible_node", "no_eligible_node"])

    def test_a_selected_node_with_fewer_than_four_anchors_can_use_the_opportunity(self) -> None:
        raw_df = _raw_df(
            heard={1: {5: 3}, 2: {6: 4}},
            # seq 1: node 5 heard 3 anchors, its follow-up has a t-TWR position (anchors of the next round);
            # seq 2: node 6 heard 4 anchors, its follow-up has none.
            tx={1: [(5, 4)], 2: [(5, 5), (6, 4)], 3: [(6, 5)]},
        )
        opportunity_df = opportunities.make_opportunity_df(raw_df, "CENTRALIZED", {(5, 2)}, num_nodes=3)

        self.assertEqual(opportunity_df["outcome"].tolist(), ["used", "missed_failed", "no_eligible_node"])
        metrics = opportunities.opportunity_metrics(opportunity_df)
        self.assertAlmostEqual(metrics["opportunity_used"], 1 / 3)
        # Only sequences with an eligible node count for the efficiency.
        self.assertAlmostEqual(metrics["opportunity_efficiency"], 0.0)

    def test_distributed_success_needs_follow_up(self) -> None:
        raw_df = _raw_df(
            heard={1: {5: 4, 6: 4}, 2: {6: 4}, 3: {}},
            # seq 1: collision, only node 5 is confirmed (follow-up in seq 2);
            # seq 2: node 6 contends without follow-up.
            tx={1: [(5, 4), (6, 4)], 2: [(5, 5), (6, 4)]},
        )
        opportunity_df = opportunities.make_opportunity_df(raw_df, "DISTRIBUTED", {(5, 2)})

        self.assertEqual(
            opportunity_df["outcome"].tolist(),
            ["used", "missed_failed", "no_eligible_node"],
        )
        self.assertEqual(opportunity_df["num_first_slot_senders"].tolist(), [2, 1, 0])

    def test_stale_transmissions_belong_to_their_send_time_but_are_not_used(self) -> None:
        raw_df = _raw_df(
            heard={1: {5: 4, 6: 4}, 2: {6: 4, 7: 4}, 3: {6: 4}},
            # seq 2: node 5 heard no anchor and contends with the old seqNo 1,
            # followed by its follow-up in seq 3 (seqNo 2); node 6 is busy, node 7 is eligible.
            tx={1: [(6, 4)], 2: [(6, 5), (5, 4, 1)], 3: [(5, 5, 2)]},
        )
        opportunity_df = opportunities.make_opportunity_df(raw_df, "DISTRIBUTED", {(6, 2), (5, 2)})

        self.assertEqual(
            opportunity_df["outcome"].tolist(),
            ["used", "missed_stale", "missed_no_sender"],
        )
        self.assertEqual(opportunity_df["num_first_slot_senders"].tolist(), [1, 1, 0])
        self.assertEqual(opportunity_df["num_stale_first_slot_senders"].tolist(), [0, 1, 0])

    def test_other_methods_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            opportunities.make_opportunity_df(_raw_df({}, {}), "TWR", set())


if __name__ == "__main__":
    unittest.main()
