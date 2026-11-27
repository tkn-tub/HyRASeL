"""Behavior of the localization methods on small hand-made runs with exact, noise-free timestamps."""

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

# The modules of the evaluation chain live in src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import localization
from solvers import SPEED_OF_LIGHT

# Planar anchors (2D TDoA) and one mobile node in their plane.
ANCHORS = {1: (0.0, 0.0, 0.0), 2: (100.0, 0.0, 0.0), 3: (0.0, 100.0, 0.0), 4: (100.0, 100.0, 0.0)}
NODE, NODE_POSITION = 5, (30.0, 40.0, 0.0)
POSITIONS = {**ANCHORS, NODE: NODE_POSITION}
BROADCAST = 255
SETTINGS = {
    "timestamp_format": "noisy_timestamp", "use_t_twr_pos": True, "copy_t_twr_pos_to_poll": False,
    "tdoa_dimension": "auto", "tdoa_loss": "huber", "tdoa_f_scale": None,
}


def run_dataframe(messages: list[tuple], missing: set[tuple] = frozenset()) -> pd.DataFrame:
    """Run dataframe in which every node logs every message (src, dest, seq, msg_type, send time).

    A reception happens after the time of flight; (node, message index) in
    missing are not logged.
    """
    rows = []
    for node, position in POSITIONS.items():
        for index, (src, dest, seq, msg_type, sent) in enumerate(messages):
            if (node, index) in missing:
                continue
            t = sent + math.dist(position, POSITIONS[src]) / SPEED_OF_LIGHT
            rows.append({
                "NodeId": node, "srcId": src, "destId": dest, "seqNo": seq, "interfaceIdx": 0, "msgType": msg_type,
                "X": position[0], "Y": position[1], "Z": position[2], "src_is_anchor": src in ANCHORS,
                "NodeType": "anchor" if node in ANCHORS else "node", "noisy_timestamp": t,
            })
    return pd.DataFrame(rows)


def twr_messages() -> list[tuple]:
    """One TWR exchange of the node with all anchors (sequence number 1)."""
    messages = [(NODE, BROADCAST, 1, localization.POLL, 0.0)]
    for anchor in ANCHORS:
        messages.append((anchor, NODE, 1, localization.RESP, 1e-3 + anchor * 1e-4))
    messages.append((NODE, BROADCAST, 1, localization.FINAL, 2e-3))
    return messages


def tdoa_messages(rounds: int) -> list[tuple]:
    """Self-localization messages (type 3) of all anchors in turn, one round per sequence number."""
    messages = []
    for seq in range(1, rounds + 1):
        for anchor in ANCHORS:
            messages.append((anchor, BROADCAST, seq, 3, seq * 10e-3 + anchor * 1e-3))
    return messages


def t_twr_messages(node_rounds: range) -> list[tuple]:
    """Four rounds of tdoa_messages; in node_rounds, the node sends after the anchors."""
    messages = tdoa_messages(rounds=4)
    for seq in node_rounds:
        messages.append((NODE, BROADCAST, seq, 3, seq * 10e-3 + 5e-3))
    return sorted(messages, key=lambda message: message[4])


class LocalizationBehaviorTest(unittest.TestCase):
    def test_twr_localizes_the_final_of_a_complete_exchange(self) -> None:
        run = localization.RunRecords(run_dataframe(twr_messages()), "noisy_timestamp")

        results = localization.localize_twr(run, SETTINGS)

        (result,) = results.values()
        self.assertEqual(result["localization_method"], "twr")
        self.assertEqual(result["used_anchors_or_pairs"], [1, 2, 3, 4])
        self.assertLess(result["est_position_error"], 1e-3)

    def test_twr_ignores_resps_addressed_to_other_nodes(self) -> None:
        # Anchor 1 also answers the POLL of another node with the same sequence number.
        messages = [*twr_messages(), (1, NODE + 1, 1, localization.RESP, 1.5e-3)]
        run = localization.RunRecords(run_dataframe(messages), "noisy_timestamp")

        (result,) = localization.localize_twr(run, SETTINGS).values()
        self.assertEqual(result["used_anchors_or_pairs"], [1, 2, 3, 4])
        self.assertLess(result["est_position_error"], 1e-3)

    def test_twr_needs_four_anchors_that_received_the_final(self) -> None:
        final = len(twr_messages()) - 1
        run = localization.RunRecords(run_dataframe(twr_messages(), missing={(4, final)}), "noisy_timestamp")

        self.assertEqual(localization.localize_twr(run, SETTINGS), {})

    def test_t_twr_leaves_the_first_transmission_without_position(self) -> None:
        run = localization.RunRecords(run_dataframe(t_twr_messages(node_rounds=range(2, 5))), "noisy_timestamp")
        first, second, third = run.transmissions(NODE)

        results = localization.localize_t_twr(run, SETTINGS)

        self.assertEqual(set(results), {second.event_idx, third.event_idx})
        self.assertLess(results[second.event_idx]["est_position_error"], 1e-3)
        self.assertIsNone(first.t_twr_pos)

    def test_t_twr_copies_the_position_of_a_final_to_its_poll_without_own_position(self) -> None:
        run = localization.RunRecords(run_dataframe(t_twr_messages(node_rounds=range(2, 5))), "noisy_timestamp")
        first, second, third = run.transmissions(NODE)

        results = localization.localize_t_twr(run, {**SETTINGS, "copy_t_twr_pos_to_poll": True})

        # The first transmission gets the position of the second, but no result.
        self.assertEqual(set(results), {second.event_idx, third.event_idx})
        self.assertEqual(first.t_twr_pos, second.t_twr_pos)
        # The second keeps its own position.
        own = results[second.event_idx]
        self.assertEqual(second.t_twr_pos, [own["est_position_x"], own["est_position_y"], own["est_position_z"]])

    def test_tdoa_localizes_from_the_hyperbolae_of_one_round(self) -> None:
        run = localization.RunRecords(run_dataframe(tdoa_messages(rounds=2)), "noisy_timestamp")

        results = localization.localize_tdoa(run, SETTINGS)

        self.assertTrue(results)
        for result in results.values():
            self.assertFalse(result["using_anchor_stitching"])
            self.assertEqual(result["tdoa_dimension"], 2)
            self.assertLess(result["est_position_error"], 1e-3)

    def test_stitching_completes_the_hyperbolae_of_the_current_round(self) -> None:
        messages = tdoa_messages(rounds=3)
        # The node misses anchor 4's message of round 2.
        for index, (src, _dest, seq, _msg_type, _sent) in enumerate(messages):
            if src == 4 and seq == 2:
                missed = index
        run = localization.RunRecords(run_dataframe(messages, missing={(NODE, missed)}), "noisy_timestamp")
        (current,) = [r for r in run.receptions(NODE) if r.src == 1 and r.seq == 3]

        result = localization.localize_tdoa(run, SETTINGS)[current.event_idx]

        # The round of anchor 1 has the hyperbolae with anchors 2 and 3 only; stitching keeps them.
        self.assertTrue(result["using_anchor_stitching"])
        self.assertIn((1, 2), result["used_anchors_or_pairs"])
        self.assertIn((1, 3), result["used_anchors_or_pairs"])
        self.assertLess(result["est_position_error"], 1e-3)

    def test_tdoa_does_not_need_the_reference_anchor_to_receive_the_mid_messages(self) -> None:
        messages = tdoa_messages(rounds=2)
        # Anchor 1 misses all messages of the other anchors; they still receive anchor 1.
        missing = {(1, index) for index, (src, *_rest) in enumerate(messages) if src != 1}
        run = localization.RunRecords(run_dataframe(messages, missing), "noisy_timestamp")
        (current,) = [r for r in run.receptions(NODE) if r.src == 1 and r.seq == 2]

        result = localization.localize_tdoa(run, SETTINGS)[current.event_idx]

        self.assertFalse(result["using_anchor_stitching"])
        self.assertEqual(result["used_anchors_or_pairs"], [(1, 2), (1, 3), (1, 4)])
        self.assertLess(result["est_position_error"], 1e-3)

    def test_tdoa_without_a_previous_round_localizes_nothing(self) -> None:
        run = localization.RunRecords(run_dataframe(tdoa_messages(rounds=1)), "noisy_timestamp")

        self.assertEqual(localization.localize_tdoa(run, SETTINGS), {})

    def test_a_message_logged_twice_by_one_node_is_an_error(self) -> None:
        run_df = run_dataframe(tdoa_messages(rounds=1))
        run_df = pd.concat([run_df.iloc[:1], run_df], ignore_index=True)

        with self.assertRaises(ValueError):
            localization.RunRecords(run_df, "noisy_timestamp")


if __name__ == "__main__":
    unittest.main()
