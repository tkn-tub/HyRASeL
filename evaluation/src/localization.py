"""Localization methods: TWR, t-TWR and TDoA self-localization on the trace records of one run.

Every trace record is one transmission or reception of a message, logged by
one node.  A message is identified by its header (Record.msg); RunRecords
finds the record of a message at any node.  A localization step returns
{event index: result} with the estimated position of the node that logged the
event; METHODS lists the steps of every method.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

from solvers import (
    SPEED_OF_LIGHT, localization_bounds_by_dimension, solve_tdoa_least_squares, trilaterate, twr_time_of_flight,
)

# msgType of the TWR messages (AvaMessageType of the simulation).
POLL, RESP, FINAL = 0, 1, 2

# A 3D position needs 4 ranges, or 3 range differences between 4 different anchors.
MIN_ANCHORS_FOR_TWR = 4
MIN_ANCHOR_PAIRS_FOR_TDOA = 3
MIN_UNIQUE_ANCHORS_FOR_TDOA = 4

# Trace columns of the Record fields node to src_is_anchor, in field order.
RECORD_COLUMNS = ["NodeId", "srcId", "destId", "seqNo", "interfaceIdx", "msgType", "X", "Y", "Z", "src_is_anchor"]


@dataclass(slots=True)
class Record:
    """One trace record: a transmission (src == node) or reception of a message, logged by node."""

    node: int
    src: int
    dest: int
    seq: int
    interface: int
    msg_type: int
    x: float  # true position of node
    y: float
    z: float
    src_is_anchor: bool
    t: float  # timestamp used for the localization
    event_idx: int  # row in the run dataframe
    t_twr_pos: list | None = None  # t-TWR position of a transmission, set by localize_t_twr
    msg: tuple = field(init=False)  # message header, the same at the sender and all receivers

    def __post_init__(self):
        self.msg = (self.src, self.dest, self.seq, self.interface, self.msg_type)

    @property
    def position(self) -> np.ndarray:
        """True position of the node that logged the record."""
        return np.array([self.x, self.y, self.z], dtype=float)


def _records(run_df: pd.DataFrame, timestamp_format: str) -> list[Record]:
    """The rows of the run dataframe as records; event_idx is the row number."""
    if not run_df.index.equals(pd.RangeIndex(len(run_df))):
        raise ValueError("The run dataframe needs a RangeIndex")
    # Column lists instead of iterrows: much faster for millions of rows.
    columns = [run_df[column].tolist() for column in [*RECORD_COLUMNS, timestamp_format]]
    records = []
    for event_idx, values in enumerate(zip(*columns)):
        records.append(Record(*values, event_idx=event_idx))
    return records


def _solver_bounds(run_df: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """TDoA solver bounds of the run from its physical anchors (median position of each)."""
    anchor_rows = run_df.loc[run_df["NodeType"] == "anchor"]
    # One position per physical anchor from all of its records.
    anchor_positions = anchor_rows.groupby("NodeId")[["X", "Y", "Z"]].median()
    if anchor_positions.empty or not np.all(np.isfinite(anchor_positions.to_numpy())):
        raise ValueError("The run has no physical anchors with finite positions")
    return localization_bounds_by_dimension(anchor_positions.to_numpy(dtype=float))


class RunRecords:
    """Trace records of one run, indexed by node and by message, with its mobile nodes and solver bounds."""

    def __init__(self, run_df: pd.DataFrame, timestamp_format: str):
        self.records = _records(run_df, timestamp_format)
        self.by_message = {}  # (node, message header) -> record
        self.by_node = defaultdict(list)  # node -> its records in trace order
        self.transmissions_by_seq = defaultdict(list)  # (node, seq) -> its transmissions in trace order
        for record in self.records:
            self._add(record)
        self._check_records()
        self.mobile_nodes = run_df.loc[run_df["NodeType"] == "node", "NodeId"].unique().tolist()
        self.bounds = _solver_bounds(run_df)

    def _add(self, record: Record) -> None:
        """Add a record to the indices; a message logged twice by one node is an error."""
        if (record.node, record.msg) in self.by_message:
            raise ValueError(f"Node {record.node} logged message {record.msg} twice")
        self.by_message[record.node, record.msg] = record
        self.by_node[record.node].append(record)
        if record.src == record.node:
            self.transmissions_by_seq[record.node, record.seq].append(record)

    def _check_records(self) -> None:
        """Check that every node's records are one block and warn about receptions without a sender record."""
        # span() relies on every node's records being one block of rows (one trace file).
        for node, records in self.by_node.items():
            if records[-1].event_idx - records[0].event_idx != len(records) - 1:
                raise ValueError(f"The records of node {node} are not contiguous")
        # Sender and receivers log the same header; otherwise the methods miss these receptions.
        receptions_without_sender = 0
        for record in self.records:
            if record.src != record.node and self.record(record.src, record.msg) is None:
                receptions_without_sender += 1
        if receptions_without_sender:
            print(f"WARNING: {receptions_without_sender} receptions without the sender's record of the message")

    def record(self, node: int, message: tuple) -> Record | None:
        """The record of a message at node, or None if node did not log it."""
        return self.by_message.get((node, message))

    def logged_by(self, node: int, records: list[Record]) -> list[Record]:
        """The records at node of the messages of records, in the same order, without those node did not log."""
        found = []
        for record in records:
            record_at_node = self.record(node, record.msg)
            if record_at_node is not None:
                found.append(record_at_node)
        return found

    def span(self, first: Record, last: Record) -> list[Record]:
        """Records from first to last (inclusive), both logged by the same node."""
        # A slice of all records: the records of one node are consecutive rows (_check_records).
        return self.records[first.event_idx : last.event_idx + 1]

    def transmissions(self, node: int) -> list[Record]:
        """Messages sent by node, in trace order."""
        return [record for record in self.by_node[node] if record.src == node]

    def receptions(self, node: int) -> list[Record]:
        """Messages received by node, in trace order."""
        return [record for record in self.by_node[node] if record.src != node]

    def previous_round(self, node: int, seq: int, interface: int) -> list[Record]:
        """Transmissions of node in the round before its message (seq, interface), oldest first.

        These are the transmissions with the same sequence number on a lower
        interface and those with the previous sequence number on this or a
        higher interface.
        """
        # A round has one transmission per interface; the sequence number advances with interface 0.
        candidates = []
        for record in self.transmissions_by_seq.get((node, seq), []):
            if record.interface < interface:
                candidates.append(record)
        for record in self.transmissions_by_seq.get((node, seq - 1), []):
            if record.interface >= interface:
                candidates.append(record)
        return sorted(candidates, key=lambda record: record.event_idx)


def localize(run_df: pd.DataFrame, method: str, settings: dict) -> dict[int, dict]:
    """Results {event index: result} of all localization steps of a method."""
    run = RunRecords(run_df, settings["timestamp_format"])
    results = {}
    # The steps share run: t-TWR stores its positions in the records for TDoA.
    for step in METHODS[method]:
        print(f"{step.__name__}: {len(run.mobile_nodes)} mobile nodes")
        step_results = step(run, settings)
        # Every event gets the result of at most one step.
        if results.keys() & step_results.keys():
            raise ValueError(f"{step.__name__} localized events that were already localized")
        results.update(step_results)
    return results


def _result(
    localization_method: str, true_position: np.ndarray, estimated_position: np.ndarray, used_anchors_or_pairs: list,
) -> dict:
    """Estimated position and its error, as exported."""
    return {
        "localization_method": localization_method,
        "est_position_x": estimated_position[0],
        "est_position_y": estimated_position[1],
        "est_position_z": estimated_position[2],
        "est_position_error": math.dist(true_position, estimated_position),
        "used_anchors_or_pairs": used_anchors_or_pairs,
        "using_anchor_stitching": False,
    }


class Exchange(NamedTuple):
    """Records of one (virtual) TWR exchange between a node and an anchor."""

    poll_tx: Record
    poll_rx: Record
    resp_tx: Record
    resp_rx: Record
    final_tx: Record
    final_rx: Record


def _twr_position(exchanges: list[Exchange]) -> np.ndarray:
    """Trilaterated position from TWR exchanges, with the anchor positions at the reception of the FINAL."""
    times_of_flight = []
    anchor_positions = []
    for exchange in exchanges:
        # Node: round R_a and reply D_a; anchor: round R_b and reply D_b.
        # Every interval is measured on one clock, so clock offsets cancel.
        times_of_flight.append(twr_time_of_flight(
            R_a=exchange.resp_rx.t - exchange.poll_tx.t,
            R_b=exchange.final_rx.t - exchange.resp_tx.t,
            D_a=exchange.final_tx.t - exchange.resp_rx.t,
            D_b=exchange.resp_tx.t - exchange.poll_rx.t,
        ))
        anchor_positions.append(exchange.final_rx.position)
    return trilaterate(anchor_positions, times_of_flight)


def localize_twr(run: RunRecords, settings: dict) -> dict[int, dict]:
    """TWR: the FINAL of a node completes its POLL-RESP-FINAL exchange with the anchors that answered.

    The result belongs to the FINAL transmission.  RESPs carry the sequence
    number of the POLL they answer.
    """
    results = {}
    for node in run.mobile_nodes:
        # POLL, RESPs and FINAL of one exchange share the sequence number.
        records_by_seq = defaultdict(list)
        for record in run.by_node[node]:
            records_by_seq[record.seq].append(record)

        for records in records_by_seq.values():
            # Own transmissions: the POLL and, if the exchange was completed, the FINAL.
            own = [r for r in records if r.src == node]
            finals = [r for r in own if r.msg_type == FINAL]
            if not finals:
                continue
            final_tx = finals[0]

            # Anchors that answered the POLL and received the FINAL, in the order of their trace files.
            # RESPs addressed to other nodes answer their POLLs with the same sequence number.
            answers = []
            for resp_rx in records:
                if resp_rx.msg_type == RESP and resp_rx.src_is_anchor and resp_rx.dest == node:
                    final_rx = run.record(resp_rx.src, final_tx.msg)
                    if final_rx is not None:
                        answers.append((resp_rx, final_rx))
            if len(answers) < MIN_ANCHORS_FOR_TWR:
                continue
            answers.sort(key=lambda answer: answer[1].event_idx)

            # Unpacking raises unless there is exactly one POLL.
            (poll_tx,) = [r for r in own if r.msg_type == POLL]
            exchanges = []
            for resp_rx, final_rx in answers:
                # The anchor's records of the POLL and of its own RESP.
                anchor = resp_rx.src
                poll_rx = run.record(anchor, poll_tx.msg)
                resp_tx = run.record(anchor, resp_rx.msg)
                exchanges.append(Exchange(poll_tx, poll_rx, resp_tx, resp_rx, final_tx, final_rx))
            anchors = [exchange.resp_rx.src for exchange in exchanges]
            results[final_tx.event_idx] = _result("twr", final_tx.position, _twr_position(exchanges), anchors)
    return results


def _t_twr_exchange(
    run: RunRecords, anchor: int, polls: list[Record], heard: list[Record], final_tx: Record,
) -> Exchange | None:
    """Virtual TWR exchange with anchor, or None if the anchor missed a message of it.

    The RESP is the youngest message of the anchor that the node heard, the
    POLL the youngest one that was sent before the RESP was received and that
    the anchor received.
    """
    final_rx = run.record(anchor, final_tx.msg)
    resp_rx = [r for r in heard if r.src == anchor][-1]
    resp_tx = run.record(anchor, resp_rx.msg)
    if final_rx is None or resp_tx is None:
        return None
    # Youngest POLL first; a POLL sent after the RESP arrived cannot be answered by it.
    for poll_tx in reversed(polls):
        if poll_tx.t > resp_rx.t:
            continue
        poll_rx = run.record(anchor, poll_tx.msg)
        if poll_rx is not None:
            return Exchange(poll_tx, poll_rx, resp_tx, resp_rx, final_tx, final_rx)
    return None


def localize_t_twr(run: RunRecords, settings: dict) -> dict[int, dict]:
    """t-TWR: every transmission of a node is the FINAL of a virtual TWR exchange with the anchors it heard.

    The POLLs are the node's transmissions of the previous round.  The
    estimated position is stored in the FINAL's transmission record
    (t_twr_pos), where TDoA uses it as the position of a temporary anchor.
    With copy_t_twr_pos_to_poll, a POLL without an own t-TWR position also
    gets the FINAL's position, so that TDoA can use the POLL as a virtual
    anchor as well; the POLL gets no result of its own.
    """
    results = {}
    for node in run.mobile_nodes:
        for final_tx in run.transmissions(node):
            polls = run.previous_round(node, final_tx.seq, final_tx.interface)
            if not polls:
                continue
            # Receptions of the node since its oldest POLL: the candidates for the RESPs.
            heard = [r for r in run.span(polls[0], final_tx) if r.src != node]

            # Physical anchors only: t-TWR ranges to known positions.
            exchanges = {}
            for anchor in {r.src for r in heard if r.src_is_anchor}:
                exchange = _t_twr_exchange(run, anchor, polls, heard, final_tx)
                if exchange is not None:
                    exchanges[anchor] = exchange
            if len(exchanges) < MIN_ANCHORS_FOR_TWR:
                continue

            # Sorted: the solver input does not depend on the order of the set above.
            anchors = sorted(exchanges)
            estimated_position = _twr_position([exchanges[anchor] for anchor in anchors])
            # Position of the node as temporary anchor for the TDoA of the other nodes.
            final_tx.t_twr_pos = estimated_position.tolist()
            results[final_tx.event_idx] = _result("t_twr", final_tx.position, estimated_position, anchors)
    finals = [run.records[event_idx] for event_idx in results]
    if settings["copy_t_twr_pos_to_poll"]:
        # After all FINALs, so that an own t-TWR position of a POLL is kept.
        _copy_t_twr_pos_to_polls(run, finals)
    return results


def _copy_t_twr_pos_to_polls(run: RunRecords, finals: list[Record]) -> None:
    """Store the t-TWR position of every FINAL also in its POLLs that have no t-TWR position yet."""
    # finals of a node in trace order: a POLL of several FINALs gets the position of the oldest one.
    for final_tx in finals:
        for poll_tx in run.previous_round(final_tx.node, final_tx.seq, final_tx.interface):
            if poll_tx.t_twr_pos is None:
                poll_tx.t_twr_pos = final_tx.t_twr_pos


def _has_position(record: Record) -> bool:
    """Whether the sender of a transmission record is a physical anchor or has a t-TWR position.

    Temporary anchors need a t-TWR position also without use_t_twr_pos.
    """
    return record.src_is_anchor or record.t_twr_pos is not None


def _anchor_position(transmission: Record, use_t_twr_pos: bool) -> tuple | list:
    """Position of the sender of a transmission; with use_t_twr_pos, the t-TWR position of a temporary anchor."""
    if not transmission.src_is_anchor and use_t_twr_pos:
        return transmission.t_twr_pos
    # True position of the sender, logged with its transmission.
    return (float(transmission.x), float(transmission.y), float(transmission.z))


class Hyperbola(NamedTuple):
    """Records of one hyperbola besides the current message; node_* are logged by the localized node."""

    node_prev: Record
    node_mid: Record
    peer_prev: Record
    peer_mid: Record
    peer_current: Record


def _in_range(record: Record | None, event_range: tuple[int, int]) -> bool:
    """Whether record exists and its event index lies in event_range (inclusive)."""
    return record is not None and event_range[0] <= record.event_idx <= event_range[1]


def _select_prev_and_mid(
    run: RunRecords, between: list[Record],
    ref: int, ref_range: tuple[int, int], peer: int, peer_range: tuple[int, int],
) -> tuple[Record, Record] | None:
    """Previous message of ref and mid message of peer for one hyperbola, as records at the node.

    Both must have been logged by the node (between) and by peer within its
    range of event indices, the previous message by ref within its range.  ref
    need not have received the mid message, as the time difference uses only
    the timestamps of node and peer.  The mid message is the youngest one of
    peer, the previous message the youngest one of ref before it.
    """
    prev_candidates = []
    mids = []
    # Messages that node and peer logged, peer within its range.
    for record in between:
        peer_record = run.record(peer, record.msg)
        if not _in_range(peer_record, peer_range):
            continue
        if record.src == ref and _in_range(run.record(ref, record.msg), ref_range):
            prev_candidates.append(record)
        # The peer's own record of its transmission: physical anchor or t-TWR position.
        elif record.src == peer and _has_position(peer_record):
            mids.append(record)
    # The last message of ref is the current one.
    prev_candidates = prev_candidates[:-1]
    if not mids:
        return None

    # The previous message must have reached the node before the mid message.
    prevs = [record for record in prev_candidates if record.event_idx < mids[-1].event_idx]
    if not prevs:
        return None
    return prevs[-1], mids[-1]


def _tdoa_round(
    run: RunRecords, node: int, current: Record,
) -> tuple[Record, dict[int, Hyperbola], list[Record]] | None:
    """Hyperbolae of the round that a reception at node ends.

    The reference anchor (ref) sent the current and a previous message, a peer
    anchor a mid message in between; node and peer must have logged all three
    (_time_difference uses only their timestamps; the time of flight between
    ref and peer follows from their positions).  Returns ref's record of the
    current message, the hyperbolae by peer and the receptions of node from the
    oldest possible previous message to the current one, or None without any
    hyperbola.
    """
    ref = current.src
    # Candidates for the previous message: ref's transmissions of its previous round that node received.
    node_prevs = run.logged_by(node, run.previous_round(ref, current.seq, current.interface))
    if not node_prevs:
        return None
    # Receptions of node from the oldest candidate up to the current message.
    between = [r for r in run.span(node_prevs[0], current) if r.src != node]
    # ref's own transmission records of the current and of the oldest previous message.
    ref_current = run.record(ref, current.msg)
    ref_oldest = run.record(ref, node_prevs[0].msg)
    if ref_current is None or ref_oldest is None or not _has_position(ref_current):
        return None
    # Other senders of messages in between: candidates for the peer.
    mid_senders = {r.src for r in between if r.src != ref}
    if not mid_senders:
        return None

    ref_range = (ref_oldest.event_idx, ref_current.event_idx)
    hyperbolae = {}
    for peer in sorted(mid_senders):
        # The peer must have received the current and a previous message of ref.
        peer_current = run.record(peer, current.msg)
        peer_prevs = run.logged_by(peer, node_prevs)
        if peer_current is None or not peer_prevs:
            continue
        peer_range = (peer_prevs[0].event_idx, peer_current.event_idx)
        selected = _select_prev_and_mid(run, between, ref, ref_range, peer, peer_range)
        if selected is None:
            continue
        node_prev, node_mid = selected
        hyperbolae[peer] = Hyperbola(
            node_prev, node_mid, run.record(peer, node_prev.msg), run.record(peer, node_mid.msg), peer_current
        )
    if not hyperbolae:
        return None
    return ref_current, hyperbolae, between


def _time_difference(
    node_prev: float, node_mid: float, node_current: float,
    peer_prev: float, peer_mid: float, peer_current: float, t_ref_peer: float,
) -> float:
    """Arrival time difference at the node between peer and reference anchor, from the timestamps of a hyperbola.

    t_ref_peer is the time of flight between reference and peer anchor.
    """
    # Peer: round from its mid transmission to the current reception, reply from the previous reception to the mid.
    R_ba = peer_current - peer_mid
    D_ba = peer_mid - peer_prev
    # Node: from the previous to the mid and from the mid to the current message.
    S_ab = node_mid - node_prev
    S_ba = node_current - node_mid
    numerator = (R_ba * S_ab) - (S_ba * D_ba) - (t_ref_peer * R_ba) - (t_ref_peer * S_ab) + (t_ref_peer**2)
    return numerator / (R_ba + D_ba - t_ref_peer)


def _hyperbolae_measurements(
    ref_current: Record, hyperbolae: dict[int, Hyperbola], current: Record, use_t_twr_pos: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Anchor positions (reference first) and time differences of the hyperbolae of one round."""
    ref_position = _anchor_position(ref_current, use_t_twr_pos)
    positions = [ref_position]
    time_differences = []
    for hyperbola in hyperbolae.values():
        # peer_mid is the peer's own transmission and carries its position.
        peer_position = _anchor_position(hyperbola.peer_mid, use_t_twr_pos)
        # The time of flight between ref and peer follows from their positions.
        time_differences.append(_time_difference(
            hyperbola.node_prev.t, hyperbola.node_mid.t, current.t,
            hyperbola.peer_prev.t, hyperbola.peer_mid.t, hyperbola.peer_current.t,
            math.dist(ref_position, peer_position) / SPEED_OF_LIGHT,
        ))
        positions.append(peer_position)
    return np.array(positions), np.array(time_differences)


class Measurement(NamedTuple):
    """TDoA measurement of one anchor pair (reference, peer) in one round."""

    event_idx: int
    time_difference: float
    ref_position: np.ndarray
    peer_position: np.ndarray


def _stitched_tdoa_sets(rounds: list[dict]) -> tuple[list, list] | None:
    """TDoA sets from the youngest measurement of every anchor pair of the given rounds (anchor stitching).

    rounds: {(ref, peer): Measurement} per round, oldest first.  Returns the
    solver input (one set per reference anchor, newest measurement first, with
    the newest reference position) and the used pairs, or None if they do not
    determine a position.
    """
    # Youngest measurement per anchor pair, independent of its direction.
    latest = {}  # sorted pair -> (pair, measurement)
    for measurements in rounds:
        for pair, measurement in measurements.items():
            sorted_pair = tuple(sorted(pair))
            if sorted_pair not in latest or measurement.event_idx > latest[sorted_pair][1].event_idx:
                latest[sorted_pair] = (pair, measurement)
    used_pairs = [pair for pair, _ in latest.values()]
    anchors = set()
    for pair in used_pairs:
        anchors.update(pair)
    if len(used_pairs) < MIN_ANCHOR_PAIRS_FOR_TDOA or len(anchors) < MIN_UNIQUE_ANCHORS_FOR_TDOA:
        return None

    # Measurements of different rounds can have different reference anchors: one TDoA set per reference.
    by_ref = defaultdict(list)
    for pair, measurement in latest.values():
        by_ref[pair[0]].append(measurement)
    tdoa_sets = []
    for measurements in by_ref.values():
        measurements = sorted(measurements, key=lambda measurement: measurement.event_idx, reverse=True)
        # A virtual reference anchor moves: its newest position stands for all measurements of the set.
        positions = [measurements[0].ref_position]
        time_differences = []
        for measurement in measurements:
            positions.append(measurement.peer_position)
            time_differences.append(measurement.time_difference)
        tdoa_sets.append((np.asarray(positions), np.asarray(time_differences)))
    return tdoa_sets, used_pairs


def _tdoa_result(
    true_position: np.ndarray, estimated_position: np.ndarray, diagnostics: dict, used_pairs: list, stitched: bool,
) -> dict:
    """Exported result of a TDoA solve with the solver diagnostics."""
    result = _result("self_loc", true_position, estimated_position, used_pairs)
    result.update(diagnostics)
    result["using_anchor_stitching"] = stitched
    return result


def localize_tdoa(run: RunRecords, settings: dict) -> dict[int, dict]:
    """TDoA self-localization: every reception of an anchor message ends a round of hyperbolae.

    With fewer than MIN_ANCHOR_PAIRS_FOR_TDOA hyperbolae, the round is
    completed with the youngest measurements of the other anchor pairs of the
    earlier rounds since the oldest possible previous message (anchor
    stitching).  The result belongs to the reception.
    """
    solver_options = {
        "localization_bounds_by_dimension": run.bounds,
        "dimension_mode": settings["tdoa_dimension"],
        "loss": settings["tdoa_loss"],
        "huber_f_scale_m": settings["tdoa_f_scale"],
    }
    results = {}
    for node in run.mobile_nodes:
        rounds = {}  # event index -> {(ref, peer): Measurement} of the round, for anchor stitching
        for current in run.receptions(node):
            found = _tdoa_round(run, node, current)
            if found is None:
                continue
            ref_current, hyperbolae, between = found
            positions, time_differences = _hyperbolae_measurements(
                ref_current, hyperbolae, current, settings["use_t_twr_pos"]
            )
            # Measurements of this round by anchor pair (ref, peer), kept for stitching in later rounds.
            pairs = [(current.src, peer) for peer in hyperbolae]
            measurements = {}
            for pair, time_difference, peer_position in zip(pairs, time_differences, positions[1:]):
                measurement = Measurement(current.event_idx, time_difference, positions[0], peer_position)
                measurements[pair] = measurement
            rounds[current.event_idx] = measurements

            # Too few hyperbolae in this round: complete it from the earlier rounds.
            stitched = len(pairs) < MIN_ANCHOR_PAIRS_FOR_TDOA
            if stitched:
                # The rounds since the oldest possible previous message, ending with the current one.
                stitched_rounds = []
                for record in between:
                    if record.event_idx in rounds:
                        stitched_rounds.append(rounds[record.event_idx])
                stitching = _stitched_tdoa_sets(stitched_rounds)
                if stitching is None:
                    continue
                tdoa_sets, used_pairs = stitching
            else:
                # Enough hyperbolae: one TDoA set with ref as reference.
                tdoa_sets, used_pairs = [(positions, time_differences)], pairs
            estimated_position, diagnostics = solve_tdoa_least_squares(tdoa_sets, **solver_options)
            result = _tdoa_result(current.position, estimated_position, diagnostics, used_pairs, stitched)
            results[current.event_idx] = result
    return results


# Localization steps of every method, in order; TDoA uses the t-TWR positions of temporary anchors.
METHODS = {
    "TWR": [localize_twr],
    "HYPERBOLIC_SELF_LOCALIZATION": [localize_tdoa],
    "CENTRALIZED": [localize_t_twr, localize_tdoa],
    "DISTRIBUTED": [localize_t_twr, localize_tdoa],
}
