"""v-anchor opportunities of HyRASeL (Centralized and Distributed).

Reconstructs from the raw traces whether the opportunity for a new v-anchor in
every anchor sequence was used, missed or had no eligible node.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# Methods with a v-anchor selection; all others have no opportunities.
SUPPORTED_METHODS = ("CENTRALIZED", "DISTRIBUTED")

# AvaApp::overheardSufficientAnchors(): seqNoCounter >= 4.
MIN_HEARD_ANCHORS = 4

# Outcomes of a sequence, also the order of the opportunity_<outcome> metrics:
#   used              a new v-anchor transmitted in the first v-anchor slot
#                     (Distributed: and its contention succeeded)
#   missed_no_sender  >= 1 eligible node, but nobody transmitted
#   missed_failed     node(s) transmitted, but no contention succeeded
#   missed_stale      only nodes with a stale seqNo (of an earlier sequence)
#                     transmitted
#   no_eligible_node  no node heard enough anchors
OUTCOMES = ("used", "missed_no_sender", "missed_failed", "missed_stale", "no_eligible_node")

# File names of the raw traces, e.g. node[12]_rep3_log.csv.
TRACE_NAME_PATTERN = re.compile(
    r"^(anchor|persistentVAnchor|node)\[(\d+)\]_rep(\d+)_log\.csv$",
    re.IGNORECASE,
)

# Columns of opportunities.csv (one row per sequence):
#   seqNo                         anchor sequence number
#   seq_start_time                [s] start of the sequence (slot 0)
#   num_nodes                     simulated nodes (denominator of the eligible fraction)
#   num_eligible_nodes            nodes that heard >= MIN_HEARD_ANCHORS anchors of the sequence
#   num_first_slot_senders        nodes that sent in the first v-anchor slot (incl. stale)
#   num_stale_first_slot_senders  of these, nodes that sent with a stale seqNo
#   used                          True if the opportunity was used
#   outcome                       one of OUTCOMES
OPPORTUNITY_COLUMNS = [
    "seqNo",
    "seq_start_time",
    "num_nodes",
    "num_eligible_nodes",
    "num_first_slot_senders",
    "num_stale_first_slot_senders",
    "used",
    "outcome",
]


# -----------------------------------------------------------------------------
# Loading raw traces
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _raw_dirs_by_parameter_dir(experiment_path: Path) -> dict[str, Path]:
    """Raw directories by post-processed name (without '#' escapes and trailing '-').

    Example: 'Method=HYPERBOLIC#_SELF#_LOCALIZATION-' -> 'Method=HYPERBOLIC_SELF_LOCALIZATION'.
    Cached, because the extraction asks once per run for the same experiment.
    """
    return {
        raw_dir.name.replace("#", "").rstrip("-"): raw_dir
        for raw_dir in experiment_path.iterdir()
        if raw_dir.is_dir()
    }


def find_raw_traces(experiment_path: Path, parameter_dir_name: str, rep: int) -> list[Path]:
    """Raw OMNeT++ traces of one run (all anchors and nodes of one repetition).

    parameter_dir_name is the name of the post-processed run directory, e.g.
    'interAnchorDist=200,Method=CENTRALIZED'.  Returns [] if the experiment,
    the parameter directory or the repetition does not exist.
    """
    if not experiment_path.is_dir():
        return []
    raw_dir = _raw_dirs_by_parameter_dir(experiment_path).get(parameter_dir_name)
    if raw_dir is None:
        return []
    # The pattern check keeps rep1 from matching rep10, rep11, ...
    return sorted(
        path
        for path in raw_dir.glob(f"*_rep{rep}_log.csv")
        if TRACE_NAME_PATTERN.match(path.name)
    )


def load_raw_run(trace_paths: list[Path]) -> pd.DataFrame:
    """One dataframe with all records of one run.

    Adds NodeId and NodeType ('anchor', 'persistentvanchor' or 'node') of the
    device that wrote the record.  NodeIds follow
    post_processing.set_node_id_in_df and the OMNeT++ nodeId parameter:
    anchors 1..A, then persistent v-anchors, then nodes.
    """
    parsed = {}
    for path in trace_paths:
        node_type, index, _ = TRACE_NAME_PATTERN.match(path.name).groups()
        parsed[path] = (node_type.casefold(), int(index))

    counts = {
        node_type: sum(parsed_type == node_type for parsed_type, _ in parsed.values())
        for node_type in ("anchor", "persistentvanchor", "node")
    }
    offsets = {
        "anchor": 0,
        "persistentvanchor": counts["anchor"],
        "node": counts["anchor"] + counts["persistentvanchor"],
    }

    dataframes = []
    for path, (node_type, index) in parsed.items():
        dataframe = pd.read_csv(path, skipinitialspace=True)
        dataframe.columns = dataframe.columns.str.strip()
        dataframe["NodeId"] = offsets[node_type] + index + 1
        dataframe["NodeType"] = node_type
        dataframes.append(dataframe)

    if not dataframes:
        return pd.DataFrame()
    return pd.concat(dataframes, ignore_index=True)


# -----------------------------------------------------------------------------
# Opportunities
# -----------------------------------------------------------------------------

def make_opportunity_df(
    raw_df: pd.DataFrame,
    method: str,
    num_nodes: int | None = None,
) -> pd.DataFrame:
    """One row per anchor sequence with the number of eligible nodes and the outcome.

    Every anchor sequence offers one opportunity for a new v-anchor: the first
    v-anchor slot after the anchor slots.  A node is eligible if it heard at
    least MIN_HEARD_ANCHORS anchors of the sequence.  Centralized: the node
    selected by the anchors may transmit in this slot.  Distributed: every
    eligible node contends; a contention succeeded if the node sends the
    follow-up in the next sequence.

    raw_df:    all records of one run (load_raw_run).
    method:    'CENTRALIZED' or 'DISTRIBUTED'; decides when an opportunity
               counts as used.
    num_nodes: number of simulated nodes.  Defaults to the nodes that appear
               in raw_df; pass the number of node traces so that nodes without
               any record count as well.

    The last sequence is dropped: the simulation may end before its v-anchor
    slots, and a Distributed contention can only be judged in the next one.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Opportunities are only defined for {SUPPORTED_METHODS}, not {method}")

    anchor_ids = np.sort(raw_df.loc[raw_df["NodeType"] == "anchor", "NodeId"].unique())
    num_anchors = len(anchor_ids)
    if num_nodes is None:
        num_nodes = raw_df.loc[raw_df["NodeType"] == "node", "NodeId"].nunique()

    # --- 1. Own transmissions ------------------------------------------------
    # Every device logs its own transmissions (srcId == own id); all other
    # records are receptions.
    tx_df = raw_df.loc[raw_df["srcId"] == raw_df["NodeId"]]
    anchor_tx_df = tx_df.loc[tx_df["NodeType"] == "anchor"]
    node_tx_df = tx_df.loc[tx_df["NodeType"] == "node"]

    # One opportunity per anchor sequence (without the last one, see above).
    seq_nos = np.sort(anchor_tx_df["seqNo"].unique())[:-1]
    if len(seq_nos) == 0:
        return pd.DataFrame(columns=OPPORTUNITY_COLUMNS)

    # --- 2. Time grid of the slots -------------------------------------------
    # Anchor k (1-based) sends in slot k - 1.  anchor_times has one row per
    # sequence and one column per anchor slot.  The slot duration is the mean
    # spacing between the first and the last anchor (median over sequences);
    # the start of a sequence is the median of "anchor send time - its slot
    # offset", which also works if single anchor transmissions are missing.
    anchor_slot = anchor_tx_df["NodeId"].map(
        {anchor_id: slot for slot, anchor_id in enumerate(anchor_ids)}
    )
    anchor_times = anchor_tx_df.assign(slot=anchor_slot).pivot_table(
        index="seqNo", columns="slot", values="sim_timestamp", aggfunc="min"
    )
    slot_duration = float(
        ((anchor_times[num_anchors - 1] - anchor_times[0]) / (num_anchors - 1)).median()
    )
    seq_start_time = (
        anchor_times - np.arange(num_anchors) * slot_duration
    ).median(axis=1)

    # --- 3. Sequence and slot of every node transmission ---------------------
    # Assign every node transmission to the sequence in whose time span it was
    # sent, not to its seqNo: a node that heard no anchor of a sequence still
    # sends with the seqNo of the previous one ("stale").  Subtracting half a
    # slot before the lookup keeps transmissions that are sent slightly
    # before their nominal slot time (Centralized: -0.1 slots) in the right
    # sequence.
    seq_start_time = seq_start_time.sort_index()
    if not seq_start_time.is_monotonic_increasing:
        raise ValueError("Anchor sequences are not ordered in time")
    sequence_index = np.searchsorted(
        seq_start_time.to_numpy(),
        node_tx_df["sim_timestamp"].to_numpy() - 0.5 * slot_duration,
        side="right",
    ) - 1
    # Transmissions before the first sequence (index -1) cannot be assigned.
    node_tx_df = node_tx_df.loc[sequence_index >= 0].assign(
        sequence=seq_start_time.index.to_numpy()[sequence_index[sequence_index >= 0]]
    )
    node_tx_df = node_tx_df.assign(
        slot=np.rint(
            (node_tx_df["sim_timestamp"] - node_tx_df["sequence"].map(seq_start_time))
            / slot_duration
        ),
        stale=node_tx_df["seqNo"] != node_tx_df["sequence"],
    )

    # --- 4. Was the opportunity used? ----------------------------------------
    # The first v-anchor slot directly follows the anchor slots (slot index
    # num_anchors); the follow-up of the previous sequence is sent one slot
    # later.
    first_slot_tx = node_tx_df.loc[node_tx_df["slot"] == num_anchors]
    fresh_first_slot_tx = first_slot_tx.loc[~first_slot_tx["stale"]]
    follow_up_tx = node_tx_df.loc[node_tx_df["slot"] == num_anchors + 1]

    first_slot_senders = first_slot_tx.groupby("sequence")["NodeId"].nunique()
    stale_first_slot_senders = (
        first_slot_tx.loc[first_slot_tx["stale"]].groupby("sequence")["NodeId"].nunique()
    )
    # Transmissions with a stale seqNo never count as used.
    if method == "CENTRALIZED":
        # Only the selected node may send in the first slot, so any
        # transmission there answers the selection of this sequence.
        used_seq_nos = set(fresh_first_slot_tx["sequence"])
    else:
        # A contention in sequence s succeeded if its node sends the follow-up
        # in sequence s + 1 (only nodes confirmed by the anchors do so).
        follow_ups = set(zip(follow_up_tx["sequence"] - 1, follow_up_tx["NodeId"]))
        used_seq_nos = {
            seq_no
            for seq_no, node_id in zip(
                fresh_first_slot_tx["sequence"], fresh_first_slot_tx["NodeId"]
            )
            if (seq_no, node_id) in follow_ups
        }

    # --- 5. Eligible nodes ---------------------------------------------------
    # Eligible: heard at least MIN_HEARD_ANCHORS different anchors of the
    # sequence (receptions of anchor packets in the node traces; the seqNo of
    # anchor packets is always the current one).
    anchor_rx_df = raw_df.loc[
        (raw_df["NodeType"] == "node")
        & raw_df["srcId"].isin(anchor_ids)
        & (raw_df["srcId"] != raw_df["NodeId"])
    ]
    heard_anchors = anchor_rx_df.groupby(["seqNo", "NodeId"])["srcId"].nunique()
    num_eligible = (
        (heard_anchors >= MIN_HEARD_ANCHORS)
        .groupby(level="seqNo")
        .sum()
    )

    # --- 6. One row per sequence with its outcome ----------------------------
    opportunity_df = pd.DataFrame({"seqNo": seq_nos})
    opportunity_df["seq_start_time"] = opportunity_df["seqNo"].map(seq_start_time)
    opportunity_df["num_nodes"] = num_nodes
    opportunity_df["num_eligible_nodes"] = (
        opportunity_df["seqNo"].map(num_eligible).fillna(0).astype(int)
    )
    opportunity_df["num_first_slot_senders"] = (
        opportunity_df["seqNo"].map(first_slot_senders).fillna(0).astype(int)
    )
    opportunity_df["num_stale_first_slot_senders"] = (
        opportunity_df["seqNo"].map(stale_first_slot_senders).fillna(0).astype(int)
    )
    opportunity_df["used"] = opportunity_df["seqNo"].isin(used_seq_nos)
    num_fresh_senders = (
        opportunity_df["num_first_slot_senders"]
        - opportunity_df["num_stale_first_slot_senders"]
    )
    # np.select takes the first matching condition: this is the precedence of
    # the outcomes (see OUTCOMES).
    opportunity_df["outcome"] = np.select(
        [
            opportunity_df["used"],
            opportunity_df["num_eligible_nodes"] == 0,
            (num_fresh_senders == 0) & (opportunity_df["num_stale_first_slot_senders"] > 0),
            opportunity_df["num_first_slot_senders"] == 0,
        ],
        ["used", "no_eligible_node", "missed_stale", "missed_no_sender"],
        default="missed_failed",
    )
    return opportunity_df[OPPORTUNITY_COLUMNS]


def opportunity_metrics(opportunity_df: pd.DataFrame) -> dict[str, float]:
    """Run-level opportunity metrics; empty for a run without sequences.

        opportunity_<outcome>   fraction of sequences with this outcome (sum = 1)
        opportunity_missed      sum of the missed_* fractions
        opportunity_efficiency  used / sequences with >= 1 eligible node
                                (NaN if there is none)
        mean_eligible_fraction  mean fraction of eligible nodes per sequence
        num_opportunities       number of evaluated sequences
    """
    num_opportunities = len(opportunity_df)
    if num_opportunities == 0:
        return {}

    outcome_fractions = (
        opportunity_df["outcome"]
        .value_counts(normalize=True)
        .reindex(OUTCOMES, fill_value=0.0)
    )
    num_usable = (opportunity_df["num_eligible_nodes"] > 0).sum()
    return {
        **{f"opportunity_{outcome}": float(outcome_fractions[outcome]) for outcome in OUTCOMES},
        "opportunity_missed": float(
            sum(outcome_fractions[outcome] for outcome in OUTCOMES if outcome.startswith("missed_"))
        ),
        "opportunity_efficiency": (
            float(opportunity_df["used"].sum() / num_usable) if num_usable else np.nan
        ),
        "mean_eligible_fraction": float(
            (opportunity_df["num_eligible_nodes"] / opportunity_df["num_nodes"]).mean()
        ),
        "num_opportunities": num_opportunities,
    }


def load_run_opportunities(
    experiment_path: Path,
    parameter_dir_name: str,
    rep: int,
    method: str,
) -> pd.DataFrame | None:
    """Opportunity table of one run, or None if the method has none or traces are missing.

    Entry point for extract_statistics.py.  The number of nodes is taken from
    the number of node traces, so nodes that never received anything still
    count in the eligible fraction.
    """
    if method not in SUPPORTED_METHODS:
        return None
    trace_paths = find_raw_traces(experiment_path, parameter_dir_name, rep)
    if not trace_paths:
        return None
    num_nodes = sum(path.name.startswith("node[") for path in trace_paths)
    return make_opportunity_df(load_raw_run(trace_paths), method, num_nodes)
