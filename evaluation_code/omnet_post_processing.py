# %%
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from itertools import product
from pathlib import Path
import math
from datetime import datetime


from scipy.optimize import least_squares


# GLOBAL CONSTANTS
POLL = 0
RESP = 1
FINAL = 2
SELF_LOC = 3
TWR = "TWR"
HYP_SELF_LOC = "HYPERBOLIC_SELF_LOCALIZATION"
CENTRALIZED_SELF_LOC = "CENTRALIZED"
DISTRIBUED_SELF_LOC = "DISTRIBUTED"

SPEED_OF_LIGHT = 299792458.0
MIN_ANCHORS_FOR_TWR = 4
MIN_VALID_ANCHOR_PAIRS = 3
MIN_TIME_DIFFS_FOR_SELF_LOCALIZATION = 3

CLOCK = "clock_timestamp"
SIM = "sim_timestamp"
NOISY = "noisy_timestamp"


def build_transmission_lookup(df: pd.DataFrame) -> dict[tuple[int, int, int, int], dict | None]:
    """ Build an O(1) lookup table for transmissions keyed by receiver, sender, sequence number, and repetition.

    Args:
        df (pd.DataFrame): Input dataframe containing transmission and reception events.

    Returns:
        dict[tuple[int, int, int, int], dict | None]: Mapping from (NodeId, srcId, seqNo, Rep) to row data. Duplicate keys are mapped to None.
    """
    lookup: dict[tuple[int, int, int, int], dict | None] = {}
    duplicate_keys: set[tuple[int, int, int, int]] = set()

    for row in df.to_dict("records"):
        key = (int(row["NodeId"]), int(row["srcId"]), int(row["seqNo"]), int(row["Rep"]))
        if key in lookup:
            duplicate_keys.add(key)
            lookup[key] = None
        elif key not in duplicate_keys:
            lookup[key] = row

    return lookup




def _get_position_from_record(row: dict | pd.Series) -> np.ndarray:
    """ Extract a 3D position vector from a row-like object.

    Args:
        row (dict | pd.Series): Row containing X, Y, and Z coordinate fields.

    Returns:
        np.ndarray: Position vector with x, y, and z coordinates.
    """
    return np.array([row["X"], row["Y"], row["Z"]], dtype=float)

def build_node_reception_lookup(node_rx_df: pd.DataFrame) -> dict[tuple[int, int], pd.Series | None]:
    """ Build a lookup table for received messages of one node keyed by sender and sequence number.

    Args:
        node_rx_df (pd.DataFrame): Reception dataframe for one node and one repetition.

    Returns:
        dict[tuple[int, int], pd.Series | None]: Mapping from (srcId, seqNo) to the matching reception row. Duplicate keys are mapped to None.
    """
    lookup: dict[tuple[int, int], pd.Series | None] = {}
    duplicate_keys: set[tuple[int, int]] = set()

    for _, row in node_rx_df.iterrows():
        key = (int(row["srcId"]), int(row["seqNo"]))
        if key in lookup:
            duplicate_keys.add(key)
            lookup[key] = None
        elif key not in duplicate_keys:
            lookup[key] = row

    return lookup

def solve_tdoa_least_squares(
    tdoa_sets: list[tuple[np.ndarray, np.ndarray]],
    speed_of_signal: float = 299792458.0,
    initial_guess: np.ndarray | None = None,
    sigma_r: float = 1.0,  # optional standard deviation of the range measurement in meters
    compute_dop: bool = True,
    max_nfev: int | None = None
) -> tuple[np.ndarray, dict]:
    """ Estimate a position from one or more TDoA anchor sets using nonlinear least squares.

    Args:
        tdoa_sets (list[tuple[np.ndarray, np.ndarray]]): List of tuples containing anchor positions and time differences.
        speed_of_signal (float): Propagation speed used to convert time differences into range differences.
        initial_guess (np.ndarray | None): Optional initial position estimate for the optimizer.
        sigma_r (float): Standard deviation of the range measurement in meters for DOP estimation.
        compute_dop (bool): Whether to compute PDOP, HDOP, VDOP, covariance, and condition number.
        max_nfev (int | None): Optional maximum number of function evaluations for the optimizer.

    Returns:
        tuple[np.ndarray, dict]: Estimated position and diagnostic information from the optimizer.
    """

    # If no initial estimate is provided, use the centroid of all anchors.
    if initial_guess is None:
        all_anchors = np.vstack([anchors for anchors, _ in tdoa_sets])
        initial_guess = np.mean(all_anchors, axis=0)

    def residuals(pos):
        """ Compute stacked TDoA residuals for the current optimizer position.

        Args:
            pos (np.ndarray): Current position estimate evaluated by the optimizer.

        Returns:
            np.ndarray: Concatenated residual vector over all TDoA anchor sets.
        """
        all_residuals = []

        for anchor_positions, time_differences in tdoa_sets:

            range_differences = time_differences * speed_of_signal

            ref_anchor = anchor_positions[0]
            other_anchors = anchor_positions[1:]

            dist_to_ref = np.linalg.norm(pos - ref_anchor)

            dists_to_others = np.linalg.norm(other_anchors - pos, axis=1)

            calculated_diff = dists_to_others - dist_to_ref

            res = range_differences - calculated_diff

            all_residuals.append(res)

        return np.concatenate(all_residuals)

    result = least_squares(
        residuals,
        initial_guess,
        method="lm",
        max_nfev=max_nfev,
    )
    # result = least_squares(residuals, 
    #                     initial_guess, 
    #                     method="trf",
    #                     loss="huber",
    #                     f_scale=5.0
    #                     )

    info = {
        "success": result.success,
        "cost": float(result.cost),
        "iterations": result.nfev,
        "residuals": result.fun
    }

    if compute_dop:
        # Compute DOP values from the optimizer Jacobian at the estimated position.
        J = result.jac  # shape: (num_residuals, dim_position)
        JTJ = J.T @ J
        Q = np.linalg.pinv(JTJ) * sigma_r**2

        dim = Q.shape[0]
        PDOP = float(np.sqrt(np.trace(Q)))
        if dim == 3:
            HDOP = float(np.sqrt(Q[0, 0] + Q[1, 1]))
            VDOP = float(np.sqrt(Q[2, 2]))
        else:
            HDOP = float(np.sqrt(Q[0, 0] + Q[1, 1]))
            VDOP = None

        cond = float(np.linalg.cond(JTJ))
        info.update({"PDOP": PDOP, "HDOP": HDOP, "VDOP": VDOP, "Q_pos": Q, "cond": cond})
    else:
        info.update({"PDOP": None, "HDOP": None, "VDOP": None, "Q_pos": None, "cond": None})

    return result.x, info


def compute_self_loc_statistics(df: pd.DataFrame, timestamp_format: str = SIM, use_v_twr_pos: bool = False) -> pd.DataFrame:
    """ Compute virtual TWR and self-localization statistics for all mobile nodes.

    Args:
        df (pd.DataFrame): Input dataframe containing all simulation events.
        timestamp_format (str): Timestamp column used for all timing calculations.
        use_v_twr_pos (bool): Whether virtual TWR positions should be used for non-anchor nodes.

    Returns:
        pd.DataFrame: Input dataframe enriched with virtual TWR and self-localization result columns.
    """
    # Initialize columns that are filled during post-processing.

    df["_idx"] = df.index
    tx_lookup = build_transmission_lookup(df)

    df["self_loc_pos_estimate"] = None
    df["self_loc_error"] = None
    df["residuals"] = None
    df["v_twr_pos"] = None
    df["v_twr_error"] = None
    df["using_anchor_stitching"] = False
    df["PDOP"] = None
    df["cond"] = None

    # Extract dataframe metadata used for iteration.
    repetitions     = df["Rep"].unique()
    unique_seq_nos  = df["seqNo"].unique()
    mobile_node_ids    = df.loc[df["Node_Name"].str.contains("node", case=False, na=False), "NodeId"].unique()

    # Repetitions are handled together with node IDs below.


    # Compute virtual TWR statistics before self-localization.
    _estimate_node_positions_virtual_twr(df, tx_lookup, mobile_node_ids, repetitions, timestamp_format)

    # Rebuild lookup after virtual TWR columns have been updated.
    tx_lookup = build_transmission_lookup(df)


    # Compute self-localization statistics.
    # _estimate_node_position_self_loc()
    # Get Self Loc Statistics
    for node_id, rep in product(mobile_node_ids, repetitions):
        # Get all messages received by the selected node.
        node_rx_df = _get_node_reception_data(df, node_id, rep).copy()
        node_rx_df["tdoa_info"] = None
        rx_sender_seq_lookup = build_node_reception_lookup(node_rx_df)
        node_rx_by_seq = {seq: group for seq, group in node_rx_df.groupby("seqNo", sort=False)}

        # Preserve the original script semantics: iterate over the globally observed sequence numbers
        # in their original order, not over the per-node group order. This matters because anchor
        # stitching stores intermediate state in node_rx_df and is therefore order-dependent.
        for seq_no in unique_seq_nos:
            seq_rx_df = node_rx_by_seq.get(seq_no)
            if seq_rx_df is None:
                continue

            # Self-localization.
            # Each row represents one message; process each incoming message in dataframe index order.
            for msg_record_idx in seq_rx_df.index:
                _compute_localization_per_incoming_message(df, tx_lookup, node_id, seq_rx_df, node_rx_df, rx_sender_seq_lookup, rep, seq_no, msg_record_idx, timestamp_format=timestamp_format, use_v_twr_pos=use_v_twr_pos)

            # Anchor stitching is handled inside the per-message localization function.


    return df

def _estimate_node_positions_virtual_twr(df: pd.DataFrame, df_lookup: dict | pd.DataFrame, mobile_node_ids: list, repetitions: list, timestamp_format: str = SIM):
    """ Estimate virtual TWR positions for mobile nodes from overheard anchor responses.

    Args:
        df (pd.DataFrame): Dataframe that is updated in place with virtual TWR results.
        df_lookup (dict): Lookup table for transmission and reception rows.
        mobile_node_ids (list): IDs of mobile nodes to process.
        repetitions (list): Experiment repetitions to process.
        timestamp_format (str): Timestamp column used for timing calculations.

    Returns:
        None: Results are written directly into df.
    """
    # Compute virtual TWR statistics.
    for node_id, rep in product(mobile_node_ids, repetitions):
        # Get all entries where the node transmits.
        node_tx_df = df.query("srcId == @node_id and NodeId == @node_id and Rep == @rep")
        node_rx_df = _get_node_reception_data(df, node_id, rep) # pd.Series
        for msg_record in node_tx_df.iloc[::-1].itertuples(): # Itertuples for speed
            # The loop uses tuples for speed while lookup rows are dictionaries.
            v_final_seq_no = msg_record.seqNo
            v_poll_seq_no  = v_final_seq_no - 1
            # Check whether a previous message exists.
            v_poll_tx = _get_transmission_for_receiver(df_lookup, node_id, node_id, v_poll_seq_no, rep)
            if v_poll_tx is None: 
                continue # No previous message exists.
            
            v_poll_tx_idx = int(v_poll_tx["_idx"]) # Lookup dictionary row index.
            v_final_tx_idx = msg_record.Index # Original dataframe index.
            between_df = node_rx_df.loc[v_poll_tx_idx:v_final_tx_idx]

            # Only anchors are used for localization.
            anchor_ids = between_df.query("src_is_anchor == True")["srcId"].unique()

            confirmed_anchor_ids = []
            for anchor_id in anchor_ids: 
                # Did they overhear the final message
                final_rx = _get_transmission_for_receiver(df_lookup, node_id, anchor_id, v_final_seq_no, rep)
                poll_rx = _get_transmission_for_receiver(df_lookup, node_id, anchor_id, v_poll_seq_no, rep)
                if (final_rx is not None and poll_rx is not None):
                    confirmed_anchor_ids.append(anchor_id)

            if (len(confirmed_anchor_ids) < 4):
                continue

            v_poll_tx_ts = float(v_poll_tx[timestamp_format])

            v_final_tx_ts = getattr(msg_record, timestamp_format)

            anchor_positions = []
            tofs = []
            for anchor_id in confirmed_anchor_ids:
                assert(anchor_id != node_id)
                anchor_seq_no = between_df.query("srcId == @anchor_id")["seqNo"]
                assert(len(anchor_seq_no) == 1)
                anchor_seq_no = anchor_seq_no.iloc[0]

                v_resp_rx_ts = between_df.query("srcId == @anchor_id")[timestamp_format].squeeze()
            
                # v_resp_tx_ts = df.query("srcId == @anchor_id and NodeId == @anchor_id and seqNo == @anchor_seq_no and Rep == @rep")[timestamp_format].squeeze()
                v_resp_tx_ts = _get_transmission_for_receiver(df_lookup, anchor_id, anchor_id, anchor_seq_no, rep)[timestamp_format]

                # v_poll_rx_ts = df.query("srcId == @node_id   and NodeId == @anchor_id and seqNo == @v_poll_seq_no and Rep == @rep")[timestamp_format].squeeze()
                v_poll_rx_ts = _get_transmission_for_receiver(df_lookup, node_id, anchor_id, v_poll_seq_no, rep)[timestamp_format]

                # v_final_rx_ts= df.query("srcId == @node_id   and NodeId == @anchor_id and seqNo == @v_final_seq_no and Rep == @rep")[timestamp_format].squeeze()
                v_final_rx_ts = _get_transmission_for_receiver(df_lookup, node_id, anchor_id, v_final_seq_no, rep)[timestamp_format]

                tof = calc_twr_tof(
                    R_a = (v_resp_rx_ts - v_poll_tx_ts),
                    R_b = (v_final_rx_ts - v_resp_tx_ts), 
                    D_a = (v_final_tx_ts - v_resp_rx_ts),
                    D_b = (v_resp_tx_ts - v_poll_rx_ts)
                )
                tofs.append(tof)

                try: 
                    float(tof)
                except: 
                    raise Exception("Time-of-flight calculation returned a non-numeric value")

                # anchor_pos = np.array(df.query("srcId == @node_id   and NodeId == @anchor_id and seqNo == @v_final_seq_no and Rep == @rep")[["X", "Y", "Z"]].squeeze())
                anchor_pos = _get_position_from_record(_get_transmission_for_receiver(df_lookup, node_id, anchor_id, v_final_seq_no, rep))
                anchor_positions.append(anchor_pos)

            # node_position = np.array(node_tx_df.query("seqNo == @v_final_seq_no and srcId == @node_id and NodeId == @node_id and Rep == @rep")[["X", "Y", "Z"]].squeeze())
            node_position = _get_position_from_record(_get_transmission_for_receiver(df_lookup, node_id, node_id, v_final_seq_no, rep))
            estimated_pos = estimate_twr_position(anchor_positions, tofs)

            pos_error = math.dist(node_position, estimated_pos)

            df.at[v_final_tx_idx, "v_twr_pos"] = estimated_pos.tolist()
            df.at[v_final_tx_idx, "v_twr_error"] = pos_error



def _compute_localization_per_incoming_message(df: pd.DataFrame, df_lookup: dict, node_id: int, seq_rx_df: pd.DataFrame, node_rx_df: pd.DataFrame, rx_sender_seq_lookup: dict, rep: int, seq_no: int, msg_record_idx: int, timestamp_format: str = SIM, use_v_twr_pos: bool = False):
    """ Compute a self-localization estimate for one received message.

    Args:
        df (pd.DataFrame): Global dataframe that is updated with localization results.
        df_lookup (dict): Lookup table for transmission and reception rows.
        node_id (int): ID of the receiving node being localized.
        seq_rx_df (pd.DataFrame): All receptions with the current sequence number for the node.
        node_rx_df (pd.DataFrame): All receptions of the node in the current repetition.
        rx_sender_seq_lookup (dict): Lookup table for node receptions keyed by sender and sequence number.
        rep (int): Experiment repetition.
        seq_no (int): Current sequence number.
        msg_record_idx (int): Index of the current received message.
        timestamp_format (str): Timestamp column used for timing calculations.
        use_v_twr_pos (bool): Whether virtual TWR positions should be used for non-anchor nodes.

    Returns:
        bool | None: True if a direct localization was computed, otherwise None.
    """
    
    # Get the message received by the current node.
    node_current_rx = seq_rx_df.loc[msg_record_idx]
    
    # Get the sender of the current message.
    sender_id = node_current_rx["srcId"]

    # Get the previous message received from the same sender.
    node_prev_rx = rx_sender_seq_lookup.get((int(sender_id), int(seq_no - 1)))

    # Without a previous sender transmission, no localization can be computed.
    if node_prev_rx is None:
        return

    # Get the indices of the current and previous sender transmissions.
    prev_index = node_prev_rx.name
    current_index = node_current_rx.name 
    between_df = node_rx_df.loc[prev_index:current_index]

    # Get all anchor IDs that were heard by both sender and receiver.
    anchor_ids, anchor_seq_nos = _get_valid_anchors_with_seqnos(between_df, df_lookup, sender_id, seq_no, rep)

    # Get the true node position at the message reception time.
    true_pos = (node_current_rx["X"], node_current_rx["Y"], node_current_rx["Z"])
    current_index = node_current_rx.name

    # If the sender is a v-Anchor and does not have a current position from v-twr, we cannot use it further and 
        # thus, we stop and continue with the next incoming message
    sender_current_tx = _get_transmission_for_receiver(df_lookup, sender_id, sender_id, seq_no, rep)
    sender_current_pos = sender_current_tx["v_twr_pos"]
    if (sender_current_pos is None) & (sender_current_tx["node_is_anchor"]==False):
        return

    # Compute all time differences and anchor positions for this round.
    anchor_positions, time_diffs = _compute_delta_ts_for_round(df, df_lookup, anchor_ids, anchor_seq_nos, sender_id, node_id, seq_no, rep, timestamp_format=timestamp_format, use_v_twr_pos=use_v_twr_pos)    
    
    # At least three additional anchors are required because the sender is the reference anchor.
    valid_anchor_pairs = [(int(sender_id), int(anchor_id)) for anchor_id in anchor_ids]

    # Start anchor stitching logic.
    # Store reference-anchor and other-anchor positions for possible anchor stitching.
    tdoa_info = {}
    for anchor_id, time_diff, anchor_pos in zip(anchor_ids, time_diffs, anchor_positions[1:]):
        sender_position = anchor_positions[0]
        tdoa_info[(int(sender_id), int(anchor_id))] = [current_index, time_diff, sender_position, anchor_pos]
            
    node_rx_df.at[current_index, "tdoa_info"] = tdoa_info
    

    # Perform anchor stitching.
    if (len(valid_anchor_pairs) < MIN_VALID_ANCHOR_PAIRS):
        # If this is not the case, try to get the position with anchor pairs of previous rounds
        stitched_anchor_pairs = [] 
        for idx, row in between_df.iterrows():
            if row["tdoa_info"] is not None:
                for key in row["tdoa_info"]:
                    stitched_anchor_pairs.append({key: row["tdoa_info"][key]})
        
        seen_pairs = {}
        for tdoa_info_dict in stitched_anchor_pairs:
            for anchor_pair, tdoa_info in tdoa_info_dict.items():
                # Sort the pair so that (1, 3) and (3, 1) are treated as the same pair.
                sorted_anchor_pair = tuple(sorted(anchor_pair))

                if sorted_anchor_pair not in seen_pairs:
                    # If not seen yet, add it to list
                    seen_pairs[sorted_anchor_pair] = (anchor_pair, tdoa_info)
                else:
                    # Already seen -> keep only if index is bigger
                    existing_index = seen_pairs[sorted_anchor_pair][1][0]
                    current_index = tdoa_info[0]
                    if current_index > existing_index:
                        seen_pairs[sorted_anchor_pair] = (anchor_pair, tdoa_info)

        # Convert the result back into list-of-dicts format.
        result = [{original_pair: info_list} for original_pair, info_list in seen_pairs.values()]
        

        if len(result) >= MIN_TIME_DIFFS_FOR_SELF_LOCALIZATION:            
            # Build lists of anchor positions and time differences.
            ref_anchor_groups = {}

            for pair_dict in result:
                for anchor_pair, pair_info in pair_dict.items():
                    # The first tuple entry is the reference anchor.
                    ref_anchor = anchor_pair[0]  
                    index, time_diff, ref_pos, other_pos = pair_info

                    if ref_anchor not in ref_anchor_groups:
                        ref_anchor_groups[ref_anchor] = []

                    ref_anchor_groups[ref_anchor].append((index, time_diff, ref_pos, other_pos))

            # Sort each reference-anchor group by index so that the newest position is first.
            final_stitched_list = []

            for ref_anchor, entries in ref_anchor_groups.items():
                # Sort by descending index.
                entries_sorted = sorted(entries, key=lambda x: x[0], reverse=True)

                # Use the newest entry as the reference-anchor position.
                ref_pos_latest = entries_sorted[0][2]

                # Create position and time-difference lists.
                positions_list = [ref_pos_latest]  # Reference anchor first.
                time_diffs_list = []

                for entry in entries_sorted:
                    _, td, _, other_pos = entry
                    positions_list.append(other_pos)
                    time_diffs_list.append(td)

                final_stitched_list.append((np.asarray(positions_list), np.asarray(time_diffs_list)))

            estimated_pos, info = solve_tdoa_least_squares(final_stitched_list)
            # Mark that this position was obtained with anchor stitching.
            df.at[current_index, "using_anchor_stitching"] = True
            error = math.dist(true_pos, estimated_pos)    
            df.at[current_index, "self_loc_pos_estimate"] = estimated_pos
            df.at[current_index, "self_loc_error"] = error
            df.at[current_index, "residuals"] = info["residuals"]
            df.at[current_index, "PDOP"] = info["PDOP"]
            df.at[current_index, "cond"] = info["cond"]
        return

    # If fewer than three time differences are available, 3D hyperbolic localization is not possible.
    if (len(time_diffs) < MIN_TIME_DIFFS_FOR_SELF_LOCALIZATION):
        return
    
    # Estimate the position from time differences and anchor positions.
    estimated_pos, info = solve_tdoa_least_squares([(anchor_positions, time_diffs)])

    error = math.dist(true_pos, estimated_pos)    
    df.at[current_index, "self_loc_pos_estimate"] = estimated_pos
    df.at[current_index, "self_loc_error"] = error
    df.at[current_index, "residuals"] = info["residuals"]
    df.at[current_index, "PDOP"] = info["PDOP"]
    df.at[current_index, "cond"] = info["cond"]
    return True



def _get_valid_anchors_with_seqnos(between_df: pd.DataFrame, df_lookup: pd.DataFrame, sender_id: int, seq_no: int, rep: int) -> list:
    """ Identify anchors that provide valid current and previous messages for TDoA estimation.

    Args:
        between_df (pd.DataFrame): Reception rows between the previous and current sender message.
        df_lookup (dict): Lookup table for transmission and reception rows.
        sender_id (int): ID of the reference sender.
        seq_no (int): Current sequence number.
        rep (int): Experiment repetition.

    Returns:
        tuple[list, list]: Valid anchor IDs and their corresponding sequence numbers.
    """
    # Collect all anchors except the sender itself.
    anchor_ids = []
    anchor_seq_nos = []

    # Check for all anchors whose message was received within last ranging round
    for anchor_id in between_df["srcId"].unique():
        # The sender itself may be in the list, but it must not be used as an additional anchor.
        if anchor_id == sender_id: 
            continue

        # Check whether the sender received the anchor messages required for TDoA estimation.
            # This is a prerequisite for TDoA estimation.
        anchor_current_rx_sender    = _get_transmission_for_receiver(df_lookup, sender_id, anchor_id, seq_no, rep)
        anchor_prev_rx_sender       = _get_transmission_for_receiver(df_lookup, sender_id, anchor_id, seq_no-1, rep)
        
        # If either the current or previous message was not received, this anchor is invalid.
        if anchor_current_rx_sender is None or anchor_prev_rx_sender is None: 
            continue
        
        # Get the most recent sequence number of the anchor.
        anchor_seq_no_df = between_df.loc[between_df["srcId"]==anchor_id]
        anchor_seq_no = anchor_seq_no_df.iloc[-1]["seqNo"]

        # If the received transmission is from a virtual anchor, check whether it has a position estimate.
            # If not, the node cannot be used as an anchor.
        if (anchor_current_rx_sender["node_is_anchor"] == False): #
            anchor_current_tx = _get_transmission_for_receiver(df_lookup, anchor_id, anchor_id, anchor_seq_no, rep)
            anchor_position_vtwr = anchor_current_tx["v_twr_pos"]
            if anchor_position_vtwr is None: 
                continue

        # Add the anchor after all validity checks passed.
        anchor_ids.append(anchor_id)
        anchor_seq_nos.append(anchor_seq_no)

    return anchor_ids, anchor_seq_nos


def _compute_delta_ts_for_round(df: pd.DataFrame, df_lookup: pd.DataFrame, anchor_ids: list, anchor_seq_nos: list, sender_id:int, node_id: int, seq_no: int, rep: int, timestamp_format: str = SIM, use_v_twr_pos: bool = False)->tuple[np.array, np.array]:
    """ Compute TDoA time differences and anchor positions for one localization round.

    Args:
        df (pd.DataFrame): Input dataframe, kept for compatibility with existing call sites.
        df_lookup (dict): Lookup table for transmission and reception rows.
        anchor_ids (list): Anchor IDs used in the current round.
        anchor_seq_nos (list): Anchor sequence numbers corresponding to anchor_ids.
        sender_id (int): ID of the reference sender.
        node_id (int): ID of the receiving node being localized.
        seq_no (int): Current sequence number of the reference sender.
        rep (int): Experiment repetition.
        timestamp_format (str): Timestamp column used for timing calculations.
        use_v_twr_pos (bool): Whether virtual TWR positions should be used for non-anchor nodes.

    Returns:
        tuple[np.ndarray, np.ndarray]: Anchor positions and computed time differences.
    """
    node_current_rx_sender      = _get_transmission_for_receiver(df_lookup, sender_id, node_id, seq_no, rep)
    node_prev_rx_sender         = _get_transmission_for_receiver(df_lookup, sender_id, node_id, seq_no-1, rep)
    sender_current_tx           = _get_transmission_for_receiver(df_lookup, sender_id, sender_id, seq_no, rep)
    sender_position = ( float(sender_current_tx["X"]),
                        float(sender_current_tx["Y"]),
                        float(sender_current_tx["Z"]))
    
    if ((sender_current_tx["src_is_anchor"]==False) and use_v_twr_pos):
        sender_position = sender_current_tx["v_twr_pos"]

    positions = [sender_position]
    time_differences = []
    for anchor_id, anchor_seq_no in zip(anchor_ids, anchor_seq_nos):
        anchor_current_rx_sender    = _get_transmission_for_receiver(df_lookup, sender_id, anchor_id, seq_no, rep)
        anchor_prev_rx_sender       = _get_transmission_for_receiver(df_lookup, sender_id, anchor_id, seq_no-1, rep)
        anchor_current_tx           = _get_transmission_for_receiver(df_lookup, anchor_id, anchor_id, anchor_seq_no, rep)
        node_current_rx_anchor      = _get_transmission_for_receiver(df_lookup, anchor_id, node_id, anchor_seq_no, rep)

        anchor_position = (float(anchor_current_rx_sender["X"]),
                           float(anchor_current_rx_sender["Y"]),
                           float(anchor_current_rx_sender["Z"]))
        
        if ((anchor_current_tx["src_is_anchor"] == False) and use_v_twr_pos):
            # Check whether v-Anchor has a position estimate
            anchor_position = anchor_current_tx["v_twr_pos"]

        dist_ab = math.dist(sender_position, anchor_position)
        t_ab = dist_ab / SPEED_OF_LIGHT
        
        delta_t = calc_self_loc_delta_t(
            s_rx1_a = node_prev_rx_sender[timestamp_format], 
            s_rx1_b = node_current_rx_anchor[timestamp_format],
            s_rx2_a = node_current_rx_sender[timestamp_format],
            b_rx1_a = anchor_prev_rx_sender[timestamp_format],
            b_tx1   = anchor_current_tx[timestamp_format],
            b_rx2_a = anchor_current_rx_sender[timestamp_format],
            t_ab    = t_ab
            )
        positions.append(anchor_position)
        time_differences.append(delta_t)

    anchor_positions = np.array(positions)
    time_diffs = np.array(time_differences)

    return anchor_positions, time_diffs 

def calc_self_loc_delta_t(s_rx1_a: float, 
                           s_rx1_b: float, 
                           s_rx2_a: float,
                           b_rx1_a: float, 
                           b_tx1: float, 
                           b_rx2_a: float, 
                           t_ab: float) -> float:
    """ Compute the self-localization time difference from sender, anchor, and receiver timestamps.

    Args:
        s_rx1_a (float): Receiver timestamp for the previous sender message.
        s_rx1_b (float): Receiver timestamp for the anchor message.
        s_rx2_a (float): Receiver timestamp for the current sender message.
        b_rx1_a (float): Anchor timestamp for the previous sender message.
        b_tx1 (float): Anchor transmit timestamp.
        b_rx2_a (float): Anchor timestamp for the current sender message.
        t_ab (float): Propagation time between sender and anchor.

    Returns:
        float: Estimated time difference for hyperbolic localization.
    """
    R_ba = b_rx2_a - b_tx1
    D_ba = b_tx1 - b_rx1_a
    S_ab = s_rx1_b - s_rx1_a
    S_ba = s_rx2_a - s_rx1_b
    numerator = (R_ba * S_ab) - (S_ba * D_ba) - (t_ab * R_ba) - (t_ab * S_ab) + (t_ab**2)
    denominator = R_ba + D_ba - t_ab
    delta_t = numerator / denominator
    return delta_t

def _get_transmission_for_receiver(df_lookup: dict | pd.DataFrame, sender_id: int, receiver_id: int, seq_no: int, rep: int) -> dict | pd.Series | None:
    """ Retrieve one transmission/reception row for a sender-receiver-sequence-repetition tuple.

    Args:
        df_lookup (dict | pd.DataFrame): Precomputed dictionary lookup or indexed dataframe.
        sender_id (int): ID of the transmitting node.
        receiver_id (int): ID of the receiving node.
        seq_no (int): Sequence number of the message.
        rep (int): Experiment repetition.

    Returns:
        dict | pd.Series | None: Matching row or None if no unique row exists.
    """
    key = (int(receiver_id), int(sender_id), int(seq_no), int(rep))

    if isinstance(df_lookup, dict):
        return df_lookup.get(key)

    try:
        row = df_lookup.loc[key]
        if isinstance(row, pd.DataFrame):
            return row.iloc[0] if len(row) == 1 else None
        return row
    except KeyError:
        return None



def _get_node_reception_from_sender(node_rx_df: pd.DataFrame, sender_id: int, seq_no: int) -> pd.Series | None:
    """ Retrieve one reception row for a specific sender and sequence number.

    Args:
        node_rx_df (pd.DataFrame): Reception dataframe for one receiving node.
        sender_id (int): ID of the sender to search for.
        seq_no (int): Sequence number to search for.

    Returns:
        pd.Series | None: Matching reception row or None if no row exists.
    """
    mask = (
        (node_rx_df["srcId"] == sender_id) &
        (node_rx_df["seqNo"] == seq_no)
    )
    result = node_rx_df.loc[mask]
    assert(len(result)<=1)
    return result.iloc[0] if len(result) == 1 else None


def _get_node_reception_data(df: pd.DataFrame, node_id: int, rep: int)->pd.DataFrame:
    """ Get the reception data from all nodes that the queried node receives

    Args:
        df (pd.DataFrame): Input dataframe to be filtered
        node_id (int): Query ID for which the reception data is going to be extracted
        rep (int): The experiment repetition which is being analyzed

    Returns:
        pd.DataFrame: filtered dataframe containing reception data
    """
    mask = (
        (df["Rep"] == rep) & 
        (df["NodeId"] == node_id) &
        (df["srcId"] != node_id)  # Only received, no transmitted messages
    )
    return df[mask]


def compute_twr_response_statistics(df: pd.DataFrame, timestamp_format: str = SIM) -> pd.DataFrame: # Consider renaming this function to get_successful_twr_rangings().
    """ Get information about the successful twr transmissions

    Args:
        df (pd.DataFrame): DataFrame to be analyzed

    Returns:
        pd.DataFrame: DataFrame containing statistics
    """

    unique_nodes    = df.loc[df["Node_Name"].str.contains("node", case=False, na=False), "NodeId"].unique()
    repetitions     = df["Rep"].unique()
    rows = []
    for rep in repetitions: 

        rep_df = filter_df(df, {"Rep": rep})
        print(f"Rep: {rep}")
        print(len(rep_df))

        for (node_id, seq_no), df_subset_raw in rep_df.groupby(["NodeId", "seqNo"]):
            if node_id not in unique_nodes: # Only mobile nodes
                continue
            df_subset = df_subset_raw[(df_subset_raw["src_is_anchor"]) | (df_subset_raw["srcId"] == node_id)]

            num_responses = (df_subset["msgType"]==1).sum()
            src_list = df_subset.loc[df_subset["msgType"] == 1, "srcId"].tolist()
            poll_rx_df, resp_tx_df, final_rx_df = _get_twr_anchor_timestamps(df, src_list, node_id, rep, seq_no)
            num_confirmed_responses = len(final_rx_df)


            estimated_pos, pos_error, final_tx_ts = (None, None, None) # Initialize optional output fields.
            if (num_confirmed_responses>=MIN_ANCHORS_FOR_TWR):
                estimated_pos, pos_error = _calc_twr_position(df_subset, node_id, poll_rx_df, resp_tx_df, final_rx_df, timestamp_format=timestamp_format)
                final_tx_ts = df_subset.loc[df_subset["msgType"] == FINAL, timestamp_format].item()

            rows.append({
                timestamp_format: final_tx_ts,
                "Rep"   : rep,
                "NodeId": node_id,
                "seqNo" : seq_no, 
                "numResponses": num_responses,
                "respondingAnchors": src_list,
                "numConfirmedResponses": num_confirmed_responses,
                "estimated_position": estimated_pos, 
                "pos_error": pos_error
            })

        results_df = pd.DataFrame(rows)
        results_df = results_df.sort_values(by=["Rep", "NodeId", "seqNo"]).reset_index(drop=True)

    return results_df

def _calc_twr_position(confirmed_responses_df: pd.DataFrame, node_id: int, poll_rx_df: pd.DataFrame, resp_tx_df: pd.DataFrame, final_rx_df: pd.DataFrame, timestamp_format: str = SIM):
    """ Estimate a node position from confirmed TWR anchor responses.

    Args:
        confirmed_responses_df (pd.DataFrame): Message subset containing confirmed responses for one node and sequence.
        node_id (int): ID of the mobile node.
        poll_rx_df (pd.DataFrame): Poll reception timestamps at anchors.
        resp_tx_df (pd.DataFrame): Response transmission timestamps at anchors.
        final_rx_df (pd.DataFrame): Final message reception timestamps at anchors.
        timestamp_format (str): Timestamp column used for timing calculations.

    Returns:
        tuple[np.ndarray, float]: Estimated position and position error.
    """
    confirmed_anchor_list = final_rx_df["NodeId"].unique().tolist()

    poll_tx_ts = confirmed_responses_df.loc[confirmed_responses_df["msgType"] == POLL, timestamp_format].item()
    final_tx_row = confirmed_responses_df.loc[confirmed_responses_df["msgType"] == FINAL].iloc[0]
    final_tx_ts = final_tx_row[timestamp_format]

    poll_rx_by_anchor = poll_rx_df.set_index("NodeId", drop=False)
    resp_tx_by_anchor = resp_tx_df.set_index("NodeId", drop=False)
    final_rx_by_anchor = final_rx_df.set_index("NodeId", drop=False)
    resp_rx_by_anchor = confirmed_responses_df.loc[
        (confirmed_responses_df["NodeId"] == node_id) &
        (confirmed_responses_df["srcId"].isin(confirmed_anchor_list))
    ].set_index("srcId", drop=False)

    ToFs = []
    anchor_positions = []
    for anchor_id in confirmed_anchor_list:
        poll_rx_ts = poll_rx_by_anchor.at[anchor_id, timestamp_format]
        resp_tx_ts = resp_tx_by_anchor.at[anchor_id, timestamp_format]
        resp_rx_ts = resp_rx_by_anchor.at[anchor_id, timestamp_format]
        final_rx_ts = final_rx_by_anchor.at[anchor_id, timestamp_format]

        anchor_position = _get_position_from_record(final_rx_by_anchor.loc[anchor_id])
        anchor_positions.append(anchor_position)
        tof = calc_twr_tof(
            R_a=(resp_rx_ts - poll_tx_ts),
            R_b=(final_rx_ts - resp_tx_ts),
            D_a=(final_tx_ts - resp_rx_ts),
            D_b=(resp_tx_ts - poll_rx_ts),
        )
        ToFs.append(tof)

    node_position = _get_position_from_record(final_tx_row)
    estimated_pos = estimate_twr_position(anchor_positions, ToFs)
    pos_error = math.dist(node_position, estimated_pos)

    return estimated_pos, pos_error

def filter_df_for_localization_method(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """ Take dataframe and extract only the required parts of it for further processing

    Args:
        df (pd.DataFrame): DataFrame to be reduced
        method (str): Name of the Method

    Returns:
        pd.DataFrame: Reduced DataFrame
    """
    df_filtered = df[df["Method"]==method].copy().reset_index(drop=True)
    return df_filtered

def _get_twr_anchor_timestamps(df: pd.DataFrame, anchor_id_list: list, node_id: int,  rep: int, seq_no: int):
    """ Extract anchor timestamps required for TWR time-of-flight calculation.

    Args:
        df (pd.DataFrame): Input dataframe containing all events.
        anchor_id_list (list): Anchor IDs that responded to the poll.
        node_id (int): ID of the mobile node.
        rep (int): Experiment repetition.
        seq_no (int): Sequence number of the TWR round.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Poll reception, response transmission, and final reception dataframes.
    """

    # for anchor_id in anchor_id_list:
    poll_rx_df = df[(df["NodeId"].isin(anchor_id_list)) &
                        (df["srcId"]   == node_id) &
                        (df["Rep"]     == rep) &
                        (df["msgType"] == POLL) &
                        (df["seqNo"]   == seq_no)
                        ] 
    resp_tx_df = df[(df["NodeId"].isin(anchor_id_list)) &
                        (df["srcId"]   == df["NodeId"]) &
                        (df["Rep"]     == rep) &
                        (df["msgType"] == RESP) &
                        (df["seqNo"]   == seq_no) &
                        (df["destId"]  == node_id)
                        ] 

    final_rx_df = df[(df["NodeId"].isin(anchor_id_list)) &
                        (df["srcId"]   == node_id) &
                        (df["Rep"]     == rep) &
                        (df["msgType"] == FINAL) &
                        (df["seqNo"]   == seq_no)
                        ] 

    return poll_rx_df, resp_tx_df, final_rx_df


def extract_data_from_csv(path: str):
    '''
    Take all the csv files and turn them into one big dataframe for further processing
    
    Args: 
        path (str): Path to result directory with all CSV traces of a given scenario
    Return:
        pd.DataFrame: Combined dataframe with Node, Rep, and Method columns added
    '''
    df = pd.DataFrame()
    dfs = []
    for directory in Path(path).iterdir():
        if not directory.is_dir():
            continue    

        dir_name = directory.name
        
        kv_pattern = re.findall(r'([^=,]+)=([^,\-]+)', dir_name)
        params = {}
        for key, value in kv_pattern:
            # Check whether the value is an integer.
            if value.isdigit():
                params[key] = int(value)
            else:
                # Remove # characters from string values.
                params[key] = value.replace("#", "")

        for file in directory.iterdir():
            node_index   = file.name.find("_")
            rep_index = file.name.find("rep")
            assert rep_index != -1 and node_index != -1, f"Unexpected Filename: {file.name}"
            
            csv_file = pd.read_csv(file)
            csv_file.columns = csv_file.columns.str.replace(" ", "", regex=False)
            
            csv_file["Node_Name"]    = file.name[:node_index]
            csv_file["Rep"] = int(re.search(r'rep(\d+)', file.name).group(1)) 

            # Add dynamic experiment parameters.
            for key, value in params.items():
                csv_file[key] = value

            dfs.append(csv_file)

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    set_node_id_in_df(df)
    set_node_and_src_is_anchor(df)
    return df


def set_node_and_src_is_anchor(df:pd.DataFrame):
    """ Add boolean columns identifying whether the source and receiver are anchors.

    Args:
        df (pd.DataFrame): Input dataframe containing NodeId, srcId, and Node_Name columns.

    Returns:
        None: The dataframe is updated in place.
    """
    num_anchors = get_num_nodes_by_name(df, "anchor") 
    df["src_is_anchor"]  = df["srcId"] <= num_anchors
    df["node_is_anchor"] = df["NodeId"] <= num_anchors


def set_node_id_in_df(df: pd.DataFrame):
    '''
    Get the NodeId from the nodeName for futher processing and add it as column to provided dataframe
    
    Parameters: 
        df: DataFrame retrieved from all data

    '''
    if "NodeId" in df.columns:
        return

    num_anchors = get_num_nodes_by_name(df, "anchor")
    unique_node_names = df["Node_Name"].unique()
    df["NodeId"] = None

    mapping = {}
    for unique_name in unique_node_names:    
        id_index  = int(re.findall(r'\d+', unique_name)[0])
        if "anchor" in unique_name:
            mapping[unique_name] = id_index +1
        elif "node" in unique_name:
            mapping[unique_name] = id_index + num_anchors +1
        else:
            raise ValueError("Node must be 'Node' or 'Anchor'")
    df["NodeId"] = df["Node_Name"].map(mapping).astype(int)

def get_num_nodes_by_name(df: pd.DataFrame, name: str) -> int:
    '''
    Get the number of individual nodes of different types/name
    
    :param df: DataFrame that we use
    :type df: pd.DataFrame
    :param name: Name of the node type (usually 'node' or 'anchor')
    :type name: str
    :return: Number of nodes with specified name
    :rtype: int
    '''
    num_nodes = df[df["Node_Name"].str.contains(name)]["Node_Name"].nunique()
    assert(num_nodes != 0)
    return num_nodes 


def calc_twr_tof(R_a, R_b, D_a, D_b):
    """ Compute the TWR time of flight from round and delay timestamps.

    Args:
        R_a (float): First round duration at side A.
        R_b (float): Second round duration at side B.
        D_a (float): Reply delay at side A.
        D_b (float): Reply delay at side B.

    Returns:
        float: Estimated one-way time of flight.
    """
    numerator = R_a*R_b - D_a*D_b
    denominator = R_a + D_a + R_b + D_b
    tof = numerator/denominator
    return tof

def estimate_twr_position(anchor_positions, ToFs, c=299792458.0):
    """ Estimate a 3D position from TWR time-of-flight measurements.

    Args:
        anchor_positions (list): Anchor coordinates as tuples or arrays.
        ToFs (list): One-way time-of-flight measurements in seconds.
        c (float): Propagation speed used to convert time of flight into distances.

    Returns:
        np.ndarray: Estimated 3D position.
    """
    # ChatGPT generated
    anchors = np.array(anchor_positions, dtype=float)
    distances = np.array(ToFs, dtype=float) * c  # ToF to distance

    def residual(p, anchors, d):
        """Return the difference between modeled and measured distances."""
        return np.sqrt(((anchors - p)**2).sum(axis=1)) - d

    # Starting estimate at mean of all anchor positions
    p0 = anchors.mean(axis=0)

    res = least_squares(residual, p0, args=(anchors, distances))
    return res.x


def filter_df(df:pd.DataFrame, filters: dict)->pd.DataFrame:
    """ Filter a dataframe by exact-match column-value pairs.

    Args:
        df (pd.DataFrame): Input dataframe to filter.
        filters (dict): Mapping from column names to required values.

    Returns:
        pd.DataFrame: Filtered dataframe view.
    """
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        mask &= df[col] == val
    return df.loc[mask]



def _expand_estimated_pos(df:pd.DataFrame, est_pos_column: str)->pd.DataFrame:
    """ Expand a vector-valued estimated position column into x, y, and z columns.

    Args:
        df (pd.DataFrame): Input dataframe containing estimated positions.
        est_pos_column (str): Column containing 3D position vectors.

    Returns:
        pd.DataFrame: Filtered dataframe with x_est, y_est, and z_est columns.
    """
    df_notna = df.loc[df[est_pos_column].notna()].copy()
    # positions
    # df_notna["x_est"] = np.array(df_notna[est_pos_column].tolist())[:,0]
    # df_notna["y_est"] = np.array(df_notna[est_pos_column].tolist())[:,1]
    # df_notna["z_est"] = np.array(df_notna[est_pos_column].tolist())[:,2]
    arr = np.array(df_notna[est_pos_column].tolist())
    df_notna["x_est"], df_notna["y_est"], df_notna["z_est"] = arr[:,0], arr[:,1], arr[:,2]
    return df_notna


def _compress_and_save_to_file(df: pd.DataFrame, file_name: str, filters: list[str]):
    """ Save only rows containing selected result columns to a CSV file.

    Args:
        df (pd.DataFrame): Input dataframe to compress and save.
        file_name (str): Output filename without extension.
        filters (list[str]): Columns where at least one non-null value is required.

    Returns:
        pd.DataFrame: Filtered dataframe that was written to disk.
    """
    mask = pd.Series(False, index=df.index)
    for filter in filters:
        mask |= df[filter].notna()
    df_filtered = df.loc[mask].copy()
    script_dir = Path(__file__).parent
    output_dir = script_dir / "tmp"
    output_dir.mkdir(exist_ok=True)
    file_path = output_dir / f"{file_name}.csv"
    df_filtered.to_csv(file_path, index=False, encoding="utf-8")
    print(f"Saved data in file "+ file_name)
    return df_filtered




def load_and_concat_results(methods: list, param_values: list, param_name: str, folder: str = "tmp") -> dict:
    """
    Load CSVs and return the concatenated dataframe.
    
    Return: {TWR: df_alle_speeds, CENTRALIZED_SELF_LOC: df_alle_speeds, ...}
    """
    # TODO: Improve this loading interface and reduce hard-coded filename conventions.
    method_dfs = {}
    for method in methods:
        dfs = []
        for val in param_values:
            file_path = Path(folder) / f"{method.lower()}_{param_name}_{val}.csv"
            if not file_path.exists():
                print(f"Warning: {file_path} not found, skipping.")
                continue
            dfs.append(pd.read_csv(file_path))
        
        if dfs:
            method_dfs[method] = pd.concat(dfs, ignore_index=True)
            print(f"{method} Loaded: {len(method_dfs[method])} Rows")
    
    return method_dfs

def remove_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
    """ Remove scalar outliers from one dataframe column using the IQR rule.

    Args:
        df (pd.DataFrame): Input dataframe containing the target column.
        col (str): Column used for IQR-based filtering.
        factor (float): IQR multiplier used to define lower and upper limits.

    Returns:
        pd.DataFrame: Filtered dataframe containing rows inside the IQR limits.
    """
    q1  = df[col].quantile(0.25)
    q3  = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    filtered = df.loc[df[col].between(lower, upper)]
    print(f"{col}: {len(df) - len(filtered)} Outliers removed ({len(filtered)} Rows remain)")
    return filtered


def compute_statistics(df: pd.DataFrame, method: str, timestamp_format = SIM, use_v_twr_pos: bool = True, save_to_file: bool = True, filter_dict: dict = {}):
    """ Compute and optionally save statistics for one localization method and filter combination.

    Args:
        df (pd.DataFrame): Input dataframe already filtered to the selected method or parameter subset.
        method (str): Localization method to process.
        timestamp_format (str): Timestamp column used for timing calculations.
        use_v_twr_pos (bool): Whether virtual TWR positions should be used for non-anchor nodes.
        save_to_file (bool): Whether compressed result CSV files should be written.
        filter_dict (dict): Parameter filter values used to build the output filename suffix.

    Returns:
        None: Results are written to disk when save_to_file is enabled.
    """
    suffix = ""
    for key in filter_dict: 
        suffix += "_" + str(key) + "_" + str(filter_dict[key])
    print(f"{suffix=}")

    file_name = method + "_" + timestamp_format + suffix
    print(file_name)
    match method: 
        case "TWR":
            twr_response_df = compute_twr_response_statistics(df, timestamp_format= timestamp_format)
            _compress_and_save_to_file(twr_response_df, file_name,["estimated_position"])

        case "HYPERBOLIC_SELF_LOCALIZATION" | "CENTRALIZED" | "DISTRIBUTED":
            vtwr_suffix= ""
            if use_v_twr_pos: 
                vtwr_suffix = "_with_vtwr"
            self_loc_response_df = compute_self_loc_statistics(df, timestamp_format = timestamp_format, use_v_twr_pos=use_v_twr_pos)
            
            if save_to_file:
                _compress_and_save_to_file(self_loc_response_df, file_name + vtwr_suffix,["v_twr_pos", "self_loc_pos_estimate"])

        case _:
            raise Exception(f"{method} is not a valid method") 
    print("LOADED data to DF for " + method)


def _parse_parameter_list(parameter_string: str):
    """ Parse a command-line parameter specification into a parameter name and integer values.

    Args:
        parameter_string (str): Parameter string in the form name:v1,v2,v3.

    Returns:
        tuple[str, list[int]]: Parsed parameter name and list of integer values.
    """
    # Separate elements by comma.
    name_part, list_part = parameter_string.split(':')
    parameter_list = [int(x) for x in list_part.split(',')]
    # Cast values to integers and return them as a list.
    return (name_part, parameter_list)


def parameter_validity_check(df: pd.DataFrame, experiment_parameter_arg):
    """ Validate that requested experiment parameters exist in the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe containing experiment parameters.
        experiment_parameter_arg (list | None): Command-line parameter specifications.

    Returns:
        None: Raises an exception if a parameter is invalid.
    """
    if experiment_parameter_arg is None: 
        return
    for experiment_parameter in experiment_parameter_arg:
        name, values = _parse_parameter_list(experiment_parameter)
        # Check whether the parameter column exists.
        if name not in df.columns:
            raise Exception(f"Parameter {name} is not in the columns!")
        
        if not set(values).issubset(df[name].unique()):
            missing = set(values) - set(df[name].unique())
            raise Exception(f"{missing} is missing in values ({values}) in the column {name}")
        print(f"Column {name} exists and has entries that match with {values}")




def filter_residual_outliers_per_row(df: pd.DataFrame, residuals_column: str ="residuals", error_column: str="self_loc_error", k=3.0):
    """ Filter residual outliers independently for each row using the median absolute deviation.

    Args:
        df (pd.DataFrame): Input dataframe containing residual vectors and error values.
        residuals_column (str): Column containing residual vectors.
        error_column (str): Column that must contain a valid localization error.
        k (float): Robust sigma multiplier used as outlier threshold.

    Returns:
        pd.DataFrame: Filtered dataframe containing valid rows that passed the row-wise residual test.
    """
    df = df.copy()
    
    # Only consider rows with valid residuals and errors.
    valid_mask = df[residuals_column].notna() & df[error_column].notna()
    df_valid = df.loc[valid_mask].copy()
    
    keep_mask = []
    for r_list in df_valid[residuals_column]:
        if r_list is None:
            keep_mask.append(False)
            continue
        r = np.ravel(r_list)
        median = np.median(r)
        mad = np.median(np.abs(r - median))
        sigma_robust = 1.4826 * mad
        keep_mask.append(np.all(np.abs(r - median) <= k * sigma_robust))
    
    filtered_df = df_valid.loc[keep_mask]
    
    n_total = len(df_valid)
    n_removed = n_total - len(filtered_df)
    print(f"{residuals_column}: {n_removed} Outliers removed ({len(filtered_df)} of {n_total} valid rows remain)")
    
    return filtered_df


def filter_residual_outliers_per_parameter(
    df: pd.DataFrame,
    group_cols: list[str],
    residuals_column="residuals",
    k: float = 3.0,
    verbose: bool = True
) -> pd.DataFrame:
    """ Filter residual outliers within parameter groups using the median absolute deviation.

    Args:
        df (pd.DataFrame): Input dataframe containing residual vectors.
        group_cols (list[str]): Columns used to define independent filtering groups.
        residuals_column (str): Column containing residual vectors.
        k (float): Robust sigma multiplier used as outlier threshold.
        verbose (bool): Whether to print per-group and overall filtering statistics.

    Returns:
        pd.DataFrame: Dataframe with filtered residual rows and unchanged rows without residuals.
    """

    df = df.copy()

    # --- 1. Split rows with residuals from rows without residuals. ---
    valid_mask = df[residuals_column].notna()
    df_valid = df[valid_mask]
    df_other = df[~valid_mask]

    filtered_dfs = []

    n_valid_total = len(df_valid)
    n_removed_total = 0

    removed_per_group = []

    # --- 2. Filter each group independently. ---
    for group_key, group in df_valid.groupby(group_cols):

        if len(group) == 0:
            removed_per_group.append((group_key, 0, 0))
            continue

        # Collect residuals.
        all_residuals = np.hstack([np.ravel(row) for row in group[residuals_column]])

        # Robust statistics.
        median = np.median(all_residuals)
        mad = np.median(np.abs(all_residuals - median))
        sigma_robust = 1.4826 * mad

        # Check rows.
        keep_mask = group[residuals_column].apply(
            lambda row: np.all(np.abs(np.ravel(row) - median) <= k * sigma_robust)
        )

        filtered_group = group.loc[keep_mask]

        # Statistics.
        n_removed_group = len(group) - len(filtered_group)
        n_removed_total += n_removed_group

        removed_per_group.append((group_key, len(group), n_removed_group))

        filtered_dfs.append(filtered_group)

    # --- 3. Concatenate filtered rows and unchanged rows. ---
    filtered_valid_df = pd.concat(filtered_dfs, ignore_index=True) if filtered_dfs else pd.DataFrame()

    final_df = pd.concat([filtered_valid_df, df_other], ignore_index=True)

    # --- 4. Print summary. ---
    if verbose:
        print("\n--- Outlier Removal per Group ---")
        for group_key, n_valid, n_removed in removed_per_group:
            print(f"Group {group_key}: removed {n_removed} of {n_valid} (kept {n_valid - n_removed})")

        print("\n--- Overall ---")
        print(f"Valid rows (with residuals): {n_valid_total}")
        print(f"Removed: {n_removed_total}")
        print(f"Kept: {n_valid_total - n_removed_total}")
        print(f"Unchanged rows (no residuals): {len(df_other)}")
        print(f"Final total rows: {len(final_df)}")

    return final_df

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="omnet_data_visualization")
    parser.add_argument("--method",
                        nargs="+",
                        type=str, 
                        help="What method should be used \n\t('TWR', 'HYPERBOLIC_SELF_LOCALIZATION', 'CENTRALIZED', 'DISTRIBUTED')")
    
    parser.add_argument("--experiment_path",
                        type=str, 
                        help="Path to  experiment to use \n\t('Baseline', 'Speed', 'NodeNumber')")
    
    parser.add_argument("--experiment_parameter",
                        nargs="+",
                        help="List of parameter name and value pairs (comma-separated) for data extraction\n\te.g. 'speed:1,5,10,20' or  'NodeNum:4,8,16,32'")
    
    parser.add_argument("--timestamp_format", 
                        type=str, 
                        default="sim_timestamp",
                        help="'sim_timestamp' or 'clock_timestamp' or 'noisy_timestamp'")
    
    parser.add_argument("--use_vtwr_pos",
                        type=bool,
                        default=True, 
                        help="(For HyRASeL mechanisms), whether to use the true position of the node or the position estimated with vtwr")

    parser.add_argument("--rep_limit",
                        type=int,
                        help="Set the upper limit for the number of repetitions to consider (int)"
                        )

    args = parser.parse_args()
    
    # Mainly used for data extraction.

    # %%
    timestamp_format = args.timestamp_format
    use_vtwr_pos = args.use_vtwr_pos
    experiment_path = args.experiment_path
    rep_limit = args.rep_limit


    df = extract_data_from_csv(experiment_path)
    print(f"Loaded Data from {experiment_path}")
    print(df)


    # TODO: Add a command-line parameter for NOISE_SIGMA_T.
    NOISE_SIGMA_T = 300e-12 # 150 ps

    NOISE_SEED = 42
    rng = np.random.default_rng(NOISE_SEED)
    noise = rng.normal(
        loc=0.0,
        scale=NOISE_SIGMA_T,
        size=len(df)
    )
    # # Add timestamp noise.
    # noise = np.random.normal(
    #     loc=0.0,
    #     scale=NOISE_SIGMA_T,
    #     size=len(df)
    # )
    # This mainly makes sense for clock timestamps.
    df["noisy_timestamp"] = df["clock_timestamp"] + noise

    if args.experiment_parameter is not None: 
        parameter_names = []
        parameter_values_lists = []
        for experiment_parameter in args.experiment_parameter:
            name, values = _parse_parameter_list(experiment_parameter)
            parameter_names.append(name)
            parameter_values_lists.append(values)
        parameter_validity_check(df, args.experiment_parameter)
    
    if rep_limit is not None: 
        df = df[df["Rep"] < rep_limit].reset_index(drop=True)
        print(f"Reduced DF \n{df}")
    

    for method in args.method:  
        method_df  = filter_df_for_localization_method(df, method).copy() # Can use filter_df() for this as well

        # Process all parameter combinations.

        if not args.experiment_parameter: 
            print("No parameters provided (bc not necessary?)")
            compute_statistics(method_df, method, timestamp_format, use_vtwr_pos)
        else:
            for combination in product(*parameter_values_lists):
                filter_dict = dict(zip(parameter_names, combination))
                filtered_df = filter_df(method_df, filter_dict).copy()
                print(f"Compute for {method=}, {timestamp_format=}, {use_vtwr_pos=}, {filter_dict=}")
                compute_statistics(filtered_df, method, timestamp_format, use_vtwr_pos, filter_dict=filter_dict)
                print(f"Computation done for {method=}, {timestamp_format=}, {use_vtwr_pos=}, {filter_dict=}")
