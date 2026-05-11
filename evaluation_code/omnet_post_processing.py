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
from datetime import datetime, time


from scipy.optimize import least_squares


# OVERALL CONSTANTS 
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

def solve_tdoa_least_squares(
    tdoa_sets: list[tuple[np.ndarray, np.ndarray]],
    speed_of_signal: float = 299792458.0,
    initial_guess: np.ndarray | None = None,
    sigma_r: float = 1.0  
) -> tuple[np.ndarray, dict]:
    """
    Args:
        tdoa_sets: List of Tupels:
                   [(anchor_positions, time_diffs), ...]
                   anchor_positions: Shape (N,2) or (N,3)
                   time_diffs: Shape (N-1,)
    """
    if initial_guess is None:
        all_anchors = np.vstack([anchors for anchors, _ in tdoa_sets])
        initial_guess = np.mean(all_anchors, axis=0)

    def residuals(pos):

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

    result = least_squares(residuals, 
                           initial_guess, 
                           method="lm")

    info = {
        "success": result.success,
        "cost": float(result.cost),
        "iterations": result.nfev,
        "residuals": result.fun
    }

    # GDoP-Calculation (Generated with AI)    
    # Jacobian at estimated position
    J = result.jac  # shape: (num_residuals, dim_position)
    JTJ = J.T @ J
    Q = np.linalg.pinv(JTJ) * sigma_r**2

    # PDOP / HDOP / VDOP
    dim = Q.shape[0]
    PDOP = np.sqrt(np.trace(Q))
    if dim == 3:
        HDOP = np.sqrt(Q[0,0] + Q[1,1])
        VDOP = np.sqrt(Q[2,2])
    else:  # 2D-Case
        HDOP = np.sqrt(Q[0,0] + Q[1,1])
        VDOP = None

    cond = np.linalg.cond(JTJ)
    info.update({"PDOP": PDOP, "HDOP": HDOP, "VDOP": VDOP, "Q_pos": Q, "cond": cond})

    return result.x, info


def compute_self_loc_statistics(df: pd.DataFrame, timestamp_format: str = SIM, use_v_twr_pos: bool = False) -> pd.DataFrame:
    # Initialize/add new fields that are to be filled during runtime

    df["_idx"] = df.index
    df_lookup = df.set_index(["NodeId", "srcId", "seqNo", "Rep"]).sort_index()

    df["self_loc_pos_estimate"] = None
    df["self_loc_error"] = None
    df["residuals"] = None
    df["v_twr_pos"] = None
    df["v_twr_error"] = None
    df["using_anchor_stitching"] = False
    df["PDOP"] = None
    df["cond"] = None

    # Extract info from the df
    repetitions     = df["Rep"].unique()
    unique_seq_nos  = df["seqNo"].unique()
    mobile_node_ids    = df.loc[df["Node_Name"].str.contains("node", case=False, na=False), "NodeId"].unique()

    # # Get virtual TWR Statistics
    _estimate_node_positions_virtual_twr(df, df_lookup, mobile_node_ids, repetitions, timestamp_format)

    # Reload df_lookup with new columns
    df_lookup = df.set_index(["NodeId", "srcId", "seqNo", "Rep"]).sort_index()

    # Get Self Loc Statistics
    for node_id, rep in product(mobile_node_ids, repetitions):
        # Get all data, that the specific node receives
        node_rx_df = _get_node_reception_data(df, node_id, rep)
        node_rx_df["tdoa_info"] = None
        for seq_no in unique_seq_nos:
            seq_rx_df = node_rx_df.loc[node_rx_df["seqNo"] == seq_no]

            # SELF LOC Localization
            for msg_record_idx in seq_rx_df.index: 
                _compute_localization_per_incoming_message(df, df_lookup, node_id, seq_rx_df, node_rx_df, rep, seq_no, msg_record_idx, timestamp_format=timestamp_format, use_v_twr_pos=use_v_twr_pos)


    return df

def _estimate_node_positions_virtual_twr(df: pd.DataFrame, df_lookup: pd.DataFrame, mobile_node_ids: list, repetitions: list, timestamp_format: str = SIM):
    # Get virtual TWR Statistics
    for node_id, rep in product(mobile_node_ids, repetitions):
        # Get all entries where the node transmits
        node_tx_df = df.query("srcId == @node_id and NodeId == @node_id and Rep == @rep")
        node_rx_df = _get_node_reception_data(df, node_id, rep) # pd.Series
        for msg_record in node_tx_df.iloc[::-1].itertuples(): # Itertuples for speed
            v_final_seq_no = msg_record.seqNo
            v_poll_seq_no  = v_final_seq_no - 1
            # Was there a previous message?
            v_poll_tx = _get_transmission_for_receiver(df_lookup, node_id, node_id, v_poll_seq_no, rep)
            if v_poll_tx is None: 
                continue # There was no previous message then
            
            v_poll_tx_idx = int(v_poll_tx["_idx"]) # Pandas Series -> .name
            v_final_tx_idx = msg_record.Index # Pandas Core -> Name of Index Entry
            between_df = node_rx_df.loc[v_poll_tx_idx:v_final_tx_idx]

            # ONLY ANCHORS FOR LOCALIZATION!
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
                v_resp_tx_ts = _get_transmission_for_receiver(df_lookup, anchor_id, anchor_id, anchor_seq_no, rep)[timestamp_format]
                v_poll_rx_ts = _get_transmission_for_receiver(df_lookup, node_id, anchor_id, v_poll_seq_no, rep)[timestamp_format]
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
                    raise Exception("Some shit went wrong")
                anchor_pos = np.array(_get_transmission_for_receiver(df_lookup, node_id, anchor_id, v_final_seq_no, rep)[["X", "Y", "Z"]])
                anchor_positions.append(anchor_pos)
            node_position = np.array(_get_transmission_for_receiver(df_lookup, node_id, node_id, v_final_seq_no, rep)[["X", "Y", "Z"]])
            estimated_pos = estimate_twr_position(anchor_positions, tofs)

            pos_error = math.dist(node_position, estimated_pos)

            df.at[v_final_tx_idx, "v_twr_pos"] = estimated_pos.tolist()
            df.at[v_final_tx_idx, "v_twr_error"] = pos_error



def _compute_localization_per_incoming_message(df: pd.DataFrame, df_lookup: pd.DataFrame, node_id: int, seq_rx_df: pd.DataFrame, node_rx_df: pd.DataFrame, rep: int, seq_no: int, msg_record_idx: int, timestamp_format: str = SIM, use_v_twr_pos: bool = False):
    
    # Get the information to the message that the current node receives (one row only)
    node_current_rx = seq_rx_df.loc[msg_record_idx]
    
    # Get the sender that was transmitting this message
    sender_id = node_current_rx["srcId"]

    # Get the previous information from the previous transmission of the sender
    node_prev_rx = _get_node_reception_from_sender(node_rx_df, sender_id, seq_no-1) # pd.Series

    # If there was no previous transmission, we can continue with the next row, as we are not going to get a localization
    if node_prev_rx is None:
        return

    # Get index of the current and the last transmission of the sender
    prev_index = node_prev_rx.name
    current_index = node_current_rx.name 
    between_df = node_rx_df.loc[prev_index:current_index]

    # Get the IDs of all the anchors, that the sender and the receiver node have heard
    anchor_ids, anchor_seq_nos = _get_valid_anchors_with_seqnos(between_df, df_lookup, sender_id, seq_no, rep)

    # Get the true position at the time at which the node receives the message
    true_pos = (node_current_rx["X"], node_current_rx["Y"], node_current_rx["Z"])
    current_index = node_current_rx.name

    # If the sender is a v-Anchor and does not have a current position from v-twr, we cannot use it further and 
        # thus, we stop and continue with the next incoming message
    sender_current_tx = _get_transmission_for_receiver(df_lookup, sender_id, sender_id, seq_no, rep)
    sender_current_pos = sender_current_tx["v_twr_pos"]
    if (sender_current_pos is None) & (sender_current_tx["node_is_anchor"]==False):
        return

    # Get all the time differences and anchor positions
    anchor_positions, time_diffs = _compute_delta_ts_for_round(df, df_lookup, anchor_ids, anchor_seq_nos, sender_id, node_id, seq_no, rep, timestamp_format=timestamp_format, use_v_twr_pos=use_v_twr_pos)    
    
    # We need to overhear at least 3 additional anchors (sender is an anchor itself, thus we have 4 in total) to continue
    valid_anchor_pairs = [(int(sender_id), int(anchor_id)) for anchor_id in anchor_ids]

    # START ANCHOR STITCHING LOGIC
    # I need the position of the reference anchor, position of the individual anchor, Ich brauche die Position des Referenz Ankers, die Position des Ankers, 
    tdoa_info = {}
    for anchor_id, time_diff, anchor_pos in zip(anchor_ids, time_diffs, anchor_positions[1:]):
        sender_position = anchor_positions[0]
        tdoa_info[(int(sender_id), int(anchor_id))] = [current_index, time_diff, sender_position, anchor_pos]
            
    node_rx_df.at[current_index, "tdoa_info"] = tdoa_info
    

    # PERFORM ANCHOR STITCHING
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
                # sortiere das Paar, um (1,3) == (3,1) zu behandeln
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
        result = [{original_pair: info_list} for original_pair, info_list in seen_pairs.values()]
        

        if len(result) >= MIN_TIME_DIFFS_FOR_SELF_LOCALIZATION:            
            # Get list of anchor_positions and time_diffs
            ref_anchor_groups = {}

            for pair_dict in result:
                for anchor_pair, pair_info in pair_dict.items():
                    # First entry in tuple is the reference anchor
                    ref_anchor = anchor_pair[0]  
                    index, time_diff, ref_pos, other_pos = pair_info

                    if ref_anchor not in ref_anchor_groups:
                        ref_anchor_groups[ref_anchor] = []

                    ref_anchor_groups[ref_anchor].append((index, time_diff, ref_pos, other_pos))

            # Sort per reference anchor by index so that the newest position comes first
            final_stitched_list = []

            for ref_anchor, entries in ref_anchor_groups.items():
                # Sort descending by indey
                entries_sorted = sorted(entries, key=lambda x: x[0], reverse=True)

                # Reference-Position = Position of newest entry
                ref_pos_latest = entries_sorted[0][2]
                positions_list = [ref_pos_latest]  
                time_diffs_list = []

                for entry in entries_sorted:
                    _, td, _, other_pos = entry
                    positions_list.append(other_pos)
                    time_diffs_list.append(td)

                final_stitched_list.append((np.asarray(positions_list), np.asarray(time_diffs_list)))

            estimated_pos, info = solve_tdoa_least_squares(final_stitched_list)
            # Indicate that this position was achieved with anchor stitching
            df.at[current_index, "using_anchor_stitching"] = True
            error = math.dist(true_pos, estimated_pos)    
            df.at[current_index, "self_loc_pos_estimate"] = estimated_pos
            df.at[current_index, "self_loc_error"] = error
            df.at[current_index, "residuals"] = info["residuals"]
            df.at[current_index, "PDOP"] = info["PDOP"]
            df.at[current_index, "cond"] = info["cond"]
        return

    # If we have less than 3 time differences (which is the required minimum for 3D Hyperbolic Localization, we do not continue
    if (len(time_diffs) < MIN_TIME_DIFFS_FOR_SELF_LOCALIZATION):
        return
    
    # Use the time diffs and anchor positions to estimate the position
    estimated_pos, info = solve_tdoa_least_squares([(anchor_positions, time_diffs)])

    error = math.dist(true_pos, estimated_pos)    
    df.at[current_index, "self_loc_pos_estimate"] = estimated_pos
    df.at[current_index, "self_loc_error"] = error
    df.at[current_index, "residuals"] = info["residuals"]
    df.at[current_index, "PDOP"] = info["PDOP"]
    df.at[current_index, "cond"] = info["cond"]
    return True # DEBUG



def _get_valid_anchors_with_seqnos(between_df: pd.DataFrame, df_lookup: pd.DataFrame, sender_id: int, seq_no: int, rep: int) -> list:
    # Get all anchors that are not the sender himself and put them into a list
    anchor_ids = []
    anchor_seq_nos = []

    # Check for all anchors whose message was received within last ranging round
    for anchor_id in between_df["srcId"].unique():
        # The sender itself is also in the list, so lets not use it
        if anchor_id == sender_id: 
            continue

        # Check whether the sender has received a message from all the anchors that were received within the last ranging round
            # at the node, as this is prerequisite for TDoA estimation
        anchor_current_rx_sender    = _get_transmission_for_receiver(df_lookup, sender_id, anchor_id, seq_no, rep)
        anchor_prev_rx_sender       = _get_transmission_for_receiver(df_lookup, sender_id, anchor_id, seq_no-1, rep)
        
        # If either the current or the last message was not received, this anchor is not valid
        if anchor_current_rx_sender is None or anchor_prev_rx_sender is None: 
            continue
        
        # Get the most recent sequence number (iloc[-1]) of the anchor
        anchor_seq_no_df = between_df.loc[between_df["srcId"]==anchor_id]
        anchor_seq_no = anchor_seq_no_df.iloc[-1]["seqNo"]

        # If the received tx is from a v-anchor, we need to check, whether it was able to get its position 
            # using v-TWR; if not, we will not futher consider the node
        if (anchor_current_rx_sender["node_is_anchor"] == False): #
            anchor_current_tx = _get_transmission_for_receiver(df_lookup, anchor_id, anchor_id, anchor_seq_no, rep)
            anchor_position_vtwr = anchor_current_tx["v_twr_pos"]
            if anchor_position_vtwr is None: 
                continue

        # If everything went well, we add the node to the list
        anchor_ids.append(anchor_id)
        anchor_seq_nos.append(anchor_seq_no)

    return anchor_ids, anchor_seq_nos


def _compute_delta_ts_for_round(df: pd.DataFrame, df_lookup: pd.DataFrame, anchor_ids: list, anchor_seq_nos: list, sender_id:int, node_id: int, seq_no: int, rep: int, timestamp_format: str = SIM, use_v_twr_pos: bool = False)->tuple[np.array, np.array]:
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
                           t_ab: float)->float:
    R_ba = b_rx2_a - b_tx1
    D_ba = b_tx1 - b_rx1_a
    S_ab = s_rx1_b - s_rx1_a
    S_ba = s_rx2_a - s_rx1_b
    numerator   = (R_ba*S_ab)-(S_ba*D_ba)-(t_ab*R_ba)-(t_ab*S_ab)+(t_ab**2)
    denominator = R_ba+D_ba-t_ab
    delta_t =  (numerator / denominator)
    return delta_t

def _get_transmission_for_receiver(df_lookup:pd.DataFrame, sender_id: int, receiver_id: int, seq_no: int, rep: int)-> pd.Series | None:
    key = (receiver_id, sender_id, seq_no, rep)
    try:
        row = df_lookup.loc[key]
        if isinstance(row, pd.DataFrame):
            return row.iloc[0] if len(row) == 1 else None
        return row
    except KeyError:
        return None



def _get_node_reception_from_sender(node_rx_df: pd.DataFrame, sender_id: int, seq_no: int) -> pd.Series | None:
    """Holt Empfangsdaten eines bestimmten Senders für eine Sequenz."""
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


def compute_twr_response_statistics(df: pd.DataFrame, timestamp_format: str = SIM) -> pd.DataFrame: # Maybe rename to getSuccessfulTwrRangings()
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


            estimated_pos, pos_error, final_tx_ts = (None, None, None) # INIT
            if (num_confirmed_responses>=MIN_ANCHORS_FOR_TWR):
                estimated_pos, pos_error = _calc_twr_position(df_subset, node_id, poll_rx_df, resp_tx_df, final_rx_df, timestamp_format=timestamp_format)
                final_tx_ts = df_subset.query("msgType == @FINAL")[timestamp_format].item()

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
    confirmed_anchor_list = final_rx_df["NodeId"].unique().tolist()
    poll_tx_ts = confirmed_responses_df.query("msgType == @POLL")[timestamp_format].item()
    final_tx_ts = confirmed_responses_df.query("msgType == @FINAL")[timestamp_format].item() # Item throws error when it has more than one entry

    ToFs = []
    anchor_positions = []
    for anchor_id in confirmed_anchor_list:
        poll_rx_ts = poll_rx_df.query("NodeId == @anchor_id")[timestamp_format].item()
        resp_tx_ts = resp_tx_df.query("NodeId == @anchor_id")[timestamp_format].item()
        resp_rx_ts = confirmed_responses_df.query("srcId == @anchor_id and NodeId == @node_id")[timestamp_format].item()
        final_rx_ts= final_rx_df.query("NodeId== @anchor_id")[timestamp_format].item()

        anchor_position = tuple(final_rx_df.query("NodeId == @anchor_id")[["X", "Y", "Z"]].squeeze().tolist())
        anchor_positions.append(anchor_position)
        tof = calc_twr_tof(
            R_a = (resp_rx_ts - poll_tx_ts),
            R_b = (final_rx_ts - resp_tx_ts), 
            D_a = (final_tx_ts - resp_rx_ts),
            D_b = (resp_tx_ts - poll_rx_ts)
        )
        ToFs.append(tof)

    node_position = confirmed_responses_df.query("msgType == @FINAL")[["X", "Y", "Z"]].squeeze().tolist()
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
            if value.isdigit():
                params[key] = int(value)
            else:
                params[key] = value.replace("#", "")

        for file in directory.iterdir():
            node_index   = file.name.find("_")
            rep_index = file.name.find("rep")
            assert rep_index != -1 and node_index != -1, f"Unexpected Filename: {file.name}"
            
            csv_file = pd.read_csv(file)
            csv_file.columns = csv_file.columns.str.replace(" ", "", regex=False)
            
            csv_file["Node_Name"]    = file.name[:node_index]
            csv_file["Rep"] = int(re.search(r'rep(\d+)', file.name).group(1)) 

            # dynamic parameter
            for key, value in params.items():
                csv_file[key] = value

            dfs.append(csv_file)

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    set_node_id_in_df(df)
    set_node_and_src_is_anchor(df)
    return df


def set_node_and_src_is_anchor(df:pd.DataFrame):
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
    numerator = R_a*R_b - D_a*D_b
    denominator = R_a + D_a + R_b + D_b
    tof = numerator/denominator
    return tof

def estimate_twr_position(anchor_positions, ToFs, c=299792458.0):
    """
    Estimates the 3D position based on ToF measurements.

    Parameters
    ----------
    anchor_positions : list of tuple
        List of anchor coordinates [(x1,y1,z1), (x2,y2,z2), ...]
    ToFs : list of float
        ToF measurements in seconds (one-way) or already as distances
    c : float
        Signal propagation speed (default: speed of light in m/s)

    Returns
    -------
    np.ndarray
        Estimated position [x, y, z]
    """
    # Generated with AI
    anchors = np.array(anchor_positions, dtype=float)
    distances = np.array(ToFs, dtype=float) * c  # ToF → Distanz

    def residual(p, anchors, d):
        """Differenz zwischen gemessenen Distanzen und berechneten Abständen"""
        return np.sqrt(((anchors - p)**2).sum(axis=1)) - d

    # Starting estimate at mean of all anchor positions
    p0 = anchors.mean(axis=0)

    res = least_squares(residual, p0, args=(anchors, distances))
    return res.x


def compute_inter_measurement_time(df: pd.DataFrame, filter_col, timestamp_col=SIM):
    # Generated with AI
    result = df.loc[df[filter_col].notna()].copy()
    result["inter_measurement_time"] = result[timestamp_col].diff()
    result["update_frequency"] = 1 / result["inter_measurement_time"]
    return result.loc[result["inter_measurement_time"] > 0]


def plot_ecdf(datasets: list[dict], xlabel="Position Error [m]", ylabel="CDF",
              xlim=None, ylim=None, xticks=None, yticks=None, xscale=None, save_path: str = None):
    """
    datasets: list of dicts with keys:
        - df: DataFrame
        - x: column name to plot
        - label: legend label
        - filter_col: column to filter notna (optional, defaults to x)
        - min_val: minimum value filter (optional, e.g. for inter_measurement_time > 0)
    """
    # Generated with AI
    plt.figure()
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    for i, ds in enumerate(datasets):
        filter_col = ds.get("filter_col", ds["x"])
        df = ds["df"].loc[ds["df"][filter_col].notna()]
        if "min_val" in ds:
            df = df.loc[df[ds["x"]] > ds["min_val"]]
        sns.ecdfplot(
            data=df,
            x=ds["x"],
            label=ds["label"],
            linestyle=linestyles[i % len(linestyles)]
        )

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if xscale is not None:
        plt.xscale(xscale)
    plt.grid(True)

    if save_path is not None:
        Path("tmp/figs").mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        plt.savefig(f"tmp/figs/{timestamp}_{save_path}.pdf", bbox_inches="tight")
    plt.show()


def filter_df(df:pd.DataFrame, filters: dict)->pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        mask &= df[col] == val
    return df.loc[mask]



def _expand_estimated_pos(df:pd.DataFrame, est_pos_column: str)->pd.DataFrame:
    df_notna = df.loc[df[est_pos_column].notna()].copy()
    arr = np.array(df_notna[est_pos_column].tolist())
    df_notna["x_est"], df_notna["y_est"], df_notna["z_est"] = arr[:,0], arr[:,1], arr[:,2]
    return df_notna


def _compress_and_save_to_file(df: pd.DataFrame, file_name: str, filters: list[str]):
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

def load_results_from_csv(files: list, folder: str = "tmp", append_parameters: list = None, append_value: str= None)->pd.DataFrame:
    data_list = []
    pairs = zip(files, append_parameters) if append_parameters else ((f, None) for f in files)
    for file, appendix in pairs:
        file_path = Path(folder) / f"{file}.csv"
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping.")
            continue
        data_df = pd.read_csv(file_path)
        if append_parameters is not None and append_value:
            data_df[append_value] = appendix
        # Make residual column to array not string 
        if "residuals" in data_df.columns:
            data_df["residuals"] = data_df["residuals"].apply(
                lambda x: np.fromstring(x.strip("[]"), sep=" ") if pd.notna(x) else None
            )

        data_list.append(data_df)
    if not data_list:
        return pd.DataFrame()

    df = pd.concat(data_list, ignore_index=True)
    print(f"Loaded {files} in {len(df)} Rows")

    return df
def remove_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
    q1  = df[col].quantile(0.25)
    q3  = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    filtered = df.loc[df[col].between(lower, upper)]
    print(f"{col}: {len(df) - len(filtered)} Outliers removed ({len(filtered)} Rows remain)")
    return filtered


def compute_statistics(df: pd.DataFrame, method: str, timestamp_format = SIM, use_v_twr_pos: bool = True, save_to_file: bool = True, filter_dict: dict = {}):

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
    # Separate Elements with comma
    name_part, list_part = parameter_string.split(':')
    parameter_list = [int(x) for x in list_part.split(',')]
    # Cast to int and put into list
    return (name_part, parameter_list)


def parameter_validity_check(df: pd.DataFrame, experiment_parameter_arg):
    if experiment_parameter_arg is None: 
        return
    for experiment_parameter in experiment_parameter_arg:
        name, values = _parse_parameter_list(experiment_parameter)
        # Exists Name?
        if name not in df.columns:
            raise Exception(f"Parameter {name} is not in the columns!")
        
        if not set(values).issubset(df[name].unique()):
            missing = set(values) - set(df[name].unique())
            raise Exception(f"{missing} is missing in values ({values}) in the column {name}")
        print(f"Column {name} exists and has entries that match with {values}")


def filter_residual_outliers_globally(df: pd.DataFrame, residuals_column: str = "residuals", k: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    
    df_valid = df.loc[df[residuals_column].notna()].copy()
    
    # Flat all residuals for global MAD calculation
    all_residuals = []
    for r_list in df_valid[residuals_column]:
        all_residuals.extend(np.ravel(r_list))
    all_residuals = np.array(all_residuals)
    
    # Mean Absolute Deviation (MAD)
    median = np.median(all_residuals)
    mad = np.median(np.abs(all_residuals - median))
    sigma_robust = 1.4826 * mad
    
    # Check row-wise: alle Residuals within k*sigma
    keep_mask = df_valid[residuals_column].apply(
        lambda r: np.all(np.abs(np.ravel(r) - median) <= k * sigma_robust)
    )
    
    filtered_df = df_valid.loc[keep_mask].copy()
    
    n_total = len(df_valid)    
    n_removed = n_total - len(filtered_df)
    print(f"{residuals_column}: {n_removed} Outliers removed ({len(filtered_df)} of {n_total} valid rows remain)")
    
    return filtered_df

def filter_residual_outliers_per_row(df: pd.DataFrame, residuals_column: str ="residuals", error_column: str="self_loc_error", k=3.0):
    # Generated with AI
    df = df.copy()

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
    # Generated with AIChatGPT

    df = df.copy()

    # Split valid and others
    valid_mask = df[residuals_column].notna()
    df_valid = df[valid_mask]
    df_other = df[~valid_mask]

    filtered_dfs = []

    n_valid_total = len(df_valid)
    n_removed_total = 0

    removed_per_group = []

    # filter groups
    for group_key, group in df_valid.groupby(group_cols):

        if len(group) == 0:
            removed_per_group.append((group_key, 0, 0))
            continue

        # Collect residuals
        all_residuals = np.hstack([np.ravel(row) for row in group[residuals_column]])

        # robust statistics 
        median = np.median(all_residuals)
        mad = np.median(np.abs(all_residuals - median))
        sigma_robust = 1.4826 * mad

        # Check row
        keep_mask = group[residuals_column].apply(
            lambda row: np.all(np.abs(np.ravel(row) - median) <= k * sigma_robust)
        )

        filtered_group = group.loc[keep_mask]

        # Statistic
        n_removed_group = len(group) - len(filtered_group)
        n_removed_total += n_removed_group

        removed_per_group.append((group_key, len(group), n_removed_group))

        filtered_dfs.append(filtered_group)

    # Merge results
    filtered_valid_df = pd.concat(filtered_dfs, ignore_index=True) if filtered_dfs else pd.DataFrame()

    final_df = pd.concat([filtered_valid_df, df_other], ignore_index=True)

    # Output
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



def merge_exclusive_columns(
    df: pd.DataFrame,
    col1: str = "self_loc_error",
    col2: str = "v_twr_error",
    new_col: str = "hyrasel_pos_error"
) -> pd.DataFrame:
    """
    Combines two mutually exclusive columns into a new column.

    Args:
        df: pd.DataFrame
        col1: Name of the first column (default: "self_loc_error")
        col2: Name of the second column (default: "vtwr_pos_error")
        new_col: Name of the new column containing the merged values

    Returns:
        pd.DataFrame: Copy of the DataFrame with the new column

    Raises:
        TypeError: if df is not a pd.DataFrame
        KeyError: if a column does not exist
        ValueError: if both columns have values in the same row
    """
    # Generated with AI 

    for col in [col1, col2]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")

    overlap_mask = df[col1].notna() & df[col2].notna()
    if overlap_mask.any():
        raise ValueError(
            f"{overlap_mask.sum()} row(s) contain values in both '{col1}' and '{col2}'"
        )

    df = df.copy()
    df[new_col] = df[col1].combine_first(df[col2])

    return df

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
    
    # !Mainly for data extraction!

    # %%
    timestamp_format = args.timestamp_format
    use_vtwr_pos = args.use_vtwr_pos
    experiment_path = args.experiment_path
    rep_limit = args.rep_limit


    df = extract_data_from_csv(experiment_path)
    print(f"Loaded Data from {experiment_path}")
    print(df)

    NOISE_SIGMA_T = 300e-12 # 150 ps
    # Add noise
    noise = np.random.normal(
        loc=0.0,
        scale=NOISE_SIGMA_T,
        size=len(df)
    )

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
        method_df  = filter_df_for_localization_method(df, method).copy() 

        # All possible combinations

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
