"""Runs of a simulation config: their names, directories and raw traces, shared by all evaluation steps.

A run is one repetition of one method and sweep-value combination; its
parameters are the sweep values and Method, e.g. {"speed": 5, "Method": "TWR"}.
Its raw traces are one file per node, <node>_rep<N>_log.csv, in the raw
directory of its parameters (written by OMNeT++, e.g. speed=5,Method=TWR-).
Every step writes its results of a run to the run directory
<parameter directory>/rep<N>, e.g. speed=5,Method=TWR/rep3.
"""

from __future__ import annotations

import itertools
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

# Timestamp columns of the traces.
SIM = "sim_timestamp"
CLOCK = "clock_timestamp"
NOISY = "noisy_timestamp"
# Columns that identify a message; its sender and its receivers log the same values.
MESSAGE_COLUMNS = ["srcId", "destId", "seqNo", "interfaceIdx", "msgType"]

# Trace files: node name and repetition; gzipped traces (.csv.gz) are read as well.
TRACE_NAME_PATTERN = re.compile(r"((?:anchor|node)\[\d+\])_rep(\d+)_log\.csv(?:\.gz)?", re.IGNORECASE)
NODE_NAME_PATTERN = re.compile(r"(anchor|node)\[(\d+)\]", re.IGNORECASE)
REP_DIR_PATTERN = re.compile(r"rep(\d+)")

# Results of a config (results/<Config>/): the post-processed runs, and per evaluation directory
# (<estimator>/<filter profile>/) the extracted runs and the aggregates read by the plots.
POSTPROCESSED_DIR_NAME = "postprocessed"
EVENTS_FILE_NAME = "localization_events.csv"
RUN_METADATA_FILE_NAME = "run_metadata.json"
RUNS_DIR_NAME = "runs"
FILTERED_EVENTS_FILE_NAME = "localization_events_filtered.csv"
COVERAGE_FILE_NAME = "coverage.csv"
AVAILABILITY_FILE_NAME = "availability.csv"
AGGREGATED_FILE_NAME = "statistics_run_aggregated.csv"


# -----------------------------------------------------------------------------
# Names
# -----------------------------------------------------------------------------

def parse_value(text: str) -> int | float | str:
    """A parameter value as int or float if it is a number, otherwise the text itself (e.g. TWR)."""
    for number_type in (int, float):
        try:
            return number_type(text)
        except ValueError:
            pass
    return text


def parse_reps(specs: str | int | list) -> list[int]:
    """Repetitions of specs like 5, "1-10", "3,5,8-15" or [1, "4-6"], sorted and without duplicates."""
    reps = set()
    for spec in specs if isinstance(specs, list) else [specs]:
        for token in str(spec).split(","):
            token = token.strip()
            if not token:
                continue
            # "8-15" -> 8..15; "3" has no end and gives 3..3.
            first, _, last = token.partition("-")
            first, last = int(first), int(last or first)
            if first > last:
                raise ValueError(f"Invalid repetition range '{token}': start > end.")
            reps.update(range(first, last + 1))
    return sorted(reps)


def parameter_combinations(sweep: dict) -> list[dict]:
    """All combinations of the sweep values, e.g. {"speed": [1, 5]} -> [{"speed": 1}, {"speed": 5}].

    A single value counts as a list of one; without sweep parameters, the only combination is {}.
    """
    value_lists = [values if isinstance(values, list) else [values] for values in sweep.values()]
    return [dict(zip(sweep, values)) for values in itertools.product(*value_lists)]


def parameter_dir_name(parameters: dict) -> str:
    """Name of the parameter directory of a run: its sweep values, then Method, e.g. speed=5,Method=TWR."""
    sweep_values = [f"{key}={value}" for key, value in parameters.items() if key != "Method"]
    return ",".join([*sweep_values, f"Method={parameters['Method']}"])


def parse_parameter_dir_name(name: str) -> dict:
    """Parameters of a parameter directory name, e.g. {"speed": 5, "Method": "TWR"}."""
    parameters = {}
    for parameter in name.split(","):
        key, separator, value = parameter.partition("=")
        if not separator:
            raise ValueError(f"Not a parameter directory name: {name}")
        parameters[key.strip()] = parse_value(value.strip())
    return parameters


def parameter_key(parameters: dict) -> frozenset:
    """Key of the parameters of a run that does not depend on their order or number format (5 == 5.0)."""
    return frozenset(parameters.items())


def run_path(parent: Path, parameters: dict, rep: int) -> Path:
    """Run directory <parameter directory>/rep<N> below parent."""
    return parent / parameter_dir_name(parameters) / f"rep{rep}"


def parse_run_path(path: Path) -> tuple[dict, int]:
    """Parameters and repetition of a run directory."""
    rep = REP_DIR_PATTERN.fullmatch(path.name)
    if rep is None:
        raise ValueError(f"Not a repetition directory: {path}")
    return parse_parameter_dir_name(path.parent.name), int(rep[1])


# -----------------------------------------------------------------------------
# Raw traces
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def raw_dirs(experiment_path: Path) -> dict[frozenset, Path]:
    """Raw directories of an experiment by the parameter_key of their parameters.

    OMNeT++ escapes '_' as '#_' and ends the names with '-', e.g.
    Method=HYPERBOLIC#_SELF#_LOCALIZATION- for Method=HYPERBOLIC_SELF_LOCALIZATION.
    Cached, because the steps look up every run of the same experiment.
    """
    dirs = {}
    for raw_dir in sorted(experiment_path.iterdir()):
        if not raw_dir.is_dir():
            continue
        key = parameter_key(parse_parameter_dir_name(raw_dir.name.replace("#", "").rstrip("-")))
        if key in dirs:
            raise ValueError(f"Raw directories with the same parameters: {dirs[key].name}, {raw_dir.name}")
        dirs[key] = raw_dir
    return dirs


def trace_files(raw_dir: Path) -> dict[int, list[Path]]:
    """Trace files of a raw directory by repetition, sorted by name; raises for any other file."""
    files = {}
    for path in sorted(raw_dir.iterdir()):
        match = TRACE_NAME_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected trace file name: {path}")
        files.setdefault(int(match[2]), []).append(path)
    return files


def find_run_traces(experiment_path: Path, parameters: dict, rep: int) -> list[Path]:
    """Trace files of one run, e.g. of {"speed": 5, "Method": "TWR"} and rep 3; [] if there are none."""
    if not experiment_path.is_dir():
        return []
    raw_dir = raw_dirs(experiment_path).get(parameter_key(parameters))
    if raw_dir is None:
        return []
    return trace_files(raw_dir).get(rep, [])


def num_node_traces(paths: list[Path]) -> int:
    """Number of simulated nodes (node[i], not anchors) of a run: its node traces, also those without records."""
    return sum(TRACE_NAME_PATTERN.fullmatch(path.name)[1].lower().startswith("node") for path in paths)


def read_traces(paths: list[Path]) -> pd.DataFrame:
    """All records of the trace files of one run, with Node_Name and the node columns (add_node_columns).

    Every trace file is one block of rows, in NodeId order.
    """
    tables = []
    for path in paths:
        table = pd.read_csv(path)
        # The trace headers have a space after every comma.
        table.columns = table.columns.str.replace(" ", "", regex=False)
        table["Node_Name"] = TRACE_NAME_PATTERN.fullmatch(path.name)[1]
        tables.append(table)
    run_df = pd.concat(tables, ignore_index=True)
    add_node_columns(run_df)
    # The order of the nodes does not depend on the order of paths.
    return run_df.sort_values("NodeId", kind="stable", ignore_index=True)


def add_node_columns(run_df: pd.DataFrame) -> None:
    """Add NodeId and NodeType of the node that logged a record and whether its sender is an anchor.

    NodeIds follow the OMNeT++ nodeId parameter: anchors 1..A, then the nodes,
    where A is the number of anchor traces of the run.  Raises if the anchor
    traces are not anchor[0] to anchor[A-1] or the NodeIds do not fit the
    messages (_check_senders), e.g. because a trace file is missing.
    """
    # e.g. anchor[3] -> type anchor, index 3.
    node_types = {}
    indices = {}
    for name in run_df["Node_Name"].unique():
        match = NODE_NAME_PATTERN.fullmatch(name)
        if match is None:
            raise ValueError(f"Unsupported node name: {name!r}")
        node_types[name] = match[1].casefold()
        indices[name] = int(match[2])

    anchor_indices = []
    for name, node_type in node_types.items():
        if node_type == "anchor":
            anchor_indices.append(indices[name])
    anchor_indices.sort()
    if anchor_indices != list(range(len(anchor_indices))):
        raise ValueError(f"Anchor traces with the indices {anchor_indices}: anchor traces are missing")
    # anchor[i] -> i + 1, node[i] -> A + i + 1.
    offsets = {"anchor": 0, "node": len(anchor_indices)}
    node_ids = {name: offsets[node_type] + indices[name] + 1 for name, node_type in node_types.items()}
    run_df["NodeId"] = run_df["Node_Name"].map(node_ids).astype(int)
    run_df["NodeType"] = run_df["Node_Name"].map(node_types)
    anchor_ids = run_df.loc[run_df["NodeType"] == "anchor", "NodeId"].unique()
    # Whether the message was sent by a physical anchor, also for the records of its receivers.
    run_df["src_is_anchor"] = run_df["srcId"].isin(anchor_ids)
    _check_senders(run_df)


def _check_senders(run_df: pd.DataFrame) -> None:
    """Raise if a message logged by several nodes was not logged first by its sender (the node with NodeId srcId).

    The sender logs a message when it transmits, its receivers later.  A
    violation means a missing trace file or NodeIds that differ from the
    nodeId of the simulation.  Messages logged by one node cannot be checked.
    """
    # One group per message: the records of its sender and of all its receivers.
    messages = run_df.groupby(MESSAGE_COLUMNS)
    # Records of the sender that have the earliest timestamp of their message.
    logged_first_by_sender = (run_df["NodeId"] == run_df["srcId"]) & (
        run_df[SIM] == messages[SIM].transform("min")
    )
    shared = messages["NodeId"].transform("size") > 1
    # A message passes if one of its records is such a record of the sender.
    checked = run_df.loc[shared, MESSAGE_COLUMNS].assign(sender_first=logged_first_by_sender[shared])
    sender_first = checked.groupby(MESSAGE_COLUMNS)["sender_first"].any()
    if not sender_first.all():
        # Header of the first failing message.
        example = ", ".join(f"{c}={v}" for c, v in zip(MESSAGE_COLUMNS, sender_first.index[~sender_first][0]))
        raise ValueError(
            f"{(~sender_first).sum()} of {len(sender_first)} messages were not logged first by their sender "
            f"(e.g. {example}): a trace file is missing or the NodeIds differ from the nodeId of the simulation"
        )
