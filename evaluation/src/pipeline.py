"""Evaluation steps (post-processing, statistics, figures) for one or more configs.

Merges every config into configs/defaults.yml and runs the scripts with the
resulting options; usually called through the Makefile (see README.md).
"""

from __future__ import annotations

import argparse
import itertools
import os
import shlex
import subprocess
import sys
import re
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import yaml

import extract_statistics

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results"
POSTPROCESSED_DIR_NAME = "postprocessed"
DEFAULTS_PATH = ROOT / "configs" / "defaults.yml"
# Relative to ROOT; the jobs run with ROOT as working directory.
POSTPROCESS_SCRIPT = "src/post_processing.py"
EXTRACT_SCRIPT = "src/extract_statistics.py"
PLOT_SCRIPT = "src/plot_parameter_sweeps.py"

STEPS = {
    "postprocess": ("postprocess",),
    "extract": ("extract",),
    "plots": ("plots",),
    "analyze": ("extract", "plots"),
    "full": ("postprocess", "extract", "plots"),
}
# Steps that are started in their own tmux session (unless --foreground).
DETACHED_STEPS = {"postprocess", "extract", "analyze", "full"}
POLL_INTERVAL = 2.0  # [s]
# At most this many examples are listed in warnings.
MAX_LISTED = 5


# -----------------------------------------------------------------------------
# Config handling
# -----------------------------------------------------------------------------

def deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base; nested dicts are merged, everything else replaced."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path, defaults_path: Path = DEFAULTS_PATH) -> dict:
    """Load a config merged into the defaults; its name is the file name."""
    defaults = yaml.safe_load(defaults_path.read_text()) or {}
    config = yaml.safe_load(config_path.read_text()) or {}
    merged = deep_merge(defaults, config)
    merged["name"] = config_path.stem
    # Raw traces of the simulation config with the same name
    if "data" not in merged:
        merged["data"] = str(Path(merged["data_root"]) / merged["name"])
    for section in ("postprocess", "extract", "plots"):
        merged[section] = merged.get(section) or {}
    merged["sweep"] = merged.get("sweep") or {}
    return merged


def to_cli_args(options: dict) -> list[str]:
    """Translate a config section into command-line arguments."""
    args = []
    for key, value in options.items():
        if value is None or value is False:
            continue
        if value is True:
            args.append(f"--{key}")
        elif isinstance(value, dict):
            entries = [f"{name}={entry}" for name, entry in value.items() if entry is not None]
            if entries:
                args.extend([f"--{key}", *entries])
        elif isinstance(value, list):
            args.extend([f"--{key}", *map(str, value)])
        else:
            args.extend([f"--{key}", str(value)])
    return args


def sweep_combinations(sweep: dict) -> list[list[str]]:
    """All sweep-value combinations as --sweep_parameters entries, e.g. [['speed:2'], ...]."""
    if not sweep:
        return [[]]
    names = list(sweep)
    value_lists = [values if isinstance(values, list) else [values] for values in sweep.values()]
    return [
        [f"{name}:{value}" for name, value in zip(names, values)]
        for values in itertools.product(*value_lists)
    ]


def results_dir(config: dict) -> Path:
    """Results directory of a config: results/<config name>."""
    return RESULTS_ROOT / config["name"]


def log_dir(config: dict, run_id: str) -> Path:
    """Logs of one pipeline run next to the results: <results>/logs/<run_id>."""
    return results_dir(config) / "logs" / run_id


def evaluation_dir(config: dict, extract_args: list[str]) -> Path:
    """Statistics and figures: <results>/<estimator>/<filter_profile>, as written by the extraction."""
    return results_dir(config) / extract_statistics.evaluation_subdir(
        extract_command(config, extract_args)[2:], config["postprocess"]
    )


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------

def expand_reps(reps: str | int | list) -> list[int]:
    """Expand repetition specs like "0-10", "3,5,8-15" or [1, "4-6"] to single reps."""
    expanded = []
    for spec in reps if isinstance(reps, list) else [reps]:
        for part in str(spec).split(","):
            start, _, end = part.strip().partition("-")
            expanded.extend(range(int(start), int(end or start) + 1))
    return sorted(set(expanded))


class Job(NamedTuple):
    """One post-processing job and the run directories it writes."""
    name: str
    command: list[str]
    run_dirs: list[Path]


def parameter_dir_name(method: str, combination: list[str]) -> str:
    """Run directory name as written by the post-processing, e.g. speed=2,Method=TWR."""
    return ",".join(
        [entry.replace(":", "=", 1) for entry in combination] + [f"Method={method}"]
    )


def raw_reps_by_parameter_dir(data_dir: Path) -> dict[str, set[int]]:
    """Reps with raw traces per parameter directory, e.g. {'speed=2,Method=TWR': {0, 1}}.

    Raw directories carry '#' escapes and a trailing '-'
    (Method=HYPERBOLIC#_SELF#_LOCALIZATION-).
    """
    reps = {}
    for raw_dir in data_dir.iterdir():
        if raw_dir.is_dir():
            reps[raw_dir.name.replace("#", "").rstrip("-")] = {
                int(match.group(1))
                for trace in raw_dir.glob("*_rep*_log.csv")
                if (match := re.search(r"_rep(\d+)_log\.csv$", trace.name))
            }
    return reps


def postprocess_jobs(
    config: dict,
    extra_args: list[str],
    raw_reps: dict[str, set[int]] | None = None,
) -> tuple[list[Job], list[str]]:
    """One job per method, sweep combination and (with split_reps) rep.

    raw_reps (see raw_reps_by_parameter_dir) restricts the jobs to reps that
    have raw data.  Returns the jobs and the runs without raw data.
    """
    reps = config["reps"] if isinstance(config["reps"], list) else [config["reps"]]
    split_reps = bool(config.get("split_reps"))
    output_dir = results_dir(config) / POSTPROCESSED_DIR_NAME
    jobs, missing_raw = [], []
    for method in config["methods"]:
        for combination in sweep_combinations(config["sweep"]):
            parameter_dir = parameter_dir_name(method, combination)
            config_reps = expand_reps(reps)
            if raw_reps is not None:
                available = raw_reps.get(parameter_dir, set())
                missing_raw += [f"{parameter_dir}/rep{rep}" for rep in config_reps if rep not in available]
                if split_reps:
                    config_reps = [rep for rep in config_reps if rep in available]
            # With split_reps every rep is its own job; otherwise one job runs all reps.
            rep_groups = (
                [[str(rep)] for rep in config_reps]
                if split_reps
                else [list(map(str, reps))]
            )
            for rep_group in rep_groups:
                rep_suffix = [f"rep{rep_group[0]}"] if split_reps else []
                # '.' and ':' have a special meaning in tmux targets.
                window_name = (
                    "_".join([method, *combination, *rep_suffix])
                    .replace(".", "_")
                    .replace(":", "_")
                )
                command = [
                    sys.executable, POSTPROCESS_SCRIPT,
                    "--method", method,
                    "--experiment_path", str(config["data"]),
                    "--rep_limit", *rep_group,
                    *(["--sweep_parameters", *combination] if combination else []),
                    "--results_dir", str(output_dir),
                    *to_cli_args(config["postprocess"]),
                    *extra_args,
                ]
                run_dirs = [
                    output_dir / parameter_dir / f"rep{rep}"
                    for rep in (expand_reps(rep_group) if split_reps else config_reps)
                ]
                jobs.append(Job(window_name, command, run_dirs))
    return jobs, missing_raw


def job_is_done(job: Job) -> bool:
    """True if all runs of the job exist and were written by exactly this command."""
    for run_dir in job.run_dirs:
        metadata_path = run_dir / "run_metadata.json"
        if not (run_dir / "localization_events.csv").exists() or not metadata_path.exists():
            return False
        recorded_command = yaml.safe_load(metadata_path.read_text()).get("command")
        # The post-processing records sys.argv, i.e. the command without the interpreter.
        if recorded_command != job.command[1:]:
            return False
    return True


def existing_run_dirs(config: dict) -> set[Path]:
    """Post-processed runs in <results>/postprocessed/."""
    return {
        path.parent
        for path in (results_dir(config) / POSTPROCESSED_DIR_NAME).glob("*/rep*/localization_events.csv")
    }


def warn(message: str, items: list) -> None:
    """Print a warning with the number of items and a few examples."""
    if items:
        examples = ", ".join(map(str, sorted(items)[:MAX_LISTED]))
        more = f", ... (+{len(items) - MAX_LISTED})" if len(items) > MAX_LISTED else ""
        print(f"WARNING: {len(items)} {message}: {examples}{more}", flush=True)


def extract_command(config: dict, extra_args: list[str]) -> list[str]:
    return [
        sys.executable, EXTRACT_SCRIPT,
        "--config-path", str(results_dir(config)),
        *to_cli_args(config["extract"]),
        *extra_args,
    ]


def plot_command(config: dict, extra_args: list[str], extract_args: list[str]) -> list[str]:
    return [
        sys.executable, PLOT_SCRIPT,
        str(evaluation_dir(config, extract_args)),
        *to_cli_args(config["plots"]),
        *extra_args,
    ]


def format_command(command: list[str]) -> str:
    return shlex.join(command)


# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------

def run_command(command: list[str]) -> None:
    print(f"\n$ {format_command(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def tmux_window_ids() -> set[str]:
    """IDs of all existing tmux windows, e.g. {'@3', '@7'}."""
    output = subprocess.run(
        ["tmux", "list-windows", "-a", "-F", "#{window_id}"],
        capture_output=True, text=True, check=True,
    ).stdout
    return set(output.split())


def run_jobs_in_tmux_windows(
    jobs: list[tuple[str, list[str]]],
    max_parallel_jobs: int,
    job_log_dir: Path,
) -> None:
    """Run every job (name, command, ...) in its own window of the current tmux session.

    Each job writes its output to <job_log_dir>/<window>.log and its exit code to
    <job_log_dir>/<window>.exit.  A job whose window disappears without an exit
    code (closed from outside) counts as failed.  Raises SystemExit if any job
    failed.
    """
    if not os.environ.get("TMUX"):
        raise SystemExit("postprocess must run inside tmux (start it without --foreground).")

    job_log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    running: dict[str, tuple[Path, str]] = {}
    failed: list[str] = []
    num_finished = 0

    while pending or running:
        while pending and len(running) < max_parallel_jobs:
            window_name, command = pending.pop(0)[:2]
            log_file = job_log_dir / f"{window_name}.log"
            exit_file = job_log_dir / f"{window_name}.exit"
            print(f"start  {window_name}: {format_command(command)}", flush=True)
            script = (
                f"cd {shlex.quote(str(ROOT))} && export PYTHONUNBUFFERED=1 && "
                f"{format_command(command)} 2>&1 | tee {shlex.quote(str(log_file))}; "
                f"echo ${{PIPESTATUS[0]}} > {shlex.quote(str(exit_file))}"
            )
            window_id = subprocess.run(
                [
                    "tmux", "new-window", "-d", "-P", "-F", "#{window_id}",
                    "-n", window_name, f"bash -c {shlex.quote(script)}",
                ],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            running[window_name] = (exit_file, window_id)

        time.sleep(POLL_INTERVAL)
        # List the windows before reading the exit files: a job writes its exit
        # file before its window closes, so a finished job is never mistaken
        # for a closed one.
        open_windows = tmux_window_ids()
        for window_name, (exit_file, window_id) in list(running.items()):
            exit_text = exit_file.read_text().strip() if exit_file.exists() else ""
            if exit_text:
                exit_code = int(exit_text)
                status = "OK" if exit_code == 0 else f"FAILED (exit {exit_code})"
            elif window_id not in open_windows:
                exit_code = None
                status = "FAILED (window closed without exit code)"
            else:
                continue
            del running[window_name]
            num_finished += 1
            print(f"[{num_finished}/{len(jobs)}] {status}  {window_name}", flush=True)
            if exit_code != 0:
                failed.append(window_name)

    if failed:
        raise SystemExit(
            f"{len(failed)} post-processing job(s) failed, see logs in {job_log_dir}: "
            + ", ".join(failed)
        )


def run_config(config: dict, steps: tuple[str, ...], args: argparse.Namespace) -> None:
    print(f"\n=== {config['name']}: {' -> '.join(steps)} (results: {results_dir(config)})", flush=True)
    for step in steps:
        if step == "postprocess":
            jobs, missing_raw = postprocess_jobs(
                config, args.postprocess_args, raw_reps_by_parameter_dir(ROOT / config["data"])
            )
            warn("runs of the config have no raw data and are skipped", missing_raw)
            if args.resume:
                done = [job for job in jobs if job_is_done(job)]
                jobs = [job for job in jobs if job not in done]
                print(f"--resume: {len(done)} job(s) already done, {len(jobs)} to run", flush=True)
            if args.dry_run:
                for job in jobs:
                    print(format_command(job.command))
            else:
                run_jobs_in_tmux_windows(jobs, int(config["jobs"]), log_dir(config, args.run_id))
        else:
            if step == "extract":
                # Runs outside the config are kept and included on purpose
                # (e.g. a config split over several machines); only report them.
                expected = {
                    run_dir
                    for job in postprocess_jobs(config, [])[0]
                    for run_dir in job.run_dirs
                }
                existing = existing_run_dirs(config)
                relative = lambda paths: [path.relative_to(results_dir(config)) for path in paths]
                warn(
                    "post-processed runs are not part of the config but are included",
                    relative(existing - expected),
                )
                warn("runs of the config are not post-processed", relative(expected - existing))
            command = (
                extract_command(config, args.extract_args)
                if step == "extract"
                else plot_command(config, args.plot_args, args.extract_args)
            )
            if args.dry_run:
                print(format_command(command))
            else:
                run_command(command)


def start_tmux_session(args: argparse.Namespace, configs: list[dict]) -> None:
    """Re-run this script with the same arguments inside a new tmux session.

    pipeline.log goes to the logs of the config; for several configs (make
    <step>-all) to logs/hyrasel-<step>-all/<run_id>/, while the job logs are
    still written per config.
    """
    session = (
        f"hyrasel-{configs[0]['name']}"
        if len(configs) == 1
        else f"hyrasel-{args.step}-all"
    )
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        raise SystemExit(f"tmux session already exists: {session} (attach: tmux attach -t {session})")

    session_log_dir = (
        log_dir(configs[0], args.run_id)
        if len(configs) == 1
        else ROOT / "logs" / session / args.run_id
    )
    session_log_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--run-id", args.run_id]
    # On success the session closes (the output stays in pipeline.log); on
    # failure the pipeline window stays open until Enter is pressed.
    script = (
        f"cd {shlex.quote(str(ROOT))} && export PYTHONUNBUFFERED=1 && "
        f"{format_command(command)} 2>&1 | tee {shlex.quote(str(session_log_dir / 'pipeline.log'))}; "
        "exit_code=${PIPESTATUS[0]}; "
        "if [ $exit_code -ne 0 ]; then "
        "echo; echo \"Pipeline FAILED with exit code $exit_code. Press Enter to close.\"; read; fi"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "-n", "pipeline", f"bash -c {shlex.quote(script)}"],
        check=True,
    )
    print(f"Started tmux session: {session}")
    print(f"  attach: tmux attach -t {session}")
    print(f"  logs:   {session_log_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=STEPS, help="analyze: extract + plots, full: all three")
    parser.add_argument("configs", nargs="+", type=Path, help="YAML config files")
    parser.add_argument("--jobs", type=int, help="Override: max. parallel post-processing jobs")
    parser.add_argument("--reps", help="Override: repetitions, e.g. 0-10")
    parser.add_argument("--methods", nargs="+", help="Override: methods")
    parser.add_argument("--postprocess-args", type=shlex.split, default=[], help="Extra post-processing arguments")
    parser.add_argument("--extract-args", type=shlex.split, default=[], help="Extra extraction arguments")
    parser.add_argument("--plot-args", type=shlex.split, default=[], help="Extra plotting arguments")
    parser.add_argument("--foreground", action="store_true", help="Do not start a tmux session")
    parser.add_argument("--dry-run", action="store_true", help="Only print the commands")
    parser.add_argument("--resume", action="store_true", help="Skip post-processing jobs that are already done")
    # Shared timestamp of one pipeline run (names the log directories).
    parser.add_argument(
        "--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"), help=argparse.SUPPRESS
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configs = []
    for config_path in args.configs:
        if not config_path.is_file():
            raise SystemExit(f"Evaluation config {config_path} does not exist (see README)")
        config = load_config(config_path)
        for key in ("jobs", "reps", "methods"):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        data_dir = (ROOT / config["data"]).resolve()
        if "postprocess" in STEPS[args.step] and not data_dir.is_dir():
            raise SystemExit(f"{config['name']}: no simulation traces in {data_dir}, run the simulation first")
        configs.append(config)

    in_tmux = bool(os.environ.get("TMUX"))
    if args.step in DETACHED_STEPS and not (args.foreground or args.dry_run or in_tmux):
        start_tmux_session(args, configs)
        return

    for config in configs:
        run_config(config, STEPS[args.step], args)
    print("\nDone.")


if __name__ == "__main__":
    main()
