"""Evaluation steps (post-processing, statistics, figures) for one or more configs.

Merges every config into configs/defaults.yml and runs the scripts with the
resulting options; usually called through the Makefile (see README.md).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import extract_statistics
from config import ROOT, load_config, to_cli_args
from runs import (
    EVENTS_FILE_NAME, POSTPROCESSED_DIR_NAME, RUN_METADATA_FILE_NAME, parameter_combinations, parameter_dir_name,
    parameter_key, parse_reps, raw_dirs, run_path, trace_files,
)

RESULTS_ROOT = ROOT / "results"
# Relative to ROOT; the jobs run with ROOT as working directory.
POSTPROCESS_SCRIPT = "src/post_processing.py"
EXTRACT_SCRIPT = "src/extract_statistics.py"
PLOT_SCRIPT = "src/plot.py"

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
# Runs and directories of a config
# -----------------------------------------------------------------------------

def results_dir(config: dict) -> Path:
    """Results directory of a config: results/<config name>."""
    return RESULTS_ROOT / config["name"]


def log_dir(config: dict, run_id: str) -> Path:
    """Logs of one pipeline run next to the results: <results>/logs/<run_id>."""
    return results_dir(config) / "logs" / run_id


def evaluation_dir(config: dict, extract_args: list[str]) -> Path:
    """Statistics and figures: <results>/<estimator>/<filter_profile>, as written by the extraction.

    The estimator is the tdoa_loss of the post-processed runs, as in the
    extraction; without runs (dry run before the post-processing), the one of
    the config.
    """
    if existing_run_dirs(config):
        estimator = extract_statistics.tdoa_estimator(extract_statistics.find_runs(results_dir(config)))
    else:
        estimator = config["postprocess"]["tdoa_loss"]
    # [2:] drops interpreter and script, leaving the options of the extraction.
    profile = extract_statistics.filter_profile(extract_command(config, extract_args)[2:])
    return results_dir(config) / estimator / profile


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------

class Job(NamedTuple):
    """One post-processing job and the run directories it writes."""
    name: str
    command: list[str]
    run_dirs: list[Path]


def raw_reps_by_parameters(data_dir: Path) -> dict[frozenset, set[int]]:
    """Reps with raw traces by the runs.parameter_key of every raw directory."""
    return {key: set(trace_files(raw_dir)) for key, raw_dir in raw_dirs(data_dir).items()}


def postprocess_jobs(
    config: dict,
    extra_args: list[str],
    raw_reps: dict[frozenset, set[int]] | None = None,
) -> tuple[list[Job], list[str]]:
    """One job per method, sweep combination and (with split_reps) rep.

    raw_reps (see raw_reps_by_parameters) restricts the jobs to reps that
    have raw data.  Returns the jobs and the runs without raw data.
    """
    # Rep specs as in the config, e.g. ["0-19"] or [1, "4-6"].
    rep_specs = config["reps"] if isinstance(config["reps"], list) else [config["reps"]]
    config_reps = parse_reps(rep_specs)
    split_reps = bool(config.get("split_reps"))
    output_dir = results_dir(config) / POSTPROCESSED_DIR_NAME
    jobs, missing_raw = [], []
    for method in config["methods"]:
        for combination in parameter_combinations(config["sweep"]):
            parameters = {**combination, "Method": method}
            parameter_dir = parameter_dir_name(parameters)
            # --sweep_parameters entries, e.g. speed:2
            sweep_args = [f"{name}:{value}" for name, value in combination.items()]
            reps = config_reps
            if raw_reps is not None:
                available = raw_reps.get(parameter_key(parameters), set())
                missing_raw += [f"{parameter_dir}/rep{rep}" for rep in config_reps if rep not in available]
                # Without split_reps, the one job keeps all reps; post_processing.py skips runs without traces.
                if split_reps:
                    reps = [rep for rep in config_reps if rep in available]
            # With split_reps every rep is its own job; otherwise one job runs all reps.
            reps_per_job = [[rep] for rep in reps] if split_reps else [reps]
            for job_reps in reps_per_job:
                # The single job passes the specs unchanged (e.g. "0-19"); --rep_limit expands them.
                rep_limit = [str(job_reps[0])] if split_reps else list(map(str, rep_specs))
                rep_suffix = [f"rep{job_reps[0]}"] if split_reps else []
                # '.' and ':' have a special meaning in tmux targets.
                window_name = (
                    "_".join([method, *sweep_args, *rep_suffix])
                    .replace(".", "_")
                    .replace(":", "_")
                )
                command = [
                    sys.executable, POSTPROCESS_SCRIPT,
                    "--method", method,
                    "--experiment_path", str(config["data"]),
                    "--rep_limit", *rep_limit,
                    *(["--sweep_parameters", *sweep_args] if sweep_args else []),
                    "--results_dir", str(output_dir),
                    *to_cli_args(config["postprocess"]),
                    *extra_args,
                ]
                # Run directories the job writes; --resume (job_is_done) and the extract step check them.
                run_dirs = [run_path(output_dir, parameters, rep) for rep in job_reps]
                jobs.append(Job(window_name, command, run_dirs))
    return jobs, missing_raw


def job_is_done(job: Job) -> bool:
    """True if all runs of the job exist and were written by exactly this command."""
    for run_dir in job.run_dirs:
        metadata_path = run_dir / RUN_METADATA_FILE_NAME
        if not (run_dir / EVENTS_FILE_NAME).exists() or not metadata_path.exists():
            return False
        recorded_command = json.loads(metadata_path.read_text()).get("command")
        # The post-processing records sys.argv, i.e. the command without the interpreter.
        if recorded_command != job.command[1:]:
            return False
    return True


def existing_run_dirs(config: dict) -> set[Path]:
    """Post-processed runs in <results>/postprocessed/."""
    return {
        path.parent
        for path in (results_dir(config) / POSTPROCESSED_DIR_NAME).glob(f"*/rep*/{EVENTS_FILE_NAME}")
    }


def warn(message: str, items: list) -> None:
    """Print a warning with the number of items and a few examples."""
    if items:
        examples = ", ".join(map(str, sorted(items)[:MAX_LISTED]))
        more = f", ... (+{len(items) - MAX_LISTED})" if len(items) > MAX_LISTED else ""
        print(f"WARNING: {len(items)} {message}: {examples}{more}", flush=True)


def extract_command(config: dict, extra_args: list[str]) -> list[str]:
    """Command of the extraction step: options of the config section extract, then extra_args."""
    return [
        sys.executable, EXTRACT_SCRIPT,
        "--config-path", str(results_dir(config)),
        *to_cli_args(config["extract"]),
        *extra_args,
    ]


def plot_command(config: dict, extra_args: list[str], extract_args: list[str]) -> list[str]:
    """Command of the plot step for the evaluation directory of the extraction with extract_args."""
    return [
        sys.executable, PLOT_SCRIPT,
        str(evaluation_dir(config, extract_args)),
        "--config", config["path"],
        *extra_args,
    ]


def format_command(command: list[str]) -> str:
    """Command as a shell-quoted string."""
    return shlex.join(command)


# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------

def run_command(command: list[str]) -> None:
    """Print and run a command in the foreground; raises if it fails."""
    print(f"\n$ {format_command(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def tmux_window_ids() -> set[str]:
    """IDs of all existing tmux windows, e.g. {'@3', '@7'}."""
    output = subprocess.run(
        ["tmux", "list-windows", "-a", "-F", "#{window_id}"],
        capture_output=True, text=True, check=True,
    ).stdout
    return set(output.split())


def run_jobs_in_tmux_windows(jobs: list[Job], max_parallel_jobs: int, job_log_dir: Path) -> None:
    """Run every job in its own window of the current tmux session, named after the job.

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
        # Start jobs until max_parallel_jobs are running.
        while pending and len(running) < max_parallel_jobs:
            job = pending.pop(0)
            window_name, command = job.name, job.command
            log_file = job_log_dir / f"{window_name}.log"
            exit_file = job_log_dir / f"{window_name}.exit"
            print(f"start  {window_name}: {format_command(command)}", flush=True)
            # tee shows the output in the window and logs it; PIPESTATUS[0] is the exit code of the job, not of tee.
            script = (
                f"cd {shlex.quote(str(ROOT))} && export PYTHONUNBUFFERED=1 && "
                f"{format_command(command)} 2>&1 | tee {shlex.quote(str(log_file))}; "
                f"echo ${{PIPESTATUS[0]}} > {shlex.quote(str(exit_file))}"
            )
            # -d: in the background; -P -F: print the ID of the new window.
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
            # An exit file that is still being written is empty: the job counts as running.
            exit_text = exit_file.read_text().strip() if exit_file.exists() else ""
            if exit_text:
                exit_code = int(exit_text)
                status = "OK" if exit_code == 0 else f"FAILED (exit {exit_code})"
            elif window_id not in open_windows:
                exit_code = None
                status = "FAILED (window closed without exit code)"
            else:
                # Still running.
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


def run_postprocess(config: dict, args: argparse.Namespace) -> None:
    """Post-process the runs of a config in parallel tmux windows (or with --dry-run print the commands)."""
    jobs, missing_raw = postprocess_jobs(
        config, args.postprocess_args, raw_reps_by_parameters(ROOT / config["data"])
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


def report_unexpected_runs(config: dict) -> None:
    """Warn about post-processed runs outside the config and runs of the config that are not post-processed.

    Runs outside the config are kept and included on purpose (e.g. a config
    split over several machines); they are only reported.
    """
    # All runs of the config, without the check for raw traces.
    expected = {run_dir for job in postprocess_jobs(config, [])[0] for run_dir in job.run_dirs}
    existing = existing_run_dirs(config)
    results = results_dir(config)
    warn(
        "post-processed runs are not part of the config but are included",
        [path.relative_to(results) for path in existing - expected],
    )
    warn("runs of the config are not post-processed", [path.relative_to(results) for path in expected - existing])


def run_config(config: dict, steps: tuple[str, ...], args: argparse.Namespace) -> None:
    """Run (or with --dry-run print) the steps for one config."""
    print(f"\n=== {config['name']}: {' -> '.join(steps)} (results: {results_dir(config)})", flush=True)
    if config["sources"]:
        print("Plot-only config (sources): postprocess and extract are skipped", flush=True)
        steps = tuple(step for step in steps if step == "plots")
    for step in steps:
        if step == "postprocess":
            run_postprocess(config, args)
            continue
        if step == "extract":
            report_unexpected_runs(config)
            command = extract_command(config, args.extract_args)
        else:
            # The plot step needs the extraction options to find the evaluation directory.
            command = plot_command(config, args.plot_args, args.extract_args)
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
    # The same run ID, so the session writes to the log directory printed below.
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
    """Command-line arguments."""
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
    """Load the configs and run the step for each of them, in a new tmux session if needed."""
    args = parse_args(argv)
    configs = []
    for config_path in args.configs:
        if not config_path.is_file():
            raise SystemExit(f"Evaluation config {config_path} does not exist (see README)")
        config = load_config(config_path)
        # Command-line overrides (make JOBS=..., REPS=..., METHODS=...) replace the config values.
        for key in ("jobs", "reps", "methods"):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        data_dir = (ROOT / config["data"]).resolve()
        if "postprocess" in STEPS[args.step] and not config["sources"] and not data_dir.is_dir():
            raise SystemExit(f"{config['name']}: no simulation traces in {data_dir}, run the simulation first")
        configs.append(config)

    # Outside tmux, detached steps start this script again in a new tmux session and return.
    in_tmux = bool(os.environ.get("TMUX"))
    if args.step in DETACHED_STEPS and not (args.foreground or args.dry_run or in_tmux):
        start_tmux_session(args, configs)
        return

    for config in configs:
        run_config(config, STEPS[args.step], args)
    print("\nDone.")


if __name__ == "__main__":
    main()
