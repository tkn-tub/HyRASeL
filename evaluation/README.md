# Evaluation

Post-processing of the simulation traces, statistics and figures.
Installation and the complete workflow are described in the [top-level README](../README.md).

| Step          | Script                         | Output (in `results/<Config>/`) |
| ------------- | ------------------------------ | ------------------------------- |
| `postprocess` | `src/post_processing.py`       | `postprocessed/`: localization events per run (v-TWR, TWR and TDoA position estimates and their errors) |
| `extract`     | `src/extract_statistics.py`    | `<estimator>/<filter profile>/`: filtered events and statistics per run and aggregated over the repetitions |
| `plots`       | `src/plot_parameter_sweeps.py` | `<estimator>/<filter profile>/figures/` |

`src/pipeline.py` runs the steps for one or more configs; the Makefile wraps it.
The options of every script are listed with `python3 <script> --help`.

## Usage

```sh
make full-Speed                          # postprocess -> extract -> plots for configs/Speed.yml
make plots-Speed                         # only one step
make full-all                            # all configs
make full-Speed JOBS=8 REPS=0-4 DRY_RUN=1
```

`make help` and the header of the Makefile list all steps and options.

The post-processing requires [tmux](https://github.com/tmux/tmux): every job
runs in its own tmux window, at most `jobs` at the same time. If `make` is not
called from tmux, it starts a tmux session and prints the command to attach to
it. Logs are written to `results/<Config>/logs/`. `RESUME=1` skips jobs whose
results already exist and were created with the same command.

## Configs

`configs/<Config>.yml` evaluates the traces of the simulation config with the
same name (`../simulation/simulations/results/traces/<Config>/`, see `data_root`
in `configs/defaults.yml`). A config only contains the entries that differ from
`configs/defaults.yml`, which also documents all keys. The sections
`postprocess`, `extract` and `plots` are passed as command-line options to the
respective script. A minimal config for a parameter study:

```yaml
sweep:
  speed: [5, 10, 25, 50]       # iteration variable of omnetpp.ini and its values
plots:
  plots: [error, coverage]     # see plot_parameter_sweeps.py --help
```

## Metrics

All statistics are computed on the filtered localization events (DOP, residual
MAD, see `extract_statistics.py --help`). The simulation time is divided into
windows of 0.04 s (25 Hz):

- **Coverage** C(t): fraction of all simulated nodes that were localized in a
  window; the mean coverage is its average over all windows.
- **Availability** A_i: fraction of the windows in which node i was localized;
  `availability_q10` is the 10 % quantile over the nodes of a run.
- **Opportunities**: v-Anchor slots of HyRASeL that were used, missed, or had
  no eligible node.

## Tests

```sh
python3 -m unittest discover -s tests
```
