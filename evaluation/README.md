# Evaluation

Post-processing of the simulation traces, statistics and figures.
Installation and the complete workflow are described in the [top-level README](../README.md).

| Step          | Script                         | Output (in `results/<Config>/`) |
| ------------- | ------------------------------ | ------------------------------- |
| `postprocess` | `src/post_processing.py`       | `postprocessed/`: localization events per run (t-TWR, TWR and TDoA position estimates and their errors) |
| `extract`     | `src/extract_statistics.py`    | `<estimator>/<filter profile>/`: filtered events and statistics per run and aggregated over the repetitions |
| `plots`       | `src/plot.py`                  | `<estimator>/<filter profile>/figures/<Config>_<figure>.<format>` |

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
`postprocess` and `extract` are passed as command-line options to the
respective script, `plots` is read by `src/plot.py`. A minimal config for a
parameter study:

```yaml
sweep:
  speed: [5, 10, 25, 50]       # iteration variable of omnetpp.ini and its values
plots:
  figures:
    coverage:                  # figure name: file <Config>_coverage.pdf
      type: line
      metric: mean_coverage
```

A plot-only config draws methods of other configs side by side from their
results (evaluate the sources first; they need the same repetitions and sweep
parameter), e.g. with a hypothetical config `AnchorDensity-Variant`:

```yaml
sources:                       # <Config>: methods, or {method: name in the figures}
  AnchorDensity: [CENTRALIZED, TWR]
  AnchorDensity-Variant: {CENTRALIZED: CENTRALIZED_VARIANT}
```

## Figures

Every entry of `plots.figures` is one figure. Its name is free, so the same
plot type can be drawn several times with other options. The options of a
figure are, by increasing priority: the defaults of its plot type,
`plots.defaults` of `configs/defaults.yml` and of the config, and the figure
entry.

| Plot type       | Shows                                                       | Own options          |
| --------------- | ----------------------------------------------------------- | -------------------- |
| `line`          | statistic of a metric over the repetitions per sweep value  | `metric`, `errorbar` |
| `ecdf`          | ECDF of all samples of `error`, `availability` or `coverage` | `metric`, `layout`   |
| `timeseries`    | coverage over time                                          | `smoothing`, `layout` |
| `opportunities` | outcome of the t-Anchor opportunities                       |                      |

`layout` draws the sweep values in one axes (`combined`), in one axes each
(`facets`) or in one figure each (`separate`). The plot types and their
options are documented in [src/plot_types.py](src/plot_types.py); the options
of all figures (methods, size, axes), the display names and the method styles
in [configs/defaults.yml](configs/defaults.yml). Figure names follow
`<quantity>[_<view>][_<variant>]`, e.g. `coverage`, `coverage_time`,
`error_ecdf`, `coverage_std`.

- **New method** (value of `Method` in `omnetpp.ini`): its localization steps
  in `METHODS` of `src/localization.py`; a label in `plots.labels` and a style in
  `plots.styles` of `configs/defaults.yml` (otherwise the next free style).
- **New metric**: one `Statistic` in `run_statistics` of `src/metrics.py`; it
  is aggregated over the repetitions and drawn with `type: line`.
- **New event filter**: a field of `EventFilter` in `src/extract_statistics.py`
  (command-line option, rule in `apply`, part of the profile name).
- **New plot type**: a class in `src/plot_types.py` (see its module docstring).

Single figures, e.g. while adjusting them:
`python3 src/plot.py <evaluation dir> --config configs/<Config>.yml --figures <name>`
or `make plots-<Config> PLOT_ARGS="--figures <name>"`.

## Metrics

All statistics are computed on the filtered localization events (DOP, residual
MAD, see `extract_statistics.py --help`). The simulation time is divided into
windows of 0.04 s (25 Hz):

- **Coverage** C(t): fraction of all simulated nodes that were localized in a
  window; the mean coverage is its average over all windows.
- **Availability** A_i: fraction of the windows in which node i was localized;
  `availability_q10` is the 10 % quantile over the nodes of a run.
- **Opportunities**: t-Anchor slots of HyRASeL that were used (the new t-Anchor
  got a t-TWR position), missed, or had no eligible node (heard all anchors and
  not busy as t-Anchor of an earlier round).

## Code

`src/pipeline.py` starts the scripts of the three steps as separate processes;
each of them also runs on its own.

| Module                      | Role |
| --------------------------- | ---- |
| `src/pipeline.py`           | runs the steps of one or more configs (commands, tmux jobs, resume, dry run) |
| `src/config.py`             | loads a config merged into `configs/defaults.yml`; its sections become command-line options |
| `src/runs.py`               | names, directories and raw traces of the runs, shared by all steps |
| `src/post_processing.py`    | `postprocess`: reads the traces of every run, localizes, writes the events |
| `src/localization.py`       | TWR, t-TWR and TDoA localization steps of every method (`METHODS`) |
| `src/solvers.py`            | trilateration and TDoA least squares |
| `src/extract_statistics.py` | `extract`: event filter (`EventFilter`), outputs per run and aggregated |
| `src/metrics.py`            | coverage, availability and the statistics of a run |
| `src/opportunities.py`      | t-Anchor opportunities of CENTRALIZED and DISTRIBUTED from the raw traces and the t-TWR positions |
| `src/plot.py`               | `plots`: draws every figure of `plots.figures` |
| `src/plot_types.py`         | what every plot type shows |
| `src/plot_style.py`         | look of all figures: fonts, styles, legend, saving |

## Tests

```sh
python3 -m unittest discover -s tests
```
