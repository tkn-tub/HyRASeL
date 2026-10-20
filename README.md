# HyRASeL

Simulation and evaluation code for the paper

> **HyRASeL: Hybrid UWB Two-Way-Ranging and Asynchronous Hyperbolic Self-Localization**
> Simon Schmitz-Heinen, Anatolij Zubow, Tehmina Bibi, Falko Dressler
> Submitted to IEEE/IFIP WONS 2027

This repository contains the subset of our code base that is required to
reproduce the results of the paper.

## Structure

```text
HyRASeL/
├── simulation/                       OMNeT++ project (see simulation/README.md)
│   ├── src/                          Source code
│   └── simulations/
│       ├── omnetpp.ini               Simulation configs
│       └── results/traces/<Config>/  Output: simulation traces (CSV)
├── evaluation/                       Python: post-processing, statistics, figures (see evaluation/README.md)
│   ├── configs/<Config>.yml          Evaluation configs
│   ├── src/                          Scripts
│   ├── tests/                        Unit tests
│   └── results/<Config>/             Output: evaluation results
├── figures/                          Output: figures of the paper (make paper)
├── logs/                             Output: logs
├── Makefile                          Runs everything in a Docker container
├── Dockerfile                        Environment: OMNeT++ 6.2.0, INET 4.5.4, Python packages, tmux
└── container.sh                      Runs make in tmux inside the container (used by the Makefile)
```

## Reproducing the Results with Docker

Requires only [Docker](https://docs.docker.com/engine/install/) (usable
without `sudo`, i.e. as a member of the group `docker`) and `make`, and about
20 GB of free disk space for the simulation traces. The Docker image is built
automatically on first use (about 10 minutes). Tested on Linux (x86-64); on
macOS and Windows (WSL 2), Docker Desktop should work but was not tested.

```sh
git clone https://github.com/tkn-tub/HyRASeL.git && cd HyRASeL
make paper        # simulate and evaluate all configs of the paper
```

This starts a container in the background and returns. The run continues
when the terminal is closed:

```sh
make attach       # watch the progress (tmux; detach with Ctrl-b d)
make logs         # follow the log in logs/hyrasel.log
make stop         # abort
```

When finished, the figures of the paper are in `figures/`. The single steps
can also be run for one config (without `CONFIG`, for all configs of the paper):

```sh
make simulation  CONFIG=Baseline   # OMNeT++ simulation
make postprocess CONFIG=Baseline   # post-processing (localization)
make plots       CONFIG=Baseline   # filtering, statistics and figures
make evaluation  CONFIG=Baseline   # postprocess + plots
```

`make help` lists all targets and options, e.g. `JOBS=<n>` for the number of
parallel simulation runs and post-processing jobs. The outputs are listed in
the [structure](#structure).

Every simulation config in `simulation/simulations/omnetpp.ini` has an
evaluation config with the same name in `evaluation/configs/`. The scenario is
configured in the former, the evaluation (sweep values, filters, plots) in the
latter. A new simulation config therefore also needs an evaluation config
(see [evaluation/README.md](evaluation/README.md)).

Configs of the paper:

| Figure in the paper                                   | Config                  |
| ----------------------------------------------------- | ----------------------- |
| Effect of the number of v-Anchor slots                | `SlotNumber`            |
| Comparison with the baseline methods                  | `Baseline`              |
| Effect of the inter-anchor distance                   | `AnchorDensity`         |
| Effect of the number of nodes                         | `NodeNumber`            |
| Effect of the node speed                              | `Speed`                 |
| Effect of the anchor-node message loss probability    | `PacketLoss-AnchorNode` |
| Effect of the node-node message loss probability      | `PacketLoss-NodeNode`   |

The remaining figures are drawings.

## Running without Docker

| Software | Version |
| -------- | ------- |
| [OMNeT++](https://omnetpp.org/download/) | 6.2.0 |
| [INET](https://inet.omnetpp.org/Download.html) | 4.5.4 |
| Python   | ≥ 3.11 (packages in [evaluation/requirements.txt](evaluation/requirements.txt)) |
| tmux     | any |

1. Install OMNeT++ as described in its
   [Installation Guide](https://doc.omnetpp.org/omnetpp/InstallGuide.pdf)
   and INET as described in its [installation instructions](https://inet.omnetpp.org/Installation.html),
   or both together with [opp_env](https://github.com/omnetpp/opp_env).
2. Create the Python environment:
   `python3 -m venv evaluation/.venv && evaluation/.venv/bin/pip install -r evaluation/requirements.txt`
3. Use the Makefiles of the two directories:

```sh
source <omnetpp>/setenv
make -C simulation INET_DIR=<inet>            # build; default INET_DIR: ./inet-4.5.4
make -C simulation run CONFIG=Baseline        # or run-all
make -C evaluation full-Baseline              # or full-all
```

The simulation can also be imported into the OMNeT++ IDE (see [simulation/README.md](simulation/README.md)).

## Use of AI Tools

Parts of the code and the documentation were written or revised with the
assistance of AI models (OpenAI ChatGPT, Anthropic Claude). All of it was
reviewed by the authors.

## Contact

Simon Schmitz-Heinen, [schmitz-heinen@ccs-labs.org](mailto:schmitz-heinen@ccs-labs.org)

## License

[GPL-3.0](LICENSE)
