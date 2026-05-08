# HyRASeL

This repository contains the source code and evaluation pipeline for the paper:

**HyRASeL: Hybrid UWB Two-Way-Ranging and Asynchronous Hyperbolic Self-Localization**

The repository includes the OMNeT++ simulation code, post-processing scripts,
plotting scripts, and the final data used to generate the paper figures.

## Associated Paper

- **Title:** HyRASeL: Hybrid UWB Two-Way-Ranging and Asynchronous Hyperbolic Self-Localization
- **Authors:** Simon Schmitz-Heinen, Anatolij Zubow, Tehmina Bibi, Falko Dressler
- **Venue:** Submitted to IEEE MASS 2026
- **Location:** Hong Kong
- **DOI / Preprint:** TBD

## Repository Structure

```text
.
├── omnetpp_code/
│   └── localization/
│       └── simulation/
│           ├── omnetpp.ini
│           ├── *.ned
│       └── src/
│           ├── *.cc
│           └── *.h
│
├── evaluation_code/
│   ├── omnet_post_processing.py
│   ├── omnet_plotting.py
│   ├── eval_run.sh
│   ├── requirements.txt
│   ├── data/
│   └── tmp/
│
├── README.md
└── LICENSE
```

## Requirements

The code was tested with the following setup:

- Ubuntu 24.04 LTS x86
- OMNeT++ 6.2.0
- INET 4.5.4
- Python 3.12.0

Python dependencies are listed in:

```text
evaluation_code/requirements.txt
```

## Setup

### OMNeT++ / INET

Install OMNeT++ 6.2.0 and INET 4.5.4 according to their official installation instructions.

The OMNeT++ project is located at:

```text
omnetpp_code/localization/
```

Import this folder into the OMNeT++ IDE as an existing project.

The main simulation configuration file is:

```text
omnetpp_code/localization/simulation/omnetpp.ini
```

### Python Environment

From the repository root:

```bash
cd evaluation_code
pip install -r requirements.txt
```

## Running the Simulations

The simulations are intended to be run through the OMNeT++ IDE.

Open:

```text
omnetpp_code/localization/simulation/omnetpp.ini
```

and run the relevant paper configurations:

```text
[Config Speed]
[Config NodeNum]
[Config AnchorDensity]
[Config SlotNumber]
```

These configurations evaluate:

- node speed
- number of nodes
- anchor density
- number of slots

The generated simulation data is used as input for the post-processing pipeline.

### Optional Command-Line Execution

The simulations may also be runnable via `opp_run`, depending on the local OMNeT++ and INET setup.

Example using `Cmdenv`:

```bash
cd omnetpp_code/localization/simulation

opp_run \
    -u Cmdenv \
    -f omnetpp.ini \
    -c Speed \
    -n .:..:/path/to/inet/src
```

Example using `Qtenv`:

```bash
cd omnetpp_code/localization/simulation

opp_run \
    -u Qtenv \
    -f omnetpp.ini \
    -c Speed \
    -n .:..:/path/to/inet/src
```

If INET or the project library needs to be loaded explicitly, add the appropriate `-l` arguments for your local installation.

The exact command can be derived from the OMNeT++ IDE run configuration under:

```text
Run -> Run Configurations...
```

or from the IDE console output after launching a simulation.

Refer to the OMNeT++ guides for more information. 

## Evaluation Pipeline

The evaluation code is located in:

```text
evaluation_code/
```

### Post-Processing

Run:

```bash
cd evaluation_code
bash eval_run.sh
```

This executes `omnet_post_processing.py` for the paper scenarios and methods.

The post-processing script expects simulation data in:

```text
evaluation_code/data/simulation_data/
```

with the following structure:

```text
data/simulation_data/SlotNumber/
data/simulation_data/NodeNumber/
data/simulation_data/AnchorDensity/
data/simulation_data/Speed/
```

The `--experiment_parameter` argument uses the syntax:

```text
parameterName:value1,value2,value3
```

Examples:

```text
speed:2,5,10,25,50
numNodes:4,9,16,25,36,49
interAnchorDist:130,200,270
numSlots:2,3,4,5
```

The processed output is written to:

```text
evaluation_code/tmp/
```

### Plotting

After post-processing has completed, run:

```bash
cd evaluation_code
python3 omnet_plotting.py
```

The plotting script reads processed data from:

```text
evaluation_code/tmp/
```

and writes figures to:

```text
evaluation_code/tmp/figs/
```

The generated figure filenames contain the corresponding figure numbers from the paper.

## Reproducing the Paper Results

To reproduce the paper figures:

1. Run the OMNeT++ configurations listed above, or use the provided final data.
2. Run the post-processing pipeline:

   ```bash
   cd evaluation_code
   bash eval_run.sh
   ```

3. Generate the figures:

   ```bash
   python3 omnet_plotting.py
   ```

Generated figures are stored in:

```text
evaluation_code/tmp/figs/
```

## Data

The final data used for the paper is included via Git LFS.

Before using the repository, make sure Git LFS is installed:

```bash
git lfs install
git lfs pull
```

If the data files are missing, post-processing and plotting may fail.

## Configuration

Simulation parameters are configured in:

```text
omnetpp_code/localization/simulation/omnetpp.ini
```

The main evaluated parameters are:

| Parameter | Description |
|---|---|
| `speed` | Mobility speed of the simulated nodes |
| `numNodes` | Number of simulated nodes |
| `interAnchorDist` | Distance between anchors |
| `numSlots` | Number of slots used by the localization protocol |

Post-processing parameters are configured in:

```text
evaluation_code/eval_run.sh
```

The evaluated methods include:

```text
TWR
CENTRALIZED
DISTRIBUTED
HYPERBOLIC_SELF_LOCALIZATION
```

## License

This repository is released under the GPLv3. See `LICENSE` for details.

OMNeT++ and INET are not part of this repository and are subject to their own licenses. Users are responsible for installing OMNeT++ and INET separately and complying with their respective license terms.

## AI Usage

Parts of the code were created with assistance from large language models, including OpenAI's ChatGPT and Anthropic's Claude. All code was reviewed and adapted by the authors.

## Contact

For questions, please contact:

Simon Schmitz-Heinen  
<schmitz-heinen@ccs-labs.org>
