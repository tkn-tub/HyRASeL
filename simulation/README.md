# Simulation

OMNeT++ project (named `localization`) that simulates the localization methods
and writes the transmitted and received messages of every host as CSV traces.
Installation and the complete workflow are described in the [top-level README](../README.md).

```text
simulation/
├── src/
│   ├── apps/avaapp/             AvaApp: localization methods and CSV trace output
│   ├── common/                  AvaProtocol: INET protocol of the app messages
│   ├── linklayer/simplemac/     SimpleMac: passes frames between app and radio, no medium access
│   └── physicallayer/           TimestampedApskScalarRadio (see below)
└── simulations/
    ├── LocalizationNetwork.ned  Network
    ├── omnetpp.ini              Configs
    └── results/traces/          Output: CSV traces (see below)
```

The method is selected with `**.app.method`:

| `method`                       | In the paper                          |
| ------------------------------ | ------------------------------------- |
| `CENTRALIZED`                  | HyRASeL                               |
| `TWR`                          | TWR (baseline)                        |
| `HYPERBOLIC_SELF_LOCALIZATION` | Chiasson et al. (baseline)            |
| `DISTRIBUTED`                  | HyRASeL with distributed v-Anchor selection (not evaluated) |

## Build and Run

```sh
make INET_DIR=<path to INET 4.5.4>
make run CONFIG=Speed            # all runs of one config
make run CONFIG=Speed RUNS=0..3  # only some runs
make run-all                     # all configs
```

The header of the Makefile lists all options. Alternatively, import the project into the
OMNeT++ IDE; it references the INET project by the name `inet-4.5.4`.

## Output

Every host writes `simulations/results/traces/<Config>/<iteration variables>/<host>_rep<N>_log.csv`
with one row per transmitted or received message: simulation time, local clock
time, clock time with timestamp noise, message fields and the true position and
velocity of the host.

## Radio

`TimestampedApskScalarRadio` is INET's `ApskScalarRadio` with two additions:

- It attaches the start time of a successful reception to the received packet,
  which the app uses as receive timestamp.
- It can drop a frame that was received successfully with a fixed probability,
  to emulate blocked paths (configs `PacketLoss-*`). The probability is
  `anchorNodeLossProbability` for links between an anchor and a node and
  `nodeNodeLossProbability` for links between two nodes; links between anchors
  are not affected. The decision is drawn per frame and receiver from a
  dedicated RNG stream, so runs without injected loss are unchanged.
