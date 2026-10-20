package localization.physicallayer;

import inet.physicallayer.wireless.apsk.packetlevel.ApskScalarRadio;

@namespace(localization);

// ApskScalarRadio that carries the selected reception start with the decoded
// packet. Modulation, receiver decisions, path loss, and interference remain
// the responsibility of the configured INET radio medium and error model.
//
// Optionally, a frame that passed the normal reception decision is dropped with
// a fixed, link-class-specific probability. This models a blocked path in a
// simple way, e.g. for packet-loss sensitivity studies. The decision is taken
// independently per receiver and per frame. The link class is derived from the
// role (isAnchor) of the transmitting and receiving radio:
//   anchor <-> node  : anchorNodeLossProbability (either direction)
//   node   <-> node  : nodeNodeLossProbability
//   anchor <-> anchor: no additional loss
// Both probabilities default to 0 (no additional loss).
module TimestampedApskScalarRadio extends ApskScalarRadio
{
    parameters:
        @class(TimestampedApskRadio);
        bool isAnchor = default(false);  // Role of the host this radio belongs to
        double anchorNodeLossProbability = default(0);
        double nodeNodeLossProbability = default(0);
        int lossRngIndex = default(1);  // Local RNG; map to a dedicated global RNG via rng-1
        @signal[injectedLossDrop](type=long);
        @statistic[injectedLossDropCount](source=injectedLossDrop; record=count);
}
