#ifndef LOCALIZATION_PHYSICALLAYER_TIMESTAMPEDAPSKRADIO_H_
#define LOCALIZATION_PHYSICALLAYER_TIMESTAMPEDAPSKRADIO_H_

#include "inet/physicallayer/wireless/apsk/packetlevel/ApskRadio.h"

namespace localization {

using namespace inet;
using namespace inet::physicallayer;

/**
 * Adds receiver-local timing metadata without changing APSK reception rules.
 *
 * INET exposes reception timing through IReception while the frame is in the
 * radio. The application receives only the decoded Packet, so this subclass
 * copies the selected reception start into ReceptionTimingInd immediately
 * before the frame leaves the PHY.
 *
 * It can additionally drop an otherwise successfully decoded frame with a
 * static probability that depends on the roles (anchor/node) of transmitter and
 * receiver, to emulate blocked paths in packet-loss sensitivity studies.
 */
class INET_API TimestampedApskRadio : public ApskRadio
{
  protected:
    bool anchorRole = false;
    double anchorNodeLossProbability = 0;
    double nodeNodeLossProbability = 0;
    int lossRngIndex = 1;
    simsignal_t injectedLossDropSignal;

    virtual void initialize(int stage) override;

    /**
     * Draws the additional-loss decision for a frame that was decoded
     * successfully. Only draws a random number if the applicable probability is
     * positive, so a configuration without injected loss keeps its RNG streams.
     */
    virtual bool isDroppedByInjectedLoss(const IReception *reception);

    /** Attaches timing only to a frame that passed the PHY reception decision. */
    virtual void attachReceptionTiming(Packet *packet, const IReception *reception,
                                       bool successful) const;

    /**
     * Mirrors INET's whole-packet completion path and adds the timing tag at
     * the first point where a receiver-specific Packet object exists.
     */
    virtual void endReception(cMessage *timer) override;

  public:
    /** Role of the host this radio belongs to, as configured via isAnchor. */
    bool isAnchorRadio() const { return anchorRole; }
};

} // namespace localization

#endif
