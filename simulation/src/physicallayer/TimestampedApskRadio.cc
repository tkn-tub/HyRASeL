#include "TimestampedApskRadio.h"

#include "ReceptionTimingTag_m.h"
#include "inet/common/ModuleAccess.h"
#include "inet/physicallayer/wireless/common/medium/RadioMedium.h"

namespace localization {

Define_Module(TimestampedApskRadio);

void TimestampedApskRadio::initialize(int stage)
{
    ApskRadio::initialize(stage);
    if (stage != INITSTAGE_LOCAL)
        return;

    anchorRole = par("isAnchor");
    anchorNodeLossProbability = par("anchorNodeLossProbability");
    nodeNodeLossProbability = par("nodeNodeLossProbability");
    lossRngIndex = par("lossRngIndex");
    injectedLossDropSignal = registerSignal("injectedLossDrop");

    if (anchorNodeLossProbability < 0 || anchorNodeLossProbability > 1
        || nodeNodeLossProbability < 0 || nodeNodeLossProbability > 1)
        throw cRuntimeError(
            "TimestampedApskRadio: loss probabilities must be within [0, 1]");
}

bool TimestampedApskRadio::isDroppedByInjectedLoss(const IReception *reception)
{
    const auto transmitter = dynamic_cast<const TimestampedApskRadio *>(
        reception->getTransmission()->getTransmitter());
    if (transmitter == nullptr)
        throw cRuntimeError(
            "TimestampedApskRadio: injected loss requires a TimestampedApskRadio transmitter");

    const bool txIsAnchor = transmitter->isAnchorRadio();
    const std::string link = std::string(getContainingNode(transmitter)->getFullName())
        + (txIsAnchor ? " (anchor)" : " (node)") + " -> "
        + getContainingNode(this)->getFullName() + (anchorRole ? " (anchor)" : " (node)");

    double probability;
    const char *linkClass;
    if (anchorRole && txIsAnchor) {
        EV_DEBUG << "InjectedLoss " << link
                 << ": not applicable (anchor-anchor links have no injected loss) -> keep"
                 << endl;
        return false;
    }
    else if (anchorRole != txIsAnchor) {
        probability = anchorNodeLossProbability;
        linkClass = "anchor<->node";
    }
    else {
        probability = nodeNodeLossProbability;
        linkClass = "node<->node";
    }

    if (probability <= 0) {
        EV_DEBUG << "InjectedLoss " << link << ": class=" << linkClass
                 << ", p=0 (disabled) -> keep" << endl;
        return false;
    }

    const double draw = uniform(0, 1, lossRngIndex);
    const bool dropped = draw < probability;
    EV_DEBUG << "InjectedLoss " << link << ": class=" << linkClass
             << ", p=" << probability << ", draw=" << draw << " ("
             << (dropped ? "draw < p" : "draw >= p") << ") -> "
             << (dropped ? "DROP (marked with bit error)" : "keep") << endl;
    return dropped;
}

void TimestampedApskRadio::attachReceptionTiming(
    Packet *packet,
    const IReception *reception,
    bool successful) const
{
    // Corrupted packets are discarded by the MAC and must not expose a usable
    // ranging timestamp to higher layers.
    if (!successful)
        return;

    auto timing = packet->addTagIfAbsent<ReceptionTimingInd>();
    timing->setStartRxTime(reception->getStartTime());
}

void TimestampedApskRadio::endReception(cMessage *timer)
{
    // This is INET's normal Radio::endReception flow with one intentional
    // extension: the receiver timing tag is attached before sendUp(). Keeping
    // the remaining decisions identical preserves ApskScalarRadio behaviour.
    const auto part = static_cast<IRadioSignal::SignalPart>(timer->getKind());
    auto signal = static_cast<WirelessSignal *>(timer->getControlInfo());
    auto arrival = signal->getArrival();
    auto reception = signal->getReception();
    if (timer == receptionTimer && isReceiverMode(radioMode)
        && arrival->getEndTime() == simTime()) {
        auto transmission = signal->getTransmission();
        bool successful = medium->getReceptionDecision(
            this, signal->getListening(), transmission,
            part)->isReceptionSuccessful();
        // Additional loss is applied only to frames that survived the normal
        // reception decision; the effective loss is therefore p + (1 - p) * PER.
        if (!successful)
            EV_DEBUG << "InjectedLoss " << (IWirelessSignal *)signal
                     << ": not evaluated, normal PHY decision already failed" << endl;
        const bool injectedLoss = successful && isDroppedByInjectedLoss(reception);
        if (injectedLoss) {
            successful = false;
            emit(injectedLossDropSignal, 1L);
        }
        EV_INFO << "Reception ended: "
                << (successful ? "successfully " : "unsuccessfully ")
                << (injectedLoss ? "(injected loss) " : "")
                << (IWirelessSignal *)signal << " "
                << IRadioSignal::getSignalPartName(part)
                << " as " << reception << endl;
        auto macFrame = medium->receivePacket(this, signal);
        take(macFrame);
        decapsulate(macFrame);
        // SimpleMac discards packets with bit errors, exactly as for a PHY decoding failure.
        if (injectedLoss)
            macFrame->setBitError(true);
        attachReceptionTiming(macFrame, reception, successful);
        sendUp(macFrame);
    }
    else
        EV_INFO << "Reception ended: ignoring " << (IWirelessSignal *)signal
                << " " << IRadioSignal::getSignalPartName(part)
                << " as " << reception << endl;

    updateTransceiverState();
    updateTransceiverPart();
    if (timer == receptionTimer)
        receptionTimer = nullptr;
    delete timer;
    check_and_cast<RadioMedium *>(medium.get())->emit(
        IRadioMedium::signalArrivalEndedSignal,
        check_and_cast<const cObject *>(reception));
}

} // namespace localization
