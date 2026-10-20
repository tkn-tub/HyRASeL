//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 

#include "SimpleMac.h"

#include "inet/common/ModuleAccess.h"
#include "inet/common/ProtocolGroup.h"
#include "inet/common/ProtocolTag_m.h"
#include "inet/common/checksum/EthernetCRC.h"
#include "inet/common/packet/chunk/ByteCountChunk.h"
#include "inet/linklayer/common/InterfaceTag_m.h"
#include "inet/linklayer/common/MacAddressTag_m.h"
#include "inet/linklayer/common/UserPriority.h"
#include "inet/linklayer/common/UserPriorityTag_m.h"
#include "../../common/AvaProtocol.h"


namespace localization {

using namespace inet;
using namespace inet::physicallayer;

Define_Module(SimpleMac);

void SimpleMac::initialize(int stage)
{
    MacProtocolBase::initialize(stage);
    if (stage == INITSTAGE_LOCAL) {
        EV << "Initializing stage 0\n";
        bitrate = par("bitrate");
        headerLength = B(par("headerLength"));
        if (headerLength > B(255))
            throw cRuntimeError("The specified headerLength is too large");

        txQueue = getQueue(gate(upperLayerInGateId));


    }
    else if (stage == INITSTAGE_LINK_LAYER) {
        // subscribe for the information of the carrier sense
        radio.reference(this, "radioModule", true);
        cModule *radioModule = check_and_cast<cModule *>(radio.get());
        radioModule->subscribe(IRadio::receptionStateChangedSignal, this);
        radioModule->subscribe(IRadio::transmissionStateChangedSignal, this);
        radio->setRadioMode(IRadio::RADIO_MODE_RECEIVER);
    }
}

void SimpleMac::finish()
{
}

void SimpleMac::configureNetworkInterface()
{
    // data rate
    networkInterface->setDatarate(bitrate);
    networkInterface->setBroadcast(true);
}

void SimpleMac::handleUpperPacket(Packet *packet)
{
    EV << "Handling Upper Packet"<< endl;
    encapsulate(packet);
    sendDataFrame(packet);
}

void SimpleMac::handleLowerPacket(Packet *packet)
{
    EV << "received message from lower layer: " << packet << endl;
    // Very simple frame integrity check -> currently no FCS or so, so bit error results in packet error
    if (packet->hasBitError()){
        EV << "Packet has bit errors -> Deleting packet." << endl;
        delete packet;
        return;
    }
    else {
        EV << "Packet does not have any bit errors." << endl;
    }

    // Only the localization protocol runs over this interface
    packet->addTagIfAbsent<PacketProtocolTag>()->setProtocol(&avaProtocol);
    auto dispatchReq = packet->addTag<DispatchProtocolReq>();
    dispatchReq->setProtocol(&avaProtocol);

    // Set interface tag
    auto interfaceIndTag = packet->addTagIfAbsent<InterfaceInd>();
    interfaceIndTag->setInterfaceId(networkInterface->getInterfaceId());

    sendUp(packet);
}

void SimpleMac::handleSelfMessage(cMessage *msg)
{
    EV << "received self message: " << msg << endl;
}

void SimpleMac::encapsulate(Packet *packet)
{
//    auto macPkt = makeShared<SimpleMacHeader>(); // Omitted for now
}

void SimpleMac::decapsulate(Packet *frame)
{
    // Can ignore that for the given purpose
}

void SimpleMac::sendDataFrame(Packet *frameToSend)
{
    EV << "sending Data frame " << frameToSend->getName() << endl;
    radio->setRadioMode(IRadio::RADIO_MODE_TRANSMITTER);
    sendDown(frameToSend);
}

void SimpleMac::handleStopOperation(LifecycleOperation *operation)
{
    MacProtocolBase::handleStopOperation(operation);
}

void SimpleMac::handleCrashOperation(LifecycleOperation *operation)
{
    MacProtocolBase::handleCrashOperation(operation);
}

void SimpleMac::processUpperPacket()
{
    EV << "Received Upper Packet. What to do next?" << endl;
    Packet *packet = dequeuePacket();
    handleUpperPacket(packet);
}

bool SimpleMac::isMediumFree()
{
    return radio->getReceptionState() == IRadio::RECEPTION_STATE_IDLE;
}

bool SimpleMac::isReceiving()
{
    return radio->getReceptionState() == IRadio::RECEPTION_STATE_RECEIVING;
}

queueing::IPassivePacketSource *SimpleMac::getProvider(cGate *gate)
{
    return (gate->getId() == upperLayerInGateId) ? txQueue.get() : nullptr;
}

void SimpleMac::handleCanPullPacketChanged(cGate *gate)
{
    Enter_Method("handleCanPullPacketChanged");
    EV << "Handling Can Pull Packet Changed " << endl;
    if (gate->getId() == upperLayerInGateId)
    {
        processUpperPacket();
    }
}

void SimpleMac::handlePullPacketProcessed(Packet *packet, cGate *gate, bool successful)
{
    Enter_Method("handlePullPacketProcessed");
    throw cRuntimeError("Not supported callback");
}
void SimpleMac::receiveSignal(cComponent *source, simsignal_t signalID, intval_t value, cObject *details)
{
    Enter_Method("%s", cComponent::getSignalName(signalID));
    EV << "Receiving Signal:" << source << signalID<< value << details << endl;

    if (signalID == IRadio::transmissionStateChangedSignal) {
        IRadio::TransmissionState newRadioTransmissionState = static_cast<IRadio::TransmissionState>(value);
        if (transmissionState == IRadio::TRANSMISSION_STATE_TRANSMITTING && newRadioTransmissionState == IRadio::TRANSMISSION_STATE_IDLE) {
            // KLUDGE we used to get a cMessage from the radio (the identity was not important)
//            executeMac(EV_FRAME_TRANSMITTED, new cMessage("Transmission over"));
            radio->setRadioMode(IRadio::RADIO_MODE_RECEIVER);
        }
        transmissionState = newRadioTransmissionState;
    }
}


SimpleMac::~SimpleMac() {
}

} /* namespace inet */
