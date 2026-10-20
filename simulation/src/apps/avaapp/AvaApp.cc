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

#include "AvaApp.h"

#include "inet/applications/base/ApplicationPacket_m.h"
#include "inet/common/ModuleAccess.h"
#include "inet/common/lifecycle/ModuleOperations.h"
#include "inet/common/packet/Packet.h"
#include "inet/common/packet/chunk/ByteCountChunk.h"
#include "AvaMsg_m.h"

#include "inet/common/ModuleAccess.h"
#include "inet/common/lifecycle/ModuleOperations.h"

#include "inet/common/IProtocolRegistrationListener.h"
#include "inet/common/ProtocolTag_m.h"
#include "inet/networklayer/common/NetworkInterface.h"
#include "inet/linklayer/common/InterfaceTag_m.h"
#include "inet/networklayer/common/InterfaceTable.h"
#include "../../physicallayer/ReceptionTimingTag_m.h"

#include <cstring>
#include <filesystem>

using namespace inet;

namespace localization{

Define_Module(AvaApp);

simsignal_t AvaApp::packetReceivedSignal = registerSignal("packetReceived");
simsignal_t AvaApp::avaMsgRecording = registerSignal("avaMsgRecording");

void AvaApp::initialize(int stage)
{
//    ApplicationBase::initialize(stage);

    // Add Clock Module
    ClockUserModuleMixin::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        initializeParameters();
        calculateDerivedParameters();
    }
    else if (stage == INITSTAGE_APPLICATION_LAYER) {
            initializeTimers();
            initializeLogging();
            initializeLocalization();
            initializeRxTimestamping();
            registerProtocol(avaProtocol, gate("socketIn"), gate("socketOut"));
//            registerProtocol(avaProtocol, gate("socketIn"), SP_INDICATION);
        }
}

void AvaApp::initializeRxTimestamping(){
    int numInterfaces = getParentModule()->par("numInterfaces");
    // The PHY carries the selected start time with the decoded packet. AvaApp
    // keeps only per-interface working values used by the ranging protocols.
    rxTimestamps = std::vector<simtime_t>(numInterfaces);
    rxTimestamps_c = std::vector<clocktime_t>(numInterfaces);
}


void AvaApp::initializeLocalization(){
    switch (selectedMethod){
        case TWR:
            initializeTwr();
            break;
        case DISTRIBUTED:
            initializeDistributedSelectionSelfLoc();
            break;
        case CENTRALIZED:
            initializeCentralizedSelectionSelfLoc();
            break;
        case HYPERBOLIC_SELF_LOC:
            initializeStandardSelfLoc();
            break;
        default:
            throw cRuntimeError("No valid localization method selected");
        }
}

void AvaApp::initializeDistributedSelectionSelfLoc(){
    totalNumSlots = numAnchors + numVAnchorSlots;
    if (isAnchor){
        scheduleTimersAfterSlots(selfLocTxTimer, selfLocTxTimer_c , nodeId, startTime, startTime_c);
    }
    else{
        scheduleTimersAfterSlots(contentionTimer, contentionTimer_c, numAnchors+1, startTime, startTime_c);
    }

}

void AvaApp::initializeCentralizedSelectionSelfLoc(){
    totalNumSlots = numAnchors + numVAnchorSlots;
    if (isAnchor){
        scheduleTimersAfterSlots(selfLocTxTimer, selfLocTxTimer_c , nodeId, startTime, startTime_c);
    }
}

void AvaApp::initializeStandardSelfLoc(){
    totalNumSlots = numAnchors;
    if (isAnchor){
        scheduleTimersAfterSlots(selfLocTxTimer, selfLocTxTimer_c , nodeId, startTime, startTime_c);
    }
}


void AvaApp::initializeTwr(){
    if(isAnchor){
        return;
    }
    int startPosition = nodeId - numAnchors;
    scheduleTimersAfterSlots(pollTimer, pollTimer_c, (startPosition*(2+numAnchors)), startTime, startTime_c);
}

void AvaApp::handleMessageWhenUp(cMessage *msg) {
    if(msg->isSelfMessage()){
        handleSelfMessage(msg);
        return;
    }

    // If message arrives on the TWR interface, handle it
    auto packet = dynamic_cast<Packet*>(msg);
    ASSERT(packet);
    if (!packet){
        delete msg;
    }

    if (msg->arrivedOn("socketIn")){;
        handlePacket(packet);
        return;
    }
    else
        EV << "Message did not arrive on any in-Gate" << endl;
    delete msg;

}

void AvaApp::handleSelfMessage(cMessage *msg){
    if (msg == selfMsg) {
        return;
    }

    // Anchor Self-Localization Timer
    if ((msg == selfLocTxTimer) || (msg==selfLocTxTimer_c)){
        handleSelfLocTxTimer();
        return;
    }

    // CONTENTION TIMER
    else if (msg == contentionTimer || msg==contentionTimer_c){
        handleContentionTimer();
        return;
    }

    // SUCCESSFUL CONTENTION TIMER
    else if (msg == succContTimer || msg==succContTimer_c){
        handleSuccessfulContentionTimer();
        return;
    }

    // CENTRALIZED SELECTION / T
    else if (msg == selectedNodeTxTimer || msg == selectedNodeTxTimer_c){
        handleSelectedNodeTxTimer();
        return;
    }

    // TWR MESSAGES
    else if (msg == pollTimer || msg == pollTimer_c){
        handlePollTimer();
        return;
    }
    else if(msg == txTimer || msg == txTimer_c) {
        handleTxTimer();
        return;
    }

}


void AvaApp::handlePacket(Packet *pkt){
    if (!isForMe(pkt)){
        EV_INFO << "Packet not for me -> Dropping it." << endl;
        delete pkt;
        return;
    }

    const AvaPacketInfo info = extractAvaPacketInfo(pkt);
    const int interfaceIdx = info.interfaceIdx;
    if (interfaceIdx < 0 || interfaceIdx >= static_cast<int>(rxTimestamps.size()))
        throw cRuntimeError("AvaApp: packet contains invalid interface index %d", interfaceIdx);

    const auto timing = pkt->findTag<ReceptionTimingInd>();
    if (timing == nullptr)
        throw cRuntimeError(
            "AvaApp: received packet has no ReceptionTimingInd; configure "
            "TimestampedApskScalarRadio");

    // Timestamped radios attach the PHY-selected reception start to the exact
    // packet that survived decoding. Converting that simulation time through
    // this application's clock preserves drift for all ranging protocols.
    rxTimestamps[interfaceIdx] = timing->getStartRxTime();
    rxTimestamps_c[interfaceIdx] = computeClockTimeFromSimTime(
        rxTimestamps[interfaceIdx]);
    EV_INFO << "Rx packet over interface " << interfaceIdx << endl;
    EV_INFO << "Timestamp on interface " << rxTimestamps[interfaceIdx] << endl;

    // Receiving message; writing it into file
    writeToResultFile(rxTimestamps[interfaceIdx], rxTimestamps_c[interfaceIdx], info.seqNo, info.srcId, info.destId, info.msgType, info.interfaceIdx);

    EV_INFO << "Received AvaMsg of type " << msgNames[info.msgType] << " with SeqNo " << info.seqNo << " from " << info.srcId << " on interface " << info.interfaceIdx << endl;

    switch (selectedMethod){
    case TWR:
        handleTwrPacket(pkt, info);
        break;
    case DISTRIBUTED:
        handleDistributedSelectionSelfLocPacket(info);
        break;
    case CENTRALIZED:
        handleCentralizedSelectionSelfLocPacket(info);
        break;
    case HYPERBOLIC_SELF_LOC:
        handleStandardSelfLocPacket();
        break;
    default:
        throw cRuntimeError("No valid localization method selected");
    }

    rescheduleAllClocksIfNeeded(info);
    // Reset timestamp registry
    rxTimestamp   = SIMTIME_ZERO;
    rxTimestamps[interfaceIdx] = SIMTIME_ZERO;
    rxTimestamp_c = SIMTIME_AS_CLOCKTIME(SIMTIME_ZERO);
    rxTimestamps_c[interfaceIdx] = SIMTIME_AS_CLOCKTIME(SIMTIME_ZERO);

    EV_INFO << "Handled Packet -> Deleting it" << endl;
    delete pkt;

}

void AvaApp::handleStandardSelfLocPacket(){
    // NOTE: after the reception of a standard self-localization message, nothing happens, as we only note down the timestamp
}


void AvaApp::handleCentralizedSelectionSelfLocPacket(const AvaPacketInfo& info){
    // ANCHOR LOGIC
    if (isAnchor){
        return;
    }

    // NODE LOGIC
    updateSequenceNumberCounter(info);
    updateLocalSequenceNumber(info);
    bool selectedNodeTxTimerIsScheduled = (selectedNodeTxTimer_c->isScheduled() || selectedNodeTxTimer->isScheduled());
    if (info.srcIsAnchor && (nodeId == info.contId) && !selectedNodeTxTimerIsScheduled){
        // SCHEDULE selectedNodeTxTimer
        EV_INFO << "Selected as node! Scheduling transmission" << endl;
        int waitingSlots = numAnchors - info.srcId;
        scheduleTimersAfterSlots(selectedNodeTxTimer, selectedNodeTxTimer_c, waitingSlots);
    }

}


void AvaApp::handleDistributedSelectionSelfLocPacket(const AvaPacketInfo& info){
    // ANCHOR LOGIC
    if (isAnchor){
        updateContendingNode(info);
        return;
    }

    // NODE LOGIC
    updateSequenceNumberCounter(info);
    updateLocalSequenceNumber(info);
    bool srcOverheardOtherNode = (info.contId !=nodeId);
    bool contentionWasSuccessful = (vAnchorTxCounter != 1);
    bool succContTimerIsScheduled = (succContTimer->isScheduled() || succContTimer_c->isScheduled());
    // Not eligible for further transmission if some node overheard another node
    // Timer must be scheduled, transmitter must be anchor and contentionId is neither our ID nor -1, then we cancel
    if (succContTimerIsScheduled && (info.srcIsAnchor) && srcOverheardOtherNode && !contentionWasSuccessful){
        EV << "Contention seems not to have been successful! Canceling succContTimer" << endl;
        if (useClockTime){
            cancelClockEvent(succContTimer_c);
        }
        else{
            cancelEvent(succContTimer);
        }
    }

}

void AvaApp::handleTwrPacket(Packet *pkt, const AvaPacketInfo& info){
    // ANCHOR LOGIC
    if (isAnchor){
        if ((info.msgType == POLL)){
            int numSlots = nodeId-1;
            currentPkt = pkt->dup();
            scheduleTimersAfterSlots(txTimer, txTimer_c, numSlots);
        }
        return;
    }

    // NODE LOGIC
    updateLocalSequenceNumber(info);
    bool txTimerScheduled = txTimer->isScheduled() || txTimer_c->isScheduled();
    // Only one FINAL per POLL round: a RESPONSE whose reception ends at exactly the same
    // simTime as the txTimer fires is still delivered (zero-delay send chain) and must not
    // trigger a second FINAL while the first one is being transmitted.
    bool finalAlreadyScheduledForRound = (info.seqNo == lastFinalTwrSeqNo);
    if (!txTimerScheduled && !finalAlreadyScheduledForRound && (info.msgType == RESPONSE)){
        lastFinalTwrSeqNo = info.seqNo;
        int numSlots = numAnchors - info.srcId;
        currentPkt = pkt->dup();
        scheduleTimersAfterSlots(txTimer, txTimer_c, numSlots);
    }
}

void AvaApp::updateSequenceNumberCounter(const AvaPacketInfo& info){

     //  If the sequenceNumber (from a real Anchor) is new, we reset the counter
     if (info.seqNo != prevSeqNo && info.srcIsAnchor){
         prevSeqNo = info.seqNo;
         seqNoCounter = 0;
         // Clean heardAnchors set, as we have a new sequenceNumber
         EV << "HeardAnchorsSet Size before clearing: " << heardAnchors.size() << endl;
         heardAnchors.clear();
         EV << "HeardAnchorsSet Size after clearing: " << heardAnchors.size() << endl;
      }

     // Update for both, v-Anchors and static anchors
     auto [iterator, inserted] = heardAnchors.insert(info.srcId);
     if (inserted){
          seqNoCounter++;
     }
     EV << "Current HeardAnchorsSet Size: " << heardAnchors.size() << endl;
     EV << "Current seqNoCounter Size: " << seqNoCounter << endl;

}


void AvaApp::updateContendingNode(const AvaPacketInfo& info){
    // vAnchors sets contId to current counter; thus is not -1
    bool srcIsNotVAnchor = (info.contId == -1);
    if (!info.srcIsAnchor && srcIsNotVAnchor){ // Done only by anchors
        contendingNode = info.srcId ;
    }
}

AvaApp::AvaPacketInfo AvaApp::extractAvaPacketInfo(Packet *pkt){
    auto payload = pkt->peekAtFront<AvaMsg>();
    AvaPacketInfo info = {
        .seqNo  = payload->getSeqNo(),
        .srcId  = payload->getSenderId(),
        .destId = payload->getReceiverId(),
        .msgType= payload->getMsgType(),
        .contId = payload->getContenderId(),
        .interfaceIdx = payload->getInterfaceIdx(),
        .srcIsAnchor = payload->getSenderId() <= numAnchors
        };
    return info;
}

void AvaApp::updateLocalSequenceNumber(const AvaPacketInfo& info){
    // It is possible for nodes not to be in range of anchors and loose track of the sequenceNumber
    // Whenever we overhear a seqNo, we actualize the sequence number
    ASSERT(!isAnchor);
    if ((info.seqNo > selfLocSeqNo)){
            selfLocSeqNo = info.seqNo;
        }
}



void AvaApp::rescheduleAllClocksIfNeeded(const AvaPacketInfo& info){
    if (!useRescheduling){
        return;
    }
    // Not used in the paper (useRescheduling = false)
    int srcVAnchorSlot = ((info.contId == -1) ? 0 : info.contId); // calculated for node and anchor, but only node is relevant

    if(selfLocTxTimer_c->isScheduled() && useRescheduling){ // Only done by anchors
        int remainingSlots = (info.srcIsAnchor
                ? ((totalNumSlots + nodeId - info.srcId - 1) % totalNumSlots)
                : ((totalNumSlots + nodeId - (numAnchors + srcVAnchorSlot + 1) -1) % totalNumSlots));
        rescheduleClockTimer(selfLocTxTimer_c, remainingSlots);
    }

    if(contentionTimer_c->isScheduled() && useRescheduling){ // Only done by nodes
        int remainingSlots =  (info.srcIsAnchor ? (numAnchors - info.srcId) : (totalNumSlots - srcVAnchorSlot - 1));
        rescheduleClockTimer(contentionTimer_c, remainingSlots);
    }

    if(succContTimer_c->isScheduled() && useRescheduling){ // only done by nodes
        ASSERT(vAnchorTxCounter==-1);
        int remainingSlots = (info.srcIsAnchor
                ? (numAnchors - info.srcId + vAnchorTxCounter)
                : ((vAnchorTxCounter - srcVAnchorSlot -1 + totalNumSlots) % totalNumSlots)); // + totalNumSlots keeps the result of % non-negative
        rescheduleClockTimer(succContTimer_c, remainingSlots);
    }
}




void AvaApp::rescheduleClockTimer(ClockEvent *clockTimer, int numSlots){
    auto oldTxTime = clockTimer->getArrivalTime();
    cancelClockEvent(clockTimer);
    clocktime_t rescheduleTime_c = numSlots*slotDuration_c + aTurnaroundTime_c;
    scheduleClockEventAfter(rescheduleTime_c, clockTimer);
    auto newTxTime = clockTimer->getArrivalTime();
    EV_INFO << "Rescheduling " << clockTimer->getName() << " from " << oldTxTime << " s to " << newTxTime << " s (diff: " << oldTxTime - newTxTime << " s)" << endl;
    EV_INFO << "Remaining slots: " << numSlots << endl;
}


void AvaApp::handleSelfLocTxTimer(){
    // Reschedule the timer
    selfLocSeqNo++;
    if (selectedMethod == CENTRALIZED){
        contendingNode = selectNode(selfLocSeqNo);
        EV_INFO << "Selected Node: " << contendingNode << " for seqNo: " << selfLocSeqNo << endl;
    }

    sendAvaPacket(selfLocSeqNo, nodeId, 0xFF, SELF_LOC, contendingNode);

    // Resetting
    contendingNode = -1;

    // reschedule next transmission - Assuming perfect sync here
    scheduleTimersAfterSlots(selfLocTxTimer, selfLocTxTimer_c, totalNumSlots);
}


void AvaApp::handleContentionTimer(){

    // Immediately reschedule contentionTimer
    scheduleTimersAfterSlots(contentionTimer, contentionTimer_c, totalNumSlots);

    // If still scheduled, then I should transmit in the next slot(s)
    bool succContTimerIsScheduled = succContTimer->isScheduled() || succContTimer_c->isScheduled();

    if (succContTimerIsScheduled){
        EV << "Successful Contention Timer is scheduled; Thus not contending; " << endl;
        return;
    }

    if (passedProbabilityCheck()){
        EV << "Not Transmitting (check not passed)" << endl;
        return;
    }

     if (!overheardSufficientAnchors()){
         EV_INFO << "Not contending as not sufficiently many anchors overheard" << endl;
         return;
     }
     EV_INFO << seqNoCounter << " Anchors overheard; Thus transmitting!" << endl;

    // Transmit packet if chance is good
    EV << "Transmitting! (all checks passed)" << endl;
    sendAvaPacket(selfLocSeqNo, nodeId, 0xFF, SELF_LOC);
    vAnchorTxCounter = 1;
    scheduleTimersAfterSlots(succContTimer, succContTimer_c, totalNumSlots+vAnchorTxCounter);
}

void AvaApp::handleSuccessfulContentionTimer(){
    // vAnchorTxCounter is transmitted as contId to indicate that this is not the contention slot
    EV_INFO << "Contention was successful; Transmitting next message in slot ";

    sendAvaPacket(selfLocSeqNo, nodeId, 0xFF, SELF_LOC, vAnchorTxCounter);

    vAnchorTxCounter++;
    EV_INFO << vAnchorTxCounter << endl;
    if (vAnchorTxCounter == numVAnchorSlots){
        EV_INFO << "Reached vAnchor Transmission Limit!" << endl;
        vAnchorTxCounter = -1;
        return;
    }

    scheduleTimersAfterSlots(succContTimer, succContTimer_c, totalNumSlots+1);
    EV_INFO << "Rescheduling succContTimer!" << endl;
}

void AvaApp::handleSelectedNodeTxTimer(){

    // sufficientlyAnchors CONDITION?
    if (!overheardSufficientAnchors()){
         EV_INFO << "Not transmitting as not sufficiently many anchors were overheard ("<< seqNoCounter << ")!" << endl;
         vAnchorTxCounter = -1;
         return;
     }

    vAnchorTxCounter++;
    EV_INFO << "Selection Timer expired; Transmitting Message";
    sendAvaPacket(selfLocSeqNo, nodeId, 0xFF, SELF_LOC, vAnchorTxCounter);

    if (vAnchorTxCounter >= (numVAnchorSlots-1)){ // Stop if we have send to often
        EV_INFO << "Transmitted sufficiently often!" << endl;
        vAnchorTxCounter = -1;
        return;
    }

    EV_INFO << "Reschedule selectedNodeTxTimer!" << endl;
    scheduleTimersAfterSlots(selectedNodeTxTimer, selectedNodeTxTimer_c, totalNumSlots+1);

}

void AvaApp::handlePollTimer(){
    // Every node polls once per TWR round; the round lasts numNodes * (2 + numAnchors) slots
    sendAvaPacket(twrSeqNo++, nodeId, 0xFF, POLL, -1, "POLL");
    scheduleTimersAfterSlots(pollTimer, pollTimer_c, (numNodes*(2+numAnchors)));
}

void AvaApp::handleTxTimer(){
    ASSERT(currentPkt!=nullptr);
    auto payload = currentPkt->peekAtFront<AvaMsg>();
    int seqNo = payload->getSeqNo();
    int srcId = payload->getSenderId();
    int msgType = payload->getMsgType();

    switch (msgType){
        case POLL:
            sendAvaPacket(seqNo, nodeId, srcId, RESPONSE, -1, "RESPONSE");
            break;
        case RESPONSE:
            sendAvaPacket(seqNo, nodeId, 0xFF, FINAL, -1, "FINAL");
            break;
        case FINAL:
            if (pollTimer->isScheduled()){
                cancelEvent(pollTimer);
            }
            else if (pollTimer_c->isScheduled()){
                cancelClockEvent(pollTimer_c);
            }

        default:
            EV << "Unknown Type" << endl;
            delete currentPkt;
            return;
    }

    // RESET
    delete currentPkt;
    currentPkt = nullptr;
}



bool AvaApp::overheardSufficientAnchors(){
    return seqNoCounter >= 4;
}

bool AvaApp::passedProbabilityCheck(){
    double transmissionChance = 1.0/numNodes;
    double probCheck = uniform(0,1);
    EV << "Probability check: " << probCheck << endl;
    return probCheck > transmissionChance;
}


void AvaApp::handleSelfLocPacket(Packet *pkt){
    delete pkt;
}

bool AvaApp::isForMe(Packet *pkt){
    auto payload = pkt->peekAtFront<AvaMsg>();
    return payload->getReceiverId() == nodeId
           || payload->getReceiverId()==0xFF;
}



void AvaApp::configureAvaPacket(Packet *packet, int seqNo, int senderId, int destinationId, AvaMessageType msgType, int contenderId, int interfaceIdx){
     // Set payload
    const auto &payload = makeShared<AvaMsg>();
    payload->setSeqNo(seqNo);
    payload->setSenderId(senderId);
    payload->setReceiverId(destinationId); //Broadcast
    payload->setMsgType(msgType);
    payload->setContenderId(contenderId);
    payload->setChunkLength(B(5));
    payload->setInterfaceIdx(interfaceIdx);
    packet->insertAtBack(payload);
    if (packetLength > B(5))
        packet->insertAtBack(makeShared<ByteCountChunk>(packetLength - B(5)));

    // Access to InterfaceTable, to find system-wide ID
    IInterfaceTable *interfaceTable = check_and_cast<IInterfaceTable *>(getParentModule()->getSubmodule("interfaceTable"));
    // INET calls Vektor-Interfaces "wlan0", "wlan1" etc.
    std::string interfaceName = "wlan" + std::to_string(interfaceIdx);
    NetworkInterface *networkInterface = interfaceTable->findInterfaceByName(interfaceName.c_str());
    packet->addTag<InterfaceReq>()->setInterfaceId(networkInterface->getInterfaceId());
    packet->addTag<PacketProtocolTag>()->setProtocol(&avaProtocol);
}

void AvaApp::writeToResultFile(simtime_t simTimestamp, clocktime_t clockTimestamp, int seqNo, int srcId, int destId, int msgType, int interfaceIdx){
    if (!recordInCsv){
        return;
    }
    const Coord& pos = mobility->getCurrentPosition();
    const Coord& vel = mobility->getCurrentVelocity();

    double sigma = 300e-12;
    double noise = normal(0, sigma, 1);
    clocktime_t noisyTimestamp = clockTimestamp + clocktime_t(noise);
    EV_INFO << "Sigma: " << sigma << "; Noise: " << noise << endl;

    EV_INFO << "Writing To File @ clockTime: " << clockTimestamp << "; noisyTime: " << noisyTimestamp << "; simTime: " << simTimestamp << endl;

    logFile << simTimestamp
            << "," << clockTimestamp
            << "," << noisyTimestamp
            << "," << seqNo
            << "," << srcId
            << "," << destId
            << "," << msgType
            << "," << interfaceIdx
            << "," << pos.getX()
            << "," << pos.getY()
            << "," << pos.getZ()
            << "," << vel.getX()
            << "," << vel.getY()
            << "," << vel.getZ()
            << "\n";
}


void AvaApp::handleStartOperation(LifecycleOperation *operation){
    // Not used
}

void AvaApp::handleStopOperation(LifecycleOperation *operation){
    delayActiveOperationFinish(par("stopOperationTimeout"));
}

void AvaApp::handleCrashOperation(LifecycleOperation *operation){
    // Not used
}

uint64_t AvaApp::easyHash(uint64_t x){
    // Hash function (bases on SplitMix64)
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

int AvaApp::selectNode(int seqNo){

    printVector(selectedNodesPerSlotBuffer, "Vector before shift: ");
    // shift vector
    shiftVectorRight(selectedNodesPerSlotBuffer);
    printVector(selectedNodesPerSlotBuffer, "Vector after shift: ");

    // Get initial node selection
    uint64_t x = easyHash(hashSeed + seqNo);

    // First numAnchor + 1 ideas are reserved for anchors (or at least those should be the anchors) and hence cannot be used as nodeId/vAnchor candidate
    int idOffset = numAnchors + 1;

    int numVAnchorCandidates = numNodes;

    EV_INFO << "Number of v-Anchor Candidates: " << numVAnchorCandidates << endl;

    int selectedNode = static_cast<int>(x % numVAnchorCandidates) + idOffset;
    int counter = 1;
    int THRESHOLD = numNodes*numVAnchorSlots; // Arbitrarily chosen...
    // Reselect Node if necessary
    while((std::find(selectedNodesPerSlotBuffer.begin(), selectedNodesPerSlotBuffer.end(),selectedNode) != selectedNodesPerSlotBuffer.end())
            && (counter < THRESHOLD)){
        ASSERT(hashSeed!=0);
        EV_INFO << "Selected node already in buffer! : " << selectedNode << lastSelectedNode << " Fixing it! " << endl;
        counter++;
        x = easyHash(hashSeed*counter+seqNo);
        selectedNode = static_cast<int>(x % numVAnchorCandidates) + idOffset;
    }

    if (counter == THRESHOLD){
        selectedNode = -1;
    }

    // Set selected Node to first element/slot
    selectedNodesPerSlotBuffer[0] = selectedNode;
    printVector(selectedNodesPerSlotBuffer, "Vector after selection: ");
    lastSelectedNode = selectedNode;
    return selectedNode;
}
void AvaApp::shiftVectorRight(std::vector<int>& vec){
    for (int i = vec.size() -1; i > 0; --i){
        vec[i] = vec[i-1];
    }
    vec[0] = -1;
}

void AvaApp::printVector(std::vector<int>& vec, std::string message){
    EV_INFO << message ;
    for (int n : vec) {
        if (n == -1)
            EV_INFO << "[empty] ";
        else
            EV_INFO << n << " ";
    }
    EV_INFO << endl;
}

void AvaApp::sendAvaPacket(int seqNo, int senderId, int destId, AvaMessageType type, int contenderId, const char *packetName, int interfaceIdx){
    auto *packet = new Packet(packetName);
    configureAvaPacket(packet, seqNo, senderId, destId, type, contenderId, interfaceIdx);
    writeToResultFile(simTime(), getClockTime(), seqNo, senderId, destId, type, interfaceIdx);
    send(packet, "socketOut");
}



void AvaApp::initializeParameters(){
    cModule *mobilityModule = getParentModule()->getSubmodule("mobility");
    if (mobilityModule){
        mobility = check_and_cast<inet::IMobility *>(mobilityModule);
    }
    // Check selected Method
    selectedMethod = parseMethodString(par("method"));
    // STRUCTURE
    twrSeqNo = 1;
    selfLocSeqNo = 0;
    numReceived = 0;
    nodeId = par("nodeId");
    startTime = par("startTime");
    startTime_c = par("startTime");
    stopTime = par("stopTime");
    stopTime_c = par("stopTime");
    t_tx_msg = par("t_tx_msg");
    t_tx_msg_c = par("t_tx_msg");
    packetLength = B(par("packetLength"));
    if (packetLength < B(5))
        throw cRuntimeError("packetLength must be at least 5 B for the AvaMsg header");
    aTurnaroundTime = par("aTurnaroundTime");
    aTurnaroundTime_c = par("aTurnaroundTime");
    recordInCsv = par("recordInCsv");
    isAnchor = par("isAnchor");
    useRescheduling = par("useRescheduling");
    useClockTime = par("useClockTime");
    numAnchors = par("numAnchors");
    numNodes = par("numNodes");
    numVAnchorSlots = par("numVAnchorSlots");
    hashSeed = par("hashSeed");
    selectedNodesPerSlotBuffer = std::vector<int>(numVAnchorSlots, -1);
}

int AvaApp::parseMethodString(const char *methodStr){
    if(std::strcmp(methodStr,"TWR")==0){
        return TWR;
    }
    else if(std::strcmp(methodStr, "DISTRIBUTED")==0){
        return DISTRIBUTED;
    }
    else if(std::strcmp(methodStr, "CENTRALIZED")==0){
        return CENTRALIZED;
    }
    else if(std::strcmp(methodStr, "HYPERBOLIC_SELF_LOCALIZATION")==0){
        return HYPERBOLIC_SELF_LOC;
    }
    throw cRuntimeError("Unknown method: %s", methodStr);
}


void AvaApp::initializeTimers(){
    pollTimer   = new cMessage("PollTimer");
    txTimer     = new cMessage("TransmitTimer");

    selfLocTxTimer      = new cMessage("selfLocTxTimer");
    succContTimer       = new cMessage("succContTimer");
    contentionTimer     = new cMessage("contentionTimer");
    selectedNodeTxTimer = new cMessage("selectedNodeTxTimer");


    pollTimer_c         = new ClockEvent("pollTimer_c");
    txTimer_c           = new ClockEvent("txTimer_c");
    selfLocTxTimer_c    = new ClockEvent("selfLocTxTimer_c");
    succContTimer_c     = new ClockEvent("succContTimer_c");
    contentionTimer_c   = new ClockEvent("contentionTimer_c");
    selectedNodeTxTimer_c = new ClockEvent("selectedNodeTxTimer_c");
}

void AvaApp::calculateDerivedParameters(){
    // Slot: maximum propagation time + transmission duration + turnaround time (no guard time)
    slotDuration = t_max_prop + t_tx_msg + aTurnaroundTime;
    slotDuration_c = t_max_prop_c + t_tx_msg_c + aTurnaroundTime_c;

}


void AvaApp::initializeLogging(){
    namespace fs = std::filesystem;
    if (!recordInCsv){
        return;
    }

    cConfigurationEx *cfg = getEnvir()->getConfigEx();
    std::string configName = cfg->getActiveConfigName();
    std::string iterationVarsF = cfg->getVariable(CFGVAR_ITERATIONVARSF);
    std::string repetition = cfg->getVariable(CFGVAR_REPETITION);

    EV_INFO << "ConfigName: " << configName << endl;
    EV_INFO << "IterationVarsF: " << iterationVarsF << endl;
    EV_INFO << "Repetition: " << repetition << endl;

    fs::path traceDir = fs::path("results") / "traces" / configName / iterationVarsF;

    try {
        fs::create_directories(traceDir);
        EV_INFO << "Created directory: " << traceDir << endl;
    } catch (const fs::filesystem_error& e) {
        throw cRuntimeError("Could not create directory %s: %s", traceDir.c_str(), e.what());
    }

    std::string moduleName = getParentModule()->getFullName();
    std::string resultFilename = moduleName +"_rep" + repetition + "_log.csv";
    fs::path logFilePath = traceDir / resultFilename;


    logFile.open(logFilePath);
    if (!logFile.is_open())
        throw cRuntimeError("Could not open log file %s", logFilePath.c_str());
    logFile <<  "sim_timestamp, "
                "clock_timestamp, "
                "noisy_timestamp, "
                "seqNo, "
                "srcId, "
                "destId, "
                "msgType, "
                "interfaceIdx, "
                "X, "
                "Y, "
                "Z, "
                "v_X, "
                "v_Y, "
                "v_Z"
                "\n";
    logFile.flush();
    EV_INFO << "Logging to: " << logFilePath << endl;
}

void AvaApp::scheduleTimers(cMessage *timer, ClockEvent *clockTimer, simtime_t delay, clocktime_t delay_c){
    if (useClockTime){
        scheduleClockEventAfter(delay_c, clockTimer);
    }
    else{
        scheduleAfter(delay, timer);
    }
}

void AvaApp::scheduleTimersAfterSlots(cMessage *timer, ClockEvent *clockTimer, int numSlots, simtime_t scheduleOffset, clocktime_t scheduleOffset_c){
//    simtime_t delay = numSlots * slotDuration + aTurnaroundTime;
//    clocktime_t delay_c = numSlots * slotDuration_c + aTurnaroundTime_c;

    simtime_t delay = numSlots * slotDuration + scheduleOffset;
    clocktime_t delay_c = numSlots * slotDuration_c + scheduleOffset_c;
    scheduleTimers(timer, clockTimer, delay, delay_c);
}




void AvaApp::finish(){
    if (logFile.is_open())
        logFile.close();
}

AvaApp::~AvaApp() {
    if (logFile.is_open())
        logFile.close();
    if(currentPkt)
        delete currentPkt;
    cancelAndDelete(txTimer);
    cancelAndDelete(selfMsg);
    cancelAndDelete(pollTimer);
    cancelAndDelete(contentionTimer);
    cancelAndDelete(selfLocTxTimer);
    cancelAndDelete(succContTimer);
    cancelAndDelete(selectedNodeTxTimer);

    cancelAndDeleteClockEvent(txTimer_c);
    cancelAndDeleteClockEvent(pollTimer_c);
    cancelAndDeleteClockEvent(contentionTimer_c);
    cancelAndDeleteClockEvent(selfLocTxTimer_c);
    cancelAndDeleteClockEvent(succContTimer_c);
    cancelAndDeleteClockEvent(selectedNodeTxTimer_c);

}

}
