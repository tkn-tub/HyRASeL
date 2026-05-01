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
#include "AvaMsg_m.h"

#include "inet/common/ModuleAccess.h"
#include "inet/common/lifecycle/ModuleOperations.h"

#include "inet/common/IProtocolRegistrationListener.h"

#include <cstring>
#include <filesystem>

using namespace inet;

namespace localization{

Define_Module(AvaApp);

simsignal_t AvaApp::packetReceivedSignal = registerSignal("packetReceived");
simsignal_t AvaApp::avaMsgRecording = registerSignal("avaMsgRecording");
simsignal_t AvaApp::rxStartedSignal = registerSignal("receptionStarted");

void AvaApp::initialize(int stage)
{
    ApplicationBase::initialize(stage);

    // Add Clock Module
    ClockUserModuleMixin::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        initializeParameters();
//        initializeTimers();
        calculateDerivedParameters();
    }
    else if (stage == INITSTAGE_APPLICATION_LAYER) {
            initializeTimers();
            initializeLogging();
            initializeLocalization();
            initializeRxTimestamping();
        }
}

void AvaApp::initializeRxTimestamping(){
//    simsignal_t relSig = registerSignal("receptionStarted");
//    rxStartedSignal = registerSignal("receptionStarted");
    radioModule = getModuleByPath("^.wlan.radio");
    if (radioModule) {
        // 3. Beim Radio für dieses Signal abonnieren
        radioModule->subscribe(rxStartedSignal, this);
    }
}


void AvaApp::receiveSignal(cComponent *src, simsignal_t id, cObject *value, cObject *details) {
    // Sicherstellen, dass es das richtige Signal ist
//    static simsignal_t relSig = registerSignal("receptionStarted");

    if (id == rxStartedSignal) {

        // Assert may be removed, I just put it to see, whether interference or so is going to cause trouble
        EV_INFO << "Before sim: " << rxTimestamp << "; Before clock: "<<rxTimestamp_c << endl;
        ASSERT((rxTimestamp_c.isZero()) && (rxTimestamp.isZero()));
        rxTimestamp_c = getClockTime();
        rxTimestamp   = simTime();

        EV_INFO << "Signal empfangen von " << src->getFullPath()
                << " - Local Rx-Timestamp: " << rxTimestamp
                << " - local Rx-Timestamp_C: " << rxTimestamp_c
                << endl;

        // Hier können Sie rxTime in Ihren Statistiken oder Logs speichern
    }
}

void AvaApp::initializeDistributedSelectionSelfLoc(){
    totalNumSlots = numAnchors + numVAnchorSlots;
    if (isAnchor){
        scheduleTimersAfterSlots(selfLocTxTimer, selfLocTxTimer_c , nodeId, true);
    }
    else{
        scheduleTimersAfterSlots(contentionTimer, contentionTimer_c, numAnchors+1, true);
    }

}

void AvaApp::initializeCentralizedSelectionSelfLoc(){
    totalNumSlots = numAnchors + numVAnchorSlots;
    if (isAnchor){
        scheduleTimersAfterSlots(selfLocTxTimer, selfLocTxTimer_c , nodeId, true);
    }
}

void AvaApp::initializeStandardSelfLoc(){
    totalNumSlots = numAnchors;
    if (isAnchor){
        scheduleTimersAfterSlots(selfLocTxTimer, selfLocTxTimer_c , nodeId, true);
    }
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



void AvaApp::initializeTwr(){
    if(isAnchor){
        return;
    }
    int startPosition = nodeId - numAnchors;
    scheduleTimersAfterSlots(pollTimer, pollTimer_c, (startPosition*(2+numAnchors)), true);
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
    }

    // CONTENTION TIMER
    else if (msg == contentionTimer || msg==contentionTimer_c){
        handleContentionTimer();
    }

    // SUCCESSFUL CONTENTION TIMER
    else if (msg == succContTimer || msg==succContTimer_c){
        handleSuccessfulContentionTimer();
    }

    // CENTRALIZED SELECTION
    else if (msg == selectedNodeTxTimer || msg == selectedNodeTxTimer_c){
        handleSelectedNodeTxTimer();
    }

    // TWR MESSAGES
    else if (msg == pollTimer || msg == pollTimer_c){
        handlePollTimer();
    }
    else if(msg == txTimer || msg == txTimer_c) {
        handleTxTimer();
    }
}


void AvaApp::handlePacket(Packet *pkt){
//    emit(packetReceivedSignal, pkt);
    if (!isForMe(pkt)){
        EV_INFO << "Packet not for me -> Dropping it." << endl;
        delete pkt;
        return;
    }
    AvaPacketInfo info = extractAvaPacketInfo(pkt);
    // Receiving message; writing it into file
    writeToResultFile(rxTimestamp, rxTimestamp_c, info.seqNo, info.srcId, info.destId, info.msgType);
    EV_INFO << "Received AvaMsg of type " << msgNames[info.msgType] << " with SeqNo " << info.seqNo << " from " << info.srcId << endl;

    // Reset timers
    rxTimestamp   = SIMTIME_ZERO;
    rxTimestamp_c = SIMTIME_AS_CLOCKTIME(SIMTIME_ZERO);

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

    EV_INFO << "Handled Packet -> Deleting it" << endl;
    delete pkt;

}


void AvaApp::handleStandardSelfLocPacket(){
    // TODO
}


void AvaApp::handleCentralizedSelectionSelfLocPacket(const AvaPacketInfo& info){
    // ANCHOR LOGIC
    if (isAnchor){
        return;
    }

    // NODE LOGIC
    else{
        updateSequenceNumberCounter(info); // Technically only relevant for node, but both can do it; Relevant for all HyRaSeL Methods
        updateLocalSequenceNumber(info);
        bool selectedNodeTxTimerIsScheduled = (selectedNodeTxTimer_c->isScheduled() || selectedNodeTxTimer->isScheduled());
        if (info.srcIsAnchor && (nodeId == info.contId) && !selectedNodeTxTimerIsScheduled){
            // SCHEDULE selectedNodeTxTimer
            EV_INFO << "Selected as node! Scheduling transmission" << endl;
            int waitingSlots = numAnchors - info.srcId;
            scheduleTimersAfterSlots(selectedNodeTxTimer, selectedNodeTxTimer_c, waitingSlots);
        }
    }
}


void AvaApp::handleDistributedSelectionSelfLocPacket(const AvaPacketInfo& info){
    // ANCHOR LOGIC
    if (isAnchor){
        updateContendingNode(info);
        return;
    }

    // NODE LOGIC
    else{
        updateSequenceNumberCounter(info); // Technically only relevant for node, but both can do it; Relevant for all HyRaSeL Methods
        updateLocalSequenceNumber(info);
        bool srcOverheardOtherNode = (info.contId !=nodeId);
        bool contentionWasSuccessful = (vAnchorTxCounter != 1);
        bool succContTimerIsScheduled = (succContTimer->isScheduled() || succContTimer_c->isScheduled());
        // Not eligible for further transmission if some node overheard another node
        // Timer must be scheduled, transmitter must be anchor and contentionId is neither our ID nor -1, than we cancel
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
}

void AvaApp::handleTwrPacket(Packet *pkt, const AvaPacketInfo& info){
    // ANCHOR LOGIC
    if (isAnchor){
        if ((info.msgType == POLL)){
            int numSlots = nodeId-1;
            currentPkt = pkt->dup();
            scheduleTimersAfterSlots(txTimer, txTimer_c, numSlots);
        }
    }

    // NODE LOGIC
    else{
        updateLocalSequenceNumber(info); // TODO: check, weather this is actually necessary
        bool txTimerScheduled = txTimer->isScheduled() || txTimer_c->isScheduled();
        if (!txTimerScheduled && (info.msgType == RESPONSE)){
            // TODO: Vllt reicht es ja, wenn ich nach dem Senden des POLL einfach direkt das FINAL Schedule... Oder besser nicht -> Potentiell unnötige Nachrichten, wenn es kollidieren sollte
            // Ich sollte die Message mit dem niedrigsten Index zuerst erhalten. Ich muss wissen, wieviele Nachrichten noch folgen sollten:
            int numSlots = numAnchors - info.srcId;
            currentPkt = pkt->dup();
            scheduleTimersAfterSlots(txTimer, txTimer_c, numSlots);
        }
    }
}

void AvaApp::updateSequenceNumberCounter(const AvaPacketInfo& info){
    // Reset if sequenceNumber changed
        // -> SeqNoCounter -> Counts how many messages of a given sequence number where overheard, to check, whether sufficiently many anchors got overheard
    if (info.seqNo != prevSeqNo && info.srcIsAnchor){
        prevSeqNo = info.seqNo;
        seqNoCounter = 0;
    }
    seqNoCounter++;
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
    // FIXME: The rescheduling really does not work properly! (Sometimes, the rescheduling is still skewed by 1 aTurnaroundTime_c)
    // FIXME/TODO: We are not allowed to turn clock back - it should at most set to future, not back I think.
    // TODO: The transmission duration is hardcoded and used as a parameter. This needs to be improved and be done better. Maybe go back to the beginning of the slot and use that as a reference for the slot.
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
                : ((vAnchorTxCounter - srcVAnchorSlot -1 + totalNumSlots) % totalNumSlots)); // Stupid C++ % implementation requires me to add totalNumSlots...
        rescheduleClockTimer(succContTimer_c, remainingSlots);
    }
    // Reschedule selectedNodeTxTimer
    // Reschedule pollTimers etc.
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
    //TODO: Change for a more elegant solution, probably other message type! Currently, vAnchorTxCounter is transmitted to indicate that we are not in the contentionSlot
    // NOTE: It is implemented in a way that contId 4 is never send basically
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
    // TODO: Change to timeout Timer - default case: Transmission of FINAL triggers txTimer
        // If messages are lost, FINAL will not be sent -> We need a Timeout for recovery
        // NOTE: Needs to be even changed even further, when we have more than one node need to be localized by Ava
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



void AvaApp::configureAvaPacket(Packet *packet, int seqNo, int senderId, int destinationId, AvaMessageType msgType, int contenderId){

    const auto &payload = makeShared<AvaMsg>();
    payload->setSeqNo(seqNo);
    payload->setSenderId(senderId);
    payload->setReceiverId(destinationId); //Broadcast
    payload->setMsgType(msgType);
    payload->setContenderId(contenderId);
    payload->setChunkLength(B(5));
    packet->insertAtBack(payload);

    // Necessary for Radio Module
    auto packetProtocolTag = packet->addTag<PacketProtocolTag>();
    packetProtocolTag->setProtocol(&Protocol::unknown);

}

void AvaApp::writeToResultFile(simtime_t simTimestamp, clocktime_t clockTimestamp, int seqNo, int srcId, int destId, int msgType){
    if (!recordInCsv){
        return;
    }
    const Coord& pos = mobility->getCurrentPosition();
    const Coord& vel = mobility->getCurrentVelocity();

    EV_INFO << "Writing To File: clockTime: " << getClockTime() << "; simtime: " << simTime() << endl;

    logFile << simTimestamp
            << "," << clockTimestamp
            << "," << seqNo
            << "," << srcId
            << "," << destId
            << "," << msgType
            << "," << pos.getX()
            << "," << pos.getY()
            << "," << pos.getZ()
            << "," << vel.getX()
            << "," << vel.getY()
            << "," << vel.getZ()
            << "\n";
}


void AvaApp::handleStartOperation(LifecycleOperation *operation){

}

void AvaApp::handleStopOperation(LifecycleOperation *operation){
    delayActiveOperationFinish(par("stopOperationTimeout"));
}

void AvaApp::handleCrashOperation(LifecycleOperation *operation){

}

// AI Generated hash function (bases on SplitMix64)
uint64_t AvaApp::easyHash(uint64_t x){
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
    int selectedNode = static_cast<int>(x % numNodes) + numAnchors + 1;
    int counter = 1;
    int THRESHOLD = numNodes*numVAnchorSlots; // Arbitrarily chosen...
    // Reselect Node if necessary
    while((std::find(selectedNodesPerSlotBuffer.begin(), selectedNodesPerSlotBuffer.end(),selectedNode) != selectedNodesPerSlotBuffer.end())
            && (counter < THRESHOLD)){
        ASSERT(hashSeed!=0);
        EV_INFO << "Selected node already in buffer! : " << selectedNode << lastSelectedNode << " Fixing it! " << endl;
        counter++;
        x = easyHash(hashSeed*counter+seqNo);
        selectedNode = static_cast<int>(x % numNodes) + numAnchors + 1;
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
    for (int i = vec.size() -1; i > 0; --i)
        vec[i] = vec[i-1];
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

void AvaApp::sendAvaPacket(int seqNo, int senderId, int destId, AvaMessageType type, int contenderId, const char *packetName){
    auto *packet = new Packet(packetName);
    configureAvaPacket(packet, seqNo, senderId, destId, type, contenderId);
    writeToResultFile(simTime(), getClockTime(), seqNo, senderId, destId, type);
    send(packet, "socketOut");
}

void AvaApp::initializeParameters(){
    //

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
    logFile << "sim_timestamp, clock_timestamp, seqNo, srcId, destId, msgType, X, Y, Z, v_X, v_Y, v_Z\n";
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

void AvaApp::scheduleTimersAfterSlots(cMessage *timer, ClockEvent *clockTimer, int numSlots, bool start){
    simtime_t delay = numSlots * slotDuration + aTurnaroundTime;
    clocktime_t delay_c = numSlots * slotDuration_c + aTurnaroundTime_c;
    if (start){
        delay = delay + startTime;
        delay_c = delay_c + startTime_c;
    }
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


//    cancelAndDelete(activeOperationTimeout);
//    cancelAndDelete(activeOperationExtraTimer);
//    activeOperationTimeout;
//    if (activeOperationTimeout) {
//            cancelAndDelete(activeOperationTimeout);
//            activeOperationTimeout = nullptr;
//        }
//    if (activeOperationExtraTimer) {
//        cancelAndDelete(activeOperationExtraTimer);
//        activeOperationExtraTimer = nullptr;
//    }

}

}

