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

#ifndef AVAAPP_AVAAPP_H_
#define AVAAPP_AVAAPP_H_

#include <vector>
#include "inet/applications/base/ApplicationBase.h"
#include "inet/common/clock/ClockUserModuleMixin.h"
#include "AvaMsg_m.h"

#include "inet/mobility/contract/IMobility.h"
#include "inet/common/Protocol.h"
#include "../../common/AvaProtocol.h"

#include <fstream>
#include <unordered_set>

using namespace inet;
namespace localization{

class INET_API AvaApp : public ClockUserModuleMixin<ApplicationBase> {

private:
    std::ofstream logFile;

protected:

    enum AvaMethod {
        TWR,
        DISTRIBUTED,
        CENTRALIZED,
        HYPERBOLIC_SELF_LOC
    };

    struct AvaPacketInfo{
        int seqNo, srcId, destId, msgType, contId, interfaceIdx;
        bool srcIsAnchor;
    };

    // Parameters
    simtime_t startTime;
    clocktime_t startTime_c;
    simtime_t stopTime;
    clocktime_t stopTime_c;

    static simsignal_t packetReceivedSignal;
    static simsignal_t avaMsgRecording;

    simtime_t aTurnaroundTime;
    clocktime_t aTurnaroundTime_c;

    int nodeId = -1;
    bool isAnchor = false;
    bool recordInCsv = false;
    bool useClockTime = false;
    bool useRescheduling = false;
    int twrSeqNo = 1;
    int lastFinalTwrSeqNo = -1; // TWR round (POLL seqNo) for which a FINAL was already scheduled
    int selfLocSeqNo = 0;
    int maxRange = 300; //m
    double c = 3e8;
    simtime_t t_max_prop = maxRange/c;
    simtime_t t_tx_msg; // Duration of a transmission (parameter t_tx_msg)
    b packetLength = B(5);
    simtime_t slotDuration = -1;

    clocktime_t t_max_prop_c = maxRange/c;
    clocktime_t t_tx_msg_c;
    clocktime_t slotDuration_c;
    int numAnchors = -1;
    int numNodes = -1;
    int numVAnchorSlots = -1;
    int totalNumSlots = -1;
    int contendingNode = -1;
    int vAnchorTxCounter = -1;
    int seqNoCounter = -1;
    int prevSeqNo = -1;
    int selectedMethod = -1;
    int lastSelectedNode = -1;

    std::unordered_set<int> heardAnchors;

    // Per-interface timestamps are populated from ReceptionTimingInd only
    // after the PHY and MAC have accepted the corresponding packet.
    simtime_t rxTimestamp;
    std::vector<simtime_t> rxTimestamps;

    clocktime_t rxTimestamp_c;
    std::vector<clocktime_t> rxTimestamps_c;

    static inline const char* msgNames[] = {"POLL", "RESPONSE", "FINAL", "SELF_LOC"};



    // statistics
    int numSent = 0;
    int numReceived = 0;

    cMessage *selfMsg = nullptr;


    cMessage *pollTimer = nullptr;
    cMessage *txTimer = nullptr; // TWR: delayed response/final transmission

    cMessage *selfLocTxTimer = nullptr;
    cMessage *contentionTimer = nullptr;
    cMessage *succContTimer = nullptr;
    cMessage *selectedNodeTxTimer = nullptr;


    // Clock Events
    ClockEvent *pollTimer_c = nullptr;
    ClockEvent *txTimer_c = nullptr;

    ClockEvent *selfLocTxTimer_c =nullptr;
    ClockEvent *contentionTimer_c = nullptr;
    ClockEvent *succContTimer_c = nullptr;
    ClockEvent *selectedNodeTxTimer_c = nullptr;

    //
    inet::IMobility *mobility = nullptr;


    // Packet Buffer
    Packet *currentPkt = nullptr;


    std::vector<int> selectedNodesPerSlotBuffer;
    uint64_t hashSeed;
    static uint64_t easyHash(uint64_t x);
    int selectNode(int seqNo);
    void shiftVectorRight(std::vector<int>& vec);
    void printVector(std::vector<int>& vec, std::string message);


protected:

    virtual int numInitStages() const override { return NUM_INIT_STAGES; }
    // OMNeT++ / INET hooks
    virtual void initialize(int stage) override;
    virtual void handleMessageWhenUp(cMessage *msg) override;

    virtual void finish() override;

    virtual void handleStartOperation(LifecycleOperation *operation) override;
    virtual void handleStopOperation(LifecycleOperation *operation) override;
    virtual void handleCrashOperation(LifecycleOperation *operation) override;

    // own logic

    // handle Timer/Messages
    void handleSelfMessage(cMessage *msg);
    void handleSelfLocTxTimer();
    void handleContentionTimer();
    void handleSuccessfulContentionTimer();
    void handleSelectedNodeTxTimer();
    void handlePollTimer();
    void handleTxTimer();
    bool overheardSufficientAnchors();
    bool passedProbabilityCheck();


    void handlePacket(Packet *pkt);
    void handleSelfLocPacket(Packet *pkt);
    bool isForMe(Packet *pkt);
    void configureAvaPacket(Packet *packet, int seqNo, int senderId, int destinationId, AvaMessageType msgType, int contenderId=-1, int interfaceIdx=0);
    void writeToResultFile(simtime_t simTimestamp, clocktime_t clockTimestamp, int seqNo, int srcId, int destId, int msgType, int interfaceIdx=0);
    void sendAvaPacket(int seqNo, int senderId, int destId, AvaMessageType type, int contenderId=-1, const char *packetName = "SELF_LOC", int interfaceIdx=0);

    void initializeParameters();
    void initializeTimers();
    void initializeLogging();
    void initializeRxTimestamping();

    void initializeLocalization();
    void initializeTwr();
    void initializeStandardSelfLoc();
    void initializeDistributedSelectionSelfLoc();
    void initializeCentralizedSelectionSelfLoc();



    void calculateDerivedParameters();
    void scheduleTimers(cMessage *timer, ClockEvent *clockTimer, simtime_t delay, clocktime_t delay_c);
    void scheduleTimersAfterSlots(cMessage *timer, ClockEvent *clockTimer, int numSlots, simtime_t scheduleOffset = 0, clocktime_t scheduleOffset_c = 0);
    void rescheduleClockTimer(ClockEvent *clockTimer, int numSlots);
    void rescheduleAllClocksIfNeeded(const AvaPacketInfo& info);
    void updateLocalSequenceNumber(const AvaPacketInfo& info);
    void updateContendingNode(const AvaPacketInfo& info);
    void updateSequenceNumberCounter(const AvaPacketInfo& info);


    // Select between Options
//    void handlePacketAsNode();
//    void handlePacketAsAnchor();

    void handleTwrPacket(Packet *pkt, const AvaPacketInfo& info);
    void handleCentralizedSelectionSelfLocPacket(const AvaPacketInfo& info);
    void handleDistributedSelectionSelfLocPacket(const AvaPacketInfo& info);
    void handleStandardSelfLocPacket();


    AvaPacketInfo extractAvaPacketInfo(Packet *pkt);
    int parseMethodString(const char *methodStr);


public:

    AvaApp() {}
    ~AvaApp();
};

}

#endif /* AvaAPP_AvaAPP_H_ */
