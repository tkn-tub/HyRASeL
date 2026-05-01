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

#ifndef INET_LINKLAYER_SIMPLEMAC_SIMPLEMAC_H_
#define INET_LINKLAYER_SIMPLEMAC_SIMPLEMAC_H_

#include "inet/common/packet/Packet.h"
#include "inet/linklayer/base/MacProtocolBase.h"
#include "inet/linklayer/contract/IMacProtocol.h"
#include "inet/physicallayer/wireless/common/contract/packetlevel/IRadio.h"
#include "inet/queueing/contract/IActivePacketSink.h"
#include "inet/queueing/contract/IPacketQueue.h"

using namespace inet;
namespace localization {

class INET_API SimpleMac : public MacProtocolBase, public IMacProtocol , public queueing::IActivePacketSink{

    protected:
        double bitrate = NaN;
        B headerLength = B(-1);

        ModuleRefByPar<physicallayer::IRadio> radio;
        physicallayer::IRadio::TransmissionState transmissionState = physicallayer::IRadio::TRANSMISSION_STATE_UNDEFINED;


    protected:
        virtual void initialize(int stage) override; //
        virtual void finish() override; //
        virtual void configureNetworkInterface() override; //

        virtual void handleSelfMessage(cMessage *msg) override; //
        virtual void handleUpperPacket(Packet *packet) override; //
        virtual void handleLowerPacket(Packet *packet) override; //
//        virtual void handleWithFsm(cMessage *msg);

        virtual void receiveSignal(cComponent *source, simsignal_t signalID, intval_t value, cObject *details) override; //

        virtual void encapsulate(Packet *frame); //
        virtual void decapsulate(Packet *frame); //


        virtual void sendDataFrame(Packet *frameToSend); //

        virtual void handleStopOperation(LifecycleOperation *operation) override; //
        virtual void handleCrashOperation(LifecycleOperation *operation) override; //

        virtual void processUpperPacket(); //

        virtual bool isMediumFree(); //
        virtual bool isReceiving(); //

    public:
      // IActivePacketSink:
      virtual queueing::IPassivePacketSource *getProvider(cGate *gate) override;
      virtual void handleCanPullPacketChanged(cGate *gate) override;
      virtual void handlePullPacketProcessed(Packet *packet, cGate *gate, bool successful) override;


    public:
        virtual ~SimpleMac();
};

} /* namespace inet */

#endif /* INET_LINKLAYER_SIMPLEMAC_SIMPLEMAC_H_ */
