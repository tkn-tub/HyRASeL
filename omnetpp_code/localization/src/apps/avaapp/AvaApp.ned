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

package localization.apps.avaapp;

import inet.visualizer.common.IntegratedMultiVisualizer;

import inet.applications.contract.IApp;
import inet.queueing.contract.IActivePacketSource;
import inet.queueing.contract.IPacketFlow;
import inet.queueing.contract.IPassivePacketSink;

import inet.clock.contract.IClock;
import inet.common.MessageDispatcher;
import inet.mobility.contract.IMobility;
import inet.networklayer.common.InterfaceTable;
import inet.linklayer.contract.IMacRelayUnit;
import inet.linklayer.contract.IMacForwardingTable;
import inet.linklayer.contract.IWirelessInterface;
import inet.linklayer.common.WirelessInterface;

@namespace(localization);

simple AvaApp like IApp
{
    parameters:
        @display("i=block/app");
        @class(AvaApp);
        int nodeId = default(-1); // FIXME: Should be part of the Node, not the app
        bool isAnchor = default(false);
        volatile int destId = default(-1); // Will likely change over course of
		double startTime @unit(s) = default(1s); // application start time (start of the first packet)
        double stopTime @unit(s) = default(-1s);  // time of finishing sending, -1s means forever
        double txInterval @unit(s) = default(1s);
        double t_tx_msg @unit(s) = default(1762129ns); // 4.436363637ms => 4436363.637ns (4000us Preamble); 1762129ns; 1336363ns for 1100us preamble
        bool recordInCsv = default(true);
        bool useClockTime = default(false);
        bool useRescheduling = default(false);
        int numAnchors = default(1);
        int numNodes = default(1);
        int numVAnchorSlots = default(2);
        string method @enum("TWR", "DISTRIBUTED", "CENTRALIZED", "HYPERBOLIC_SELF_LOCALIZATION") = default("CENTRALIZED");
        int hashSeed = default(42);



		string clockModule = default(""); // relative path of a module that implements IClock; optional
    	
    	// Time to switch radio from Rx to Tx state
        double aTurnaroundTime @unit(s) = default(0.000192 s);    // 12 symbols

        bool initiator = default(false);
		@signal[packetReceivedSignal](type=inet::Packet);
		@signal[twrMsgRecording](type="int");
//		@statistic[recordTwrMessage](title="Recording TWR"; source=twrMsgRecording; record=vector);
		@statistic[packetReceived](title="Packet received"; source=packetReceivedSignal; record=stats,vector);
		@statistic[twrMsgRecording](title="Recording TWR"; source=twrMsgRecording; record=vector);
//		@lifecycleSupport;
		
		double stopOperationExtraTime @unit(s) = default(-1s);    // extra time after lifecycle stop operation finished
        double stopOperationTimeout @unit(s) = default(2s);    // timeout value for lifecycle stop operation

    gates:
        input socketIn;  // @loose @labels(TwrControlInfo/up)
        output socketOut; // @loose @labels(TwrControlInfo/down)


}