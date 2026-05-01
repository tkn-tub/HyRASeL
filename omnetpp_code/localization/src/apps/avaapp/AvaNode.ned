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

module AvaNode
{
    parameters:
        @networkNode();
        @display("i=device/pc2");
        *.interfaceTableModule = default(absPath(".interfaceTable"));
        wlan*.radio.antenna.mobilityModule = default("^.^.^.mobility");

        // Help-Variable for easier network config FIXME: Remove
        int numAnchors = default(3);
        int numNodes = default(3);

    gates:
        input radioIn[1] @directIn;


    submodules:
        interfaceTable: InterfaceTable {
            @display("p=100,200;is=s");
        }
        mobility: <default("StationaryMobility")> like IMobility {
            @display("p=100,400;is=s");
        }
        wlan: <default("WirelessInterface")> like IWirelessInterface { //TWR Interface
            parameters:
                @display("p=375,490,row,150;q=queue");
                mac.typename = default("SimpleMac");
                radio.centerFrequency = default(3494.4 MHz);
                radio.bandwidth = default(499.2 MHz);
        }
        app: AvaApp {
            @display("p=375,75,row,150");
        }
        clock: <default("OscillatorBasedClock")> like IClock {
            @display("p=100,75");
        }
    connections:
        app.socketOut --> wlan.upperLayerIn;
        app.socketIn <-- wlan.upperLayerOut;
        radioIn[0] --> { @display("m=s"); } --> wlan.radioIn;


}
