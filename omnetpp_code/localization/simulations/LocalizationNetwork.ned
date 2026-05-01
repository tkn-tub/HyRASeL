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

package localization.simulations;

import inet.visualizer.common.IntegratedMultiVisualizer;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.physicallayer.wireless.unitdisk.UnitDiskRadioMedium;
import inet.physicallayer.wireless.apsk.packetlevel.ApskScalarRadioMedium;
import inet.physicallayer.wireless.common.contract.packetlevel.IRadioMedium;

import inet.visualizer.contract.IIntegratedVisualizer;

import localization.apps.avaapp.AvaApp;
import localization.apps.avaapp.AvaNode;

network BaseNetwork
{
    parameters:
        @display("bgb=650,562");
    submodules:
        visualizer: <default(firstAvailableOrEmpty("IntegratedCanvasVisualizer"))> like IIntegratedVisualizer if typename != "" {
            parameters:
                @display("p=100,300;is=s");
        }
        configurator: Ipv4NetworkConfigurator {
            parameters:
                @display("p=100,100;is=s");
        }

    	radioMedium: <default("ApskScalarRadioMedium")> like IRadioMedium {
            parameters:
                @display("p=100,200;is=s");
        }
}


network HyraselNetwork extends BaseNetwork
{
    parameters:
        @display("bgb=650,650");
        int numAnchors = default(4);
        int numNodes = default(9);
    submodules:
        anchor[numAnchors]: AvaNode {
            parameters:
                app.isAnchor = true;
                @display("r=,,#707070;i=misc/sensorgateway");
        }

        node[numNodes]: AvaNode {
            parameters:
                app.isAnchor = false;
                @display("r=,,#707070;i=misc/sensor");
        }


}

