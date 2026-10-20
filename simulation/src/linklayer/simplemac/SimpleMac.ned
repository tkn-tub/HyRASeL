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

package localization.linklayer.simplemac;

import inet.linklayer.base.MacProtocolBase;
import inet.linklayer.contract.IMacProtocol;

// Minimal MAC without medium access control: passes frames between the app and
// the radio and drops frames with bit errors. Channel access is done by the app.

@namespace(localization);

module SimpleMac extends MacProtocolBase like IMacProtocol
{
    parameters:
        // length of MAC header
		string address @mutable = default("auto"); // MAC address as hex string (12 hex digits), or
                                                   // "auto". "auto" values will be replaced by
                                                   // a generated MAC address in init stage 0.
        int headerLength @unit(b) = default(72 b);
        double bitrate @unit(bps) = default(250000 bps);

        string radioModule = default("^.radio");   // The path to the Radio module

        @class(SimpleMac);
        @signal[linkBroken](type=inet::Packet);

}