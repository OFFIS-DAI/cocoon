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

package networks;

import ned.DatarateChannel;
import inet.node.ethernet.Eth10G;

channel C10kbps   extends DatarateChannel { datarate = 10kbps;   delay = 1ms; }
channel C100kbps  extends DatarateChannel { datarate = 100kbps;  delay = 1ms; }
channel C1Mbps    extends DatarateChannel { datarate = 1Mbps;    delay = 1ms; }
channel C10Mbps   extends DatarateChannel { datarate = 10Mbps;   delay = 1ms; }
channel C100Mbps  extends DatarateChannel { datarate = 100Mbps;  delay = 1ms; }
channel C1Gbps    extends DatarateChannel { datarate = 1Gbps;    delay = 1ms; }
