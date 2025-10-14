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

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.networklayer.ipv4.RoutingTableRecorder;
import inet.node.ethernet.Eth10G;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import MangoTimeAdvancer;


network EvaluationNetworkEthernet_50
{
    parameters:
        @display("i=block/network2;bgb=400, 400");
    submodules:
        
        timeAdvancer: MangoTimeAdvancer {
                @display("p=376.42242,21.847977");
        }
        
        router_central: Router {
                @display("p=321.8025,53.70961");
        }
        configurator: Ipv4NetworkConfigurator {
                @display("p=376.42242,56.89577");
        }
        router_0: Router {
                @display("p=196.63179,223.48659");
        }
        router_1: Router {
                @display("p=127.44653,64.6336");
        }
        node0: StandardHost {
                // central entity 
                @display("p=330,330");
        }
        node1: StandardHost {
                //  Buchenweg 2 
                @display("p= 136.19808178127278 , 266.7808129210025 ");
        }
        node2: StandardHost {
                //  Buchenweg 4 
                @display("p= 135.34602346195607 , 258.4243683256209 ");
        }
        node3: StandardHost {
                //  Buchenweg 1 
                @display("p= 109.55600467708427 , 279.1108704926446 ");
        }
        node4: StandardHost {
                //  Buchenweg 3 
                @display("p= 107.35014683264308 , 263.0050863940269 ");
        }
        node5: StandardHost {
                //  Buchenweg 5 
                @display("p= 103.98684812395368 , 247.6105312211439 ");
        }
        node6: StandardHost {
                //  Buchenweg 6 
                @display("p= 134.07065122557105 , 236.87388666719198 ");
        }
        node7: StandardHost {
                //  Buchenweg 7 
                @display("p= 109.70507018500939 , 231.05967176519334 ");
        }
        node8: StandardHost {
                //  Buchenweg 9 
                @display("p= 106.0909309007111 , 215.9685475854203 ");
        }
        node9: StandardHost {
                //  Buchenweg 11a 
                @display("p= 100.0 , 190.81692989636213 ");
        }
        node10: StandardHost {
                //  Buchenweg 11 
                @display("p= 116.87517556815874 , 193.14509646128863 ");
        }
        node11: StandardHost {
                //  Buchenweg 8 
                @display("p= 141.10889204329578 , 212.15646655112505 ");
        }
        node12: StandardHost {
                //  Buchenweg 13 
                @display("p= 124.6468965325621 , 173.46035425085574 ");
        }
        node13: StandardHost {
                //  Buchenweg 15 
                @display("p= 140.71447496837936 , 156.75369920022786 ");
        }
        node14: StandardHost {
                //  Buchenweg 17 
                @display("p= 158.76687453780323 , 143.00827796570957 ");
        }
        node15: StandardHost {
                //  Buchenweg 19 
                @display("p= 173.0003695912892 , 142.22780754975975 ");
        }
        node16: StandardHost {
                //  Buchenweg 21 
                @display("p= 187.83278567378875 , 140.1346326135099 ");
        }
        node17: StandardHost {
                //  Buchenweg 23 
                @display("p= 203.3587229428813 , 135.51941210310906 ");
        }
        node18: StandardHost {
                //  Buchenweg 25 
                @display("p= 217.60022531513823 , 136.25009013991803 ");
        }
        node19: StandardHost {
                //  Buchenweg 25a 
                @display("p= 214.6824416151503 , 100.0 ");
        }
        node20: StandardHost {
                //  Buchenweg 27 
                @display("p= 231.66422947729006 , 141.6156514696777 ");
        }
        node21: StandardHost {
                //  Buchenweg 29 
                @display("p= 246.33032117126277 , 146.27291464153677 ");
        }
        node22: StandardHost {
                //  Buchenweg 29a 
                @display("p= 249.82400997559307 , 138.59848055802286 ");
        }
        node23: StandardHost {
                //  Buchenweg 31 
                @display("p= 264.681847693515 , 160.48077862523496 ");
        }
        node24: StandardHost {
                //  Buchenweg 33 
                @display("p= 283.37649809534196 , 182.3428132813424 ");
        }
        node25: StandardHost {
                //  Buchenweg 35 
                @display("p= 295.98880094324704 , 199.7037243079394 ");
        }
        node26: StandardHost {
                //  Buchenweg 37 
                @display("p= 300.5195698341704 , 216.40198030788451 ");
        }
        node27: StandardHost {
                //  Buchenweg 39 
                @display("p= 304.4349385824753 , 231.29018875397742 ");
        }
        node28: StandardHost {
                //  Buchenweg 41 
                @display("p= 307.13492097333074 , 245.4796044435352 ");
        }
        node29: StandardHost {
                //  Buchenweg 10 
                @display("p= 147.43338752392447 , 195.65268810186535 ");
        }
        node30: StandardHost {
                //  Buchenweg 12 
                @display("p= 161.3608236007858 , 175.23027331009507 ");
        }
        node31: StandardHost {
                //  Buchenweg 14 
                @display("p= 187.3803180376999 , 169.1492010038346 ");
        }
        node32: StandardHost {
                //  Buchenweg 16 
                @display("p= 219.28502094047144 , 168.4769586417824 ");
        }
        node33: StandardHost {
                //  Buchenweg 18 
                @display("p= 242.42795590654714 , 172.18280640617013 ");
        }
        node34: StandardHost {
                //  Buchenweg 20 
                @display("p= 261.9007081936579 , 188.39938639476895 ");
        }
        node35: StandardHost {
                //  Buchenweg 22 
                @display("p= 275.12898854282685 , 207.67094972450286 ");
        }
        node36: StandardHost {
                //  Buchenweg 24 
                @display("p= 277.7864681102801 , 232.94151109457016 ");
        }
        node37: StandardHost {
                //  Buchenweg 26 
                @display("p= 279.5533846797771 , 242.5020223800093 ");
        }
        node38: StandardHost {
                //  GEN-WIND-1 
                @display("p= 181.0 , 22.0 ");
        }
        node39: StandardHost {
                //  GEN-PV-1 
                @display("p= 167.0 , 65.0 ");
        }
        node40: StandardHost {
                //  GEN-CHP-1 
                @display("p= 220.0 , 84.0 ");
        }
        node41: StandardHost {
                //  GEN-WIND-2 
                @display("p= 100.0 , 26.0 ");
        }
        node42: StandardHost {
                //  GEN-PV-2 
                @display("p= 99.0 , 75.0 ");
        }
        node43: StandardHost {
                //  GEN-BESS-1 
                @display("p= 148.0 , 29.0 ");
        }
        node44: StandardHost {
                //  LOAD-IND-1 
                @display("p= 23.0 , 88.0 ");
        }
        node45: StandardHost {
                //  LOAD-DATA-1 
                @display("p= 248.0 , 54.0 ");
        }
        node46: StandardHost {
                //  LOAD-TRAM-1 
                @display("p= 236.0 , 83.0 ");
        }
        node47: StandardHost {
                //  LOAD-EV-1 
                @display("p= 203.0 , 67.0 ");
        }
        node48: StandardHost {
                //  LOAD-RES-1 
                @display("p= 134.0 , 6.0 ");
        }
        node49: StandardHost {
                //  LOAD-COM-1 
                @display("p= 79.0 , 65.0 ");
        }
        
    connections:
        router_0.pppg++ <--> Eth10G <--> router_central.pppg++;
        router_1.pppg++ <--> Eth10G <--> router_central.pppg++;
        node0.pppg ++ <--> Eth10G <--> router_central.pppg++;
        
        router_0.pppg++ <--> C100kbps <--> node1.pppg++;
        router_0.pppg++ <--> C100kbps <--> node2.pppg++;
        router_0.pppg++ <--> C100kbps <--> node3.pppg++;
        router_0.pppg++ <--> C100kbps <--> node4.pppg++;
        router_0.pppg++ <--> C100kbps <--> node5.pppg++;
        router_0.pppg++ <--> C100kbps <--> node6.pppg++;
        router_0.pppg++ <--> C100kbps <--> node7.pppg++;
        router_0.pppg++ <--> C100kbps <--> node8.pppg++;
        router_0.pppg++ <--> C100kbps <--> node9.pppg++;
        router_0.pppg++ <--> C100kbps <--> node10.pppg++;
        router_0.pppg++ <--> C100kbps <--> node11.pppg++;
        router_0.pppg++ <--> C100kbps <--> node12.pppg++;
        router_0.pppg++ <--> C100kbps <--> node13.pppg++;
        router_0.pppg++ <--> C100kbps <--> node14.pppg++;
        router_0.pppg++ <--> C100kbps <--> node15.pppg++;
        router_0.pppg++ <--> C100kbps <--> node16.pppg++;
        router_0.pppg++ <--> C100kbps <--> node17.pppg++;
        router_0.pppg++ <--> C100kbps <--> node18.pppg++;
        router_0.pppg++ <--> C100kbps <--> node19.pppg++;
        router_0.pppg++ <--> C100kbps <--> node20.pppg++;
        router_0.pppg++ <--> C100kbps <--> node21.pppg++;
        router_0.pppg++ <--> C100kbps <--> node22.pppg++;
        router_0.pppg++ <--> C100kbps <--> node23.pppg++;
        router_0.pppg++ <--> C100kbps <--> node24.pppg++;
        router_0.pppg++ <--> C100kbps <--> node25.pppg++;
        router_0.pppg++ <--> C100kbps <--> node26.pppg++;
        router_0.pppg++ <--> C100kbps <--> node27.pppg++;
        router_0.pppg++ <--> C100kbps <--> node28.pppg++;
        router_0.pppg++ <--> C100kbps <--> node29.pppg++;
        router_0.pppg++ <--> C100kbps <--> node30.pppg++;
        router_0.pppg++ <--> C100kbps <--> node31.pppg++;
        router_0.pppg++ <--> C100kbps <--> node32.pppg++;
        router_0.pppg++ <--> C100kbps <--> node33.pppg++;
        router_0.pppg++ <--> C100kbps <--> node34.pppg++;
        router_0.pppg++ <--> C100kbps <--> node35.pppg++;
        router_0.pppg++ <--> C100kbps <--> node36.pppg++;
        router_0.pppg++ <--> C100kbps <--> node37.pppg++;
        
        router_1.pppg++ <--> C1Mbps <--> node38.pppg++;
        router_1.pppg++ <--> C1Mbps <--> node39.pppg++;
        router_1.pppg++ <--> C1Mbps <--> node40.pppg++;
        router_1.pppg++ <--> C1Mbps <--> node41.pppg++;
        router_1.pppg++ <--> C1Mbps <--> node42.pppg++;
        router_1.pppg++ <--> C1Mbps <--> node43.pppg++;
        router_1.pppg++ <--> C1Mbps <--> node44.pppg++;
        router_1.pppg++ <--> C1Mbps <--> node45.pppg++;
        router_1.pppg++ <--> C1Mbps <--> node46.pppg++;
        router_1.pppg++ <--> C1Mbps <--> node47.pppg++;
        router_1.pppg++ <--> C1Mbps <--> node48.pppg++;
        router_1.pppg++ <--> C1Mbps <--> node49.pppg++;
}


network EvaluationNetworkEthernet_20
{
    parameters:
        @display("i=block/network2;bgb=400, 400");
    submodules:
        
        timeAdvancer: MangoTimeAdvancer {
                @display("p=376.42242,21.847977");
        }
        
        router_central: Router {
                @display("p=321.8025,53.70961");
        }
        configurator: Ipv4NetworkConfigurator {
                @display("p=376.42242,56.89577");
        }
        router_0: Router {
                @display("p=196.63179,223.48659");
        }
        router_1: Router {
                @display("p=127.44653,64.6336");
        }
        node0: StandardHost {
                // central entity 
                @display("p=330,330");
        }
        node1: StandardHost {
                //  Buchenweg 2 
                @display("p= 136.19808178127278 , 266.7808129210025 ");
        }
        node2: StandardHost {
                //  Buchenweg 4 
                @display("p= 135.34602346195607 , 258.4243683256209 ");
        }
        node3: StandardHost {
                //  Buchenweg 1 
                @display("p= 109.55600467708427 , 279.1108704926446 ");
        }
        node4: StandardHost {
                //  Buchenweg 3 
                @display("p= 107.35014683264308 , 263.0050863940269 ");
        }
        node5: StandardHost {
                //  Buchenweg 5 
                @display("p= 103.98684812395368 , 247.6105312211439 ");
        }
        node6: StandardHost {
                //  Buchenweg 6 
                @display("p= 134.07065122557105 , 236.87388666719198 ");
        }
        node7: StandardHost {
                //  Buchenweg 7 
                @display("p= 109.70507018500939 , 231.05967176519334 ");
        }
        node8: StandardHost {
                //  Buchenweg 9 
                @display("p= 106.0909309007111 , 215.9685475854203 ");
        }
        node9: StandardHost {
                //  Buchenweg 11a 
                @display("p= 100.0 , 190.81692989636213 ");
        }
        node10: StandardHost {
                //  Buchenweg 11 
                @display("p= 116.87517556815874 , 193.14509646128863 ");
        }
        node11: StandardHost {
                //  Buchenweg 8 
                @display("p= 141.10889204329578 , 212.15646655112505 ");
        }
        node12: StandardHost {
                //  Buchenweg 13 
                @display("p= 124.6468965325621 , 173.46035425085574 ");
        }
        node13: StandardHost {
                //  Buchenweg 15 
                @display("p= 140.71447496837936 , 156.75369920022786 ");
        }
        node14: StandardHost {
                //  Buchenweg 17 
                @display("p= 158.76687453780323 , 143.00827796570957 ");
        }
        node15: StandardHost {
                //  Buchenweg 19 
                @display("p= 173.0003695912892 , 142.22780754975975 ");
        }
        node16: StandardHost {
                //  Buchenweg 21 
                @display("p= 187.83278567378875 , 140.1346326135099 ");
        }
        node17: StandardHost {
                //  Buchenweg 23 
                @display("p= 203.3587229428813 , 135.51941210310906 ");
        }
        node18: StandardHost {
                //  Buchenweg 25 
                @display("p= 217.60022531513823 , 136.25009013991803 ");
        }
        node19: StandardHost {
                //  Buchenweg 25a 
                @display("p= 214.6824416151503 , 100.0 ");
        }
       
    connections:
        router_0.pppg++ <--> Eth10G <--> router_central.pppg++;
        router_1.pppg++ <--> Eth10G <--> router_central.pppg++;
        node0.pppg ++ <--> Eth10G <--> router_central.pppg++;
        
        router_0.pppg++ <--> C100kbps <--> node1.pppg++;
        router_0.pppg++ <--> C100kbps <--> node2.pppg++;
        router_0.pppg++ <--> C100kbps <--> node3.pppg++;
        router_0.pppg++ <--> C100kbps <--> node4.pppg++;
        router_0.pppg++ <--> C100kbps <--> node5.pppg++;
        router_0.pppg++ <--> C100kbps <--> node6.pppg++;
        router_0.pppg++ <--> C100kbps <--> node7.pppg++;
        router_0.pppg++ <--> C100kbps <--> node8.pppg++;
        router_0.pppg++ <--> C100kbps <--> node9.pppg++;
        router_0.pppg++ <--> C100kbps <--> node10.pppg++;
        router_0.pppg++ <--> C100kbps <--> node11.pppg++;
        router_0.pppg++ <--> C100kbps <--> node12.pppg++;
        router_0.pppg++ <--> C100kbps <--> node13.pppg++;
        router_0.pppg++ <--> C100kbps <--> node14.pppg++;
        router_0.pppg++ <--> C100kbps <--> node15.pppg++;
        router_0.pppg++ <--> C100kbps <--> node16.pppg++;
        router_0.pppg++ <--> C100kbps <--> node17.pppg++;
        router_0.pppg++ <--> C100kbps <--> node18.pppg++;
        router_0.pppg++ <--> C100kbps <--> node19.pppg++;
}


network EvaluationNetworkEthernet_10
{
    parameters:
        @display("i=block/network2;bgb=400, 400");
    submodules:
        
        timeAdvancer: MangoTimeAdvancer {
                @display("p=376.42242,21.847977");
        }
        
        router_central: Router {
                @display("p=321.8025,53.70961");
        }
        configurator: Ipv4NetworkConfigurator {
                @display("p=376.42242,56.89577");
        }
        router_0: Router {
                @display("p=196.63179,223.48659");
        }
        router_1: Router {
                @display("p=127.44653,64.6336");
        }
        node0: StandardHost {
                // central entity 
                @display("p=330,330");
        }
        node1: StandardHost {
                //  Buchenweg 2 
                @display("p= 136.19808178127278 , 266.7808129210025 ");
        }
        node2: StandardHost {
                //  Buchenweg 4 
                @display("p= 135.34602346195607 , 258.4243683256209 ");
        }
        node3: StandardHost {
                //  Buchenweg 1 
                @display("p= 109.55600467708427 , 279.1108704926446 ");
        }
        node4: StandardHost {
                //  Buchenweg 3 
                @display("p= 107.35014683264308 , 263.0050863940269 ");
        }
        node5: StandardHost {
                //  Buchenweg 5 
                @display("p= 103.98684812395368 , 247.6105312211439 ");
        }
        node6: StandardHost {
                //  Buchenweg 6 
                @display("p= 134.07065122557105 , 236.87388666719198 ");
        }
        node7: StandardHost {
                //  Buchenweg 7 
                @display("p= 109.70507018500939 , 231.05967176519334 ");
        }
        node8: StandardHost {
                //  Buchenweg 9 
                @display("p= 106.0909309007111 , 215.9685475854203 ");
        }
        node9: StandardHost {
                //  Buchenweg 11a 
                @display("p= 100.0 , 190.81692989636213 ");
        }
       
    connections:
        router_0.pppg++ <--> Eth10G <--> router_central.pppg++;
        router_1.pppg++ <--> Eth10G <--> router_central.pppg++;
        node0.pppg ++ <--> Eth10G <--> router_central.pppg++;
        
        router_0.pppg++ <--> C100kbps <--> node1.pppg++;
        router_0.pppg++ <--> C100kbps <--> node2.pppg++;
        router_0.pppg++ <--> C100kbps <--> node3.pppg++;
        router_0.pppg++ <--> C100kbps <--> node4.pppg++;
        router_0.pppg++ <--> C100kbps <--> node5.pppg++;
        router_0.pppg++ <--> C100kbps <--> node6.pppg++;
        router_0.pppg++ <--> C100kbps <--> node7.pppg++;
        router_0.pppg++ <--> C100kbps <--> node8.pppg++;
        router_0.pppg++ <--> C100kbps <--> node9.pppg++;
}

network EvaluationNetworkEthernet_5
{
    parameters:
        @display("i=block/network2;bgb=400, 400");
    submodules:
        
        timeAdvancer: MangoTimeAdvancer {
                @display("p=376.42242,21.847977");
        }
        
        router_central: Router {
                @display("p=321.8025,53.70961");
        }
        configurator: Ipv4NetworkConfigurator {
                @display("p=376.42242,56.89577");
        }
        router_0: Router {
                @display("p=196.63179,223.48659");
        }
        router_1: Router {
                @display("p=127.44653,64.6336");
        }
        node0: StandardHost {
                // central entity 
                @display("p=330,330");
        }
        node1: StandardHost {
                //  Buchenweg 2 
                @display("p= 136.19808178127278 , 266.7808129210025 ");
        }
        node2: StandardHost {
                //  Buchenweg 4 
                @display("p= 135.34602346195607 , 258.4243683256209 ");
        }
        node3: StandardHost {
                //  Buchenweg 1 
                @display("p= 109.55600467708427 , 279.1108704926446 ");
        }
        node4: StandardHost {
                //  Buchenweg 3 
                @display("p= 107.35014683264308 , 263.0050863940269 ");
        }
       
    connections:
        router_0.pppg++ <--> Eth10G <--> router_central.pppg++;
        router_1.pppg++ <--> Eth10G <--> router_central.pppg++;
        node0.pppg ++ <--> Eth10G <--> router_central.pppg++;
        
        router_0.pppg++ <--> C100kbps <--> node1.pppg++;
        router_0.pppg++ <--> C100kbps <--> node2.pppg++;
        router_0.pppg++ <--> C100kbps <--> node3.pppg++;
        router_0.pppg++ <--> C100kbps <--> node4.pppg++;
}
