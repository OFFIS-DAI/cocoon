package networks;

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.networklayer.ipv4.RoutingTableRecorder;
import inet.node.ethernet.Eth10M;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import MangoTimeAdvancer;


network SimbenchNetworkEthernet {
parameters: @display("i=block/network2;bgb=389.35363110608887,1766.448845718056");
submodules:
    
    timeAdvancer: MangoTimeAdvancer {
            @display("p=337,40");
        }

	router_central: Router{@display("p=100,100");}
	configurator: Ipv4NetworkConfigurator{@display("p=100,100");}
	router_0: Router{@display("p=345,270");}
	router_1: Router{@display("p=217,974");}
	router_2: Router{@display("p=175,1461");}
	node0: StandardHost {@display("p=62,1246");}
	node1: StandardHost {@display("p=62,1246");}
	node2: StandardHost {@display("p=306,1655");}
	node3: StandardHost {@display("p=310,1543");}
	node4: StandardHost {@display("p=336,976");}
	node5: StandardHost {@display("p=336,976");}
	node6: StandardHost {@display("p=221,1262");}
	node7: StandardHost {@display("p=221,1262");}
	node8: StandardHost {@display("p=221,1262");}
	node9: StandardHost {@display("p=316,1377");}
	node10: StandardHost {@display("p=336,208");}
	node11: StandardHost {@display("p=339,687");}
	node12: StandardHost {@display("p=339,687");}
	node13: StandardHost {@display("p=312,1488");}
	node14: StandardHost {@display("p=350,387");}
	node15: StandardHost {@display("p=36,1634");}
	node16: StandardHost {@display("p=64,1190");}
	node17: StandardHost {@display("p=215,1429");}
	node18: StandardHost {@display("p=338,921");}
	node19: StandardHost {@display("p=338,921");}
	node20: StandardHost {@display("p=127,1303");}
	node21: StandardHost {@display("p=389,210");}
	node22: StandardHost {@display("p=344,732");}
	node23: StandardHost {@display("p=344,732");}
	node24: StandardHost {@display("p=117,1581");}
	node25: StandardHost {@display("p=128,1248");}
	node26: StandardHost {@display("p=13,1144");}
	node27: StandardHost {@display("p=220,1485");}
	node28: StandardHost {@display("p=344,732");}
	node29: StandardHost {@display("p=344,732");}
	node30: StandardHost {@display("p=328,1021");}
	node31: StandardHost {@display("p=345,531");}
	node32: StandardHost {@display("p=345,531");}
	node33: StandardHost {@display("p=308,1599");}
	node34: StandardHost {@display("p=334,832");}
	node35: StandardHost {@display("p=111,1748");}
	node36: StandardHost {@display("p=324,1332");}
	node37: StandardHost {@display("p=324,1332");}
	node38: StandardHost {@display("p=228,1040");}
	node39: StandardHost {@display("p=228,1040");}
	node40: StandardHost {@display("p=123,1415");}
	node41: StandardHost {@display("p=123,1415");}
	node42: StandardHost {@display("p=230,995");}
	node43: StandardHost {@display("p=230,995");}
	node44: StandardHost {@display("p=10,1244");}
	node45: StandardHost {@display("p=314,1432");}
	node46: StandardHost {@display("p=121,1470");}
	node47: StandardHost {@display("p=19,955");}
	node48: StandardHost {@display("p=19,955");}
	node49: StandardHost {@display("p=349,587");}
	node50: StandardHost {@display("p=29,855");}
	node51: StandardHost {@display("p=33,1523");}
	node52: StandardHost {@display("p=33,1523");}
	node53: StandardHost {@display("p=33,1523");}
	node54: StandardHost {@display("p=16,1044");}
	node55: StandardHost {@display("p=208,1640");}
	node56: StandardHost {@display("p=208,1640");}
	node57: StandardHost {@display("p=208,1640");}
	node58: StandardHost {@display("p=332,308");}
	node59: StandardHost {@display("p=341,631");}
	node60: StandardHost {@display("p=383,176");}
	node61: StandardHost {@display("p=337,164");}
	node62: StandardHost {@display("p=122,1637");}
	node63: StandardHost {@display("p=11,1188");}
	node64: StandardHost {@display("p=210,1585");}
	node65: StandardHost {@display("p=385,132");}
	node66: StandardHost {@display("p=136,1036");}
	node67: StandardHost {@display("p=31,1579");}
	node68: StandardHost {@display("p=21,899");}
	node69: StandardHost {@display("p=324,1332");}
	node70: StandardHost {@display("p=324,1332");}
	node71: StandardHost {@display("p=338,921");}
	node72: StandardHost {@display("p=338,921");}
	node73: StandardHost {@display("p=238,940");}
	node74: StandardHost {@display("p=346,487");}
	node75: StandardHost {@display("p=340,86");}
	node76: StandardHost {@display("p=340,86");}
	node77: StandardHost {@display("p=340,86");}
	node78: StandardHost {@display("p=43,1412");}
	node79: StandardHost {@display("p=43,1412");}
	node80: StandardHost {@display("p=327,264");}
	node81: StandardHost {@display("p=326,1076");}
	node82: StandardHost {@display("p=211,1540");}
	node83: StandardHost {@display("p=211,1540");}
	node84: StandardHost {@display("p=332,130");}
	node85: StandardHost {@display("p=219,1318");}
	node86: StandardHost {@display("p=224,1151");}
	node87: StandardHost {@display("p=389,10");}
	node88: StandardHost {@display("p=134,1081");}
	node89: StandardHost {@display("p=321,1221");}
	node90: StandardHost {@display("p=386,87");}
	node91: StandardHost {@display("p=380,265");}
	node92: StandardHost {@display("p=323,1176");}
	node93: StandardHost {@display("p=14,1099");}
	node94: StandardHost {@display("p=132,1137");}
	node95: StandardHost {@display("p=60,1301");}
	node96: StandardHost {@display("p=348,442");}
	node97: StandardHost {@display("p=348,442");}
	node98: StandardHost {@display("p=348,442");}
	node99: StandardHost {@display("p=336,776");}
connections:
	router_0.pppg++ <--> Eth10M <--> router_central.pppg++;
	router_0.pppg++ <--> Eth10M <--> node10.pppg++;
	router_0.pppg++ <--> Eth10M <--> node14.pppg++;
	router_0.pppg++ <--> Eth10M <--> node21.pppg++;
	router_0.pppg++ <--> Eth10M <--> node31.pppg++;
	router_0.pppg++ <--> Eth10M <--> node32.pppg++;
	router_0.pppg++ <--> Eth10M <--> node49.pppg++;
	router_0.pppg++ <--> Eth10M <--> node58.pppg++;
	router_0.pppg++ <--> Eth10M <--> node59.pppg++;
	router_0.pppg++ <--> Eth10M <--> node60.pppg++;
	router_0.pppg++ <--> Eth10M <--> node61.pppg++;
	router_0.pppg++ <--> Eth10M <--> node65.pppg++;
	router_0.pppg++ <--> Eth10M <--> node74.pppg++;
	router_0.pppg++ <--> Eth10M <--> node75.pppg++;
	router_0.pppg++ <--> Eth10M <--> node76.pppg++;
	router_0.pppg++ <--> Eth10M <--> node77.pppg++;
	router_0.pppg++ <--> Eth10M <--> node80.pppg++;
	router_0.pppg++ <--> Eth10M <--> node84.pppg++;
	router_0.pppg++ <--> Eth10M <--> node87.pppg++;
	router_0.pppg++ <--> Eth10M <--> node90.pppg++;
	router_0.pppg++ <--> Eth10M <--> node91.pppg++;
	router_0.pppg++ <--> Eth10M <--> node96.pppg++;
	router_0.pppg++ <--> Eth10M <--> node97.pppg++;
	router_0.pppg++ <--> Eth10M <--> node98.pppg++;
	router_1.pppg++ <--> Eth10M <--> router_central.pppg++;
	router_1.pppg++ <--> Eth10M <--> node4.pppg++;
	router_1.pppg++ <--> Eth10M <--> node5.pppg++;
	router_1.pppg++ <--> Eth10M <--> node11.pppg++;
	router_1.pppg++ <--> Eth10M <--> node12.pppg++;
	router_1.pppg++ <--> Eth10M <--> node16.pppg++;
	router_1.pppg++ <--> Eth10M <--> node18.pppg++;
	router_1.pppg++ <--> Eth10M <--> node19.pppg++;
	router_1.pppg++ <--> Eth10M <--> node22.pppg++;
	router_1.pppg++ <--> Eth10M <--> node23.pppg++;
	router_1.pppg++ <--> Eth10M <--> node26.pppg++;
	router_1.pppg++ <--> Eth10M <--> node28.pppg++;
	router_1.pppg++ <--> Eth10M <--> node29.pppg++;
	router_1.pppg++ <--> Eth10M <--> node30.pppg++;
	router_1.pppg++ <--> Eth10M <--> node34.pppg++;
	router_1.pppg++ <--> Eth10M <--> node38.pppg++;
	router_1.pppg++ <--> Eth10M <--> node39.pppg++;
	router_1.pppg++ <--> Eth10M <--> node42.pppg++;
	router_1.pppg++ <--> Eth10M <--> node43.pppg++;
	router_1.pppg++ <--> Eth10M <--> node47.pppg++;
	router_1.pppg++ <--> Eth10M <--> node48.pppg++;
	router_1.pppg++ <--> Eth10M <--> node50.pppg++;
	router_1.pppg++ <--> Eth10M <--> node54.pppg++;
	router_1.pppg++ <--> Eth10M <--> node63.pppg++;
	router_1.pppg++ <--> Eth10M <--> node66.pppg++;
	router_1.pppg++ <--> Eth10M <--> node68.pppg++;
	router_1.pppg++ <--> Eth10M <--> node71.pppg++;
	router_1.pppg++ <--> Eth10M <--> node72.pppg++;
	router_1.pppg++ <--> Eth10M <--> node73.pppg++;
	router_1.pppg++ <--> Eth10M <--> node81.pppg++;
	router_1.pppg++ <--> Eth10M <--> node86.pppg++;
	router_1.pppg++ <--> Eth10M <--> node88.pppg++;
	router_1.pppg++ <--> Eth10M <--> node89.pppg++;
	router_1.pppg++ <--> Eth10M <--> node92.pppg++;
	router_1.pppg++ <--> Eth10M <--> node93.pppg++;
	router_1.pppg++ <--> Eth10M <--> node94.pppg++;
	router_1.pppg++ <--> Eth10M <--> node99.pppg++;

	router_2.pppg++ <--> Eth10M <--> router_central.pppg++;
	router_2.pppg++ <--> Eth10M <--> node0.pppg++;
	router_2.pppg++ <--> Eth10M <--> node1.pppg++;
	router_2.pppg++ <--> Eth10M <--> node2.pppg++;
	router_2.pppg++ <--> Eth10M <--> node3.pppg++;
	router_2.pppg++ <--> Eth10M <--> node6.pppg++;
	router_2.pppg++ <--> Eth10M <--> node7.pppg++;
	router_2.pppg++ <--> Eth10M <--> node8.pppg++;
	router_2.pppg++ <--> Eth10M <--> node9.pppg++;
	router_2.pppg++ <--> Eth10M <--> node13.pppg++;
	router_2.pppg++ <--> Eth10M <--> node15.pppg++;
	router_2.pppg++ <--> Eth10M <--> node17.pppg++;
	router_2.pppg++ <--> Eth10M <--> node20.pppg++;
	router_2.pppg++ <--> Eth10M <--> node24.pppg++;
	router_2.pppg++ <--> Eth10M <--> node25.pppg++;
	router_2.pppg++ <--> Eth10M <--> node27.pppg++;
	router_2.pppg++ <--> Eth10M <--> node33.pppg++;
	router_2.pppg++ <--> Eth10M <--> node35.pppg++;
	router_2.pppg++ <--> Eth10M <--> node36.pppg++;
	router_2.pppg++ <--> Eth10M <--> node37.pppg++;
	router_2.pppg++ <--> Eth10M <--> node40.pppg++;
	router_2.pppg++ <--> Eth10M <--> node41.pppg++;
	router_2.pppg++ <--> Eth10M <--> node44.pppg++;
	router_2.pppg++ <--> Eth10M <--> node45.pppg++;
	router_2.pppg++ <--> Eth10M <--> node46.pppg++;
	router_2.pppg++ <--> Eth10M <--> node51.pppg++;
	router_2.pppg++ <--> Eth10M <--> node52.pppg++;
	router_2.pppg++ <--> Eth10M <--> node53.pppg++;
	router_2.pppg++ <--> Eth10M <--> node55.pppg++;
	router_2.pppg++ <--> Eth10M <--> node56.pppg++;
	router_2.pppg++ <--> Eth10M <--> node57.pppg++;
	router_2.pppg++ <--> Eth10M <--> node62.pppg++;
	router_2.pppg++ <--> Eth10M <--> node64.pppg++;
	router_2.pppg++ <--> Eth10M <--> node67.pppg++;
	router_2.pppg++ <--> Eth10M <--> node69.pppg++;
	router_2.pppg++ <--> Eth10M <--> node70.pppg++;
	router_2.pppg++ <--> Eth10M <--> node78.pppg++;
	router_2.pppg++ <--> Eth10M <--> node79.pppg++;
	router_2.pppg++ <--> Eth10M <--> node82.pppg++;
	router_2.pppg++ <--> Eth10M <--> node83.pppg++;
	router_2.pppg++ <--> Eth10M <--> node85.pppg++;
	router_2.pppg++ <--> Eth10M <--> node95.pppg++;
}