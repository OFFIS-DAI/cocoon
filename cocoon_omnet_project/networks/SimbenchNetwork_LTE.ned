package networks;

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.networklayer.ipv4.RoutingTableRecorder;
import inet.node.ethernet.Eth10G;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import simu5g.common.binder.Binder;
import simu5g.common.carrierAggregation.CarrierAggregation;
import simu5g.nodes.Ue;
import simu5g.nodes.eNodeB;
import simu5g.nodes.PgwStandard;
import simu5g.world.radio.LteChannelControl;
import MangoTimeAdvancer;


network SimbenchNetworkLTE {
parameters: @display("i=block/network2;bgb=389.35363110608887,1766.448845718056");
submodules:
    
    timeAdvancer: MangoTimeAdvancer {
            @display("p=337,40");
        }

	channelControl: LteChannelControl{@display("p=100,100;is=vl");}
	routingRecorder: RoutingTableRecorder{@display("p=100,100;is=vl");}
	configurator: Ipv4NetworkConfigurator{@display("p=100,100;is=vl");}
	binder: Binder{@display("p=100,100;is=vl");}
	carrierAggregation: CarrierAggregation{@display("p=100,100;is=vl");}
	server: StandardHost{@display("p=100,100;is=vl");}
	router: Router{@display("p=100,100;is=vl");}
	pgw: PgwStandard{@display("p=100,100;is=vl");}
	eNB0: eNodeB{@display("p=189,1000;is=vl");}
	node0: Ue {@display("p=62,1246");}
	node1: Ue {@display("p=62,1246");}
	node2: Ue {@display("p=306,1655");}
	node3: Ue {@display("p=310,1543");}
	node4: Ue {@display("p=336,976");}
	node5: Ue {@display("p=336,976");}
	node6: Ue {@display("p=221,1262");}
	node7: Ue {@display("p=221,1262");}
	node8: Ue {@display("p=221,1262");}
	node9: Ue {@display("p=316,1377");}
	node10: Ue {@display("p=336,208");}
	node11: Ue {@display("p=339,687");}
	node12: Ue {@display("p=339,687");}
	node13: Ue {@display("p=312,1488");}
	node14: Ue {@display("p=350,387");}
	node15: Ue {@display("p=36,1634");}
	node16: Ue {@display("p=64,1190");}
	node17: Ue {@display("p=215,1429");}
	node18: Ue {@display("p=338,921");}
	node19: Ue {@display("p=338,921");}
	node20: Ue {@display("p=127,1303");}
	node21: Ue {@display("p=389,210");}
	node22: Ue {@display("p=344,732");}
	node23: Ue {@display("p=344,732");}
	node24: Ue {@display("p=117,1581");}
	node25: Ue {@display("p=128,1248");}
	node26: Ue {@display("p=13,1144");}
	node27: Ue {@display("p=220,1485");}
	node28: Ue {@display("p=344,732");}
	node29: Ue {@display("p=344,732");}
	node30: Ue {@display("p=328,1021");}
	node31: Ue {@display("p=345,531");}
	node32: Ue {@display("p=345,531");}
	node33: Ue {@display("p=308,1599");}
	node34: Ue {@display("p=334,832");}
	node35: Ue {@display("p=111,1748");}
	node36: Ue {@display("p=324,1332");}
	node37: Ue {@display("p=324,1332");}
	node38: Ue {@display("p=228,1040");}
	node39: Ue {@display("p=228,1040");}
	node40: Ue {@display("p=123,1415");}
	node41: Ue {@display("p=123,1415");}
	node42: Ue {@display("p=230,995");}
	node43: Ue {@display("p=230,995");}
	node44: Ue {@display("p=10,1244");}
	node45: Ue {@display("p=314,1432");}
	node46: Ue {@display("p=121,1470");}
	node47: Ue {@display("p=19,955");}
	node48: Ue {@display("p=19,955");}
	node49: Ue {@display("p=349,587");}
	node50: Ue {@display("p=29,855");}
	node51: Ue {@display("p=33,1523");}
	node52: Ue {@display("p=33,1523");}
	node53: Ue {@display("p=33,1523");}
	node54: Ue {@display("p=16,1044");}
	node55: Ue {@display("p=208,1640");}
	node56: Ue {@display("p=208,1640");}
	node57: Ue {@display("p=208,1640");}
	node58: Ue {@display("p=332,308");}
	node59: Ue {@display("p=341,631");}
	node60: Ue {@display("p=383,176");}
	node61: Ue {@display("p=337,164");}
	node62: Ue {@display("p=122,1637");}
	node63: Ue {@display("p=11,1188");}
	node64: Ue {@display("p=210,1585");}
	node65: Ue {@display("p=385,132");}
	node66: Ue {@display("p=136,1036");}
	node67: Ue {@display("p=31,1579");}
	node68: Ue {@display("p=21,899");}
	node69: Ue {@display("p=324,1332");}
	node70: Ue {@display("p=324,1332");}
	node71: Ue {@display("p=338,921");}
	node72: Ue {@display("p=338,921");}
	node73: Ue {@display("p=238,940");}
	node74: Ue {@display("p=346,487");}
	node75: Ue {@display("p=340,86");}
	node76: Ue {@display("p=340,86");}
	node77: Ue {@display("p=340,86");}
	node78: Ue {@display("p=43,1412");}
	node79: Ue {@display("p=43,1412");}
	node80: Ue {@display("p=327,264");}
	node81: Ue {@display("p=326,1076");}
	node82: Ue {@display("p=211,1540");}
	node83: Ue {@display("p=211,1540");}
	node84: Ue {@display("p=332,130");}
	node85: Ue {@display("p=219,1318");}
	node86: Ue {@display("p=224,1151");}
	node87: Ue {@display("p=389,10");}
	node88: Ue {@display("p=134,1081");}
	node89: Ue {@display("p=321,1221");}
	node90: Ue {@display("p=386,87");}
	node91: Ue {@display("p=380,265");}
	node92: Ue {@display("p=323,1176");}
	node93: Ue {@display("p=14,1099");}
	node94: Ue {@display("p=132,1137");}
	node95: Ue {@display("p=60,1301");}
	node96: Ue {@display("p=348,442");}
	node97: Ue {@display("p=348,442");}
	node98: Ue {@display("p=348,442");}
	node99: Ue {@display("p=336,776");}
	
connections:
	server.pppg++ <--> Eth10G <--> router.pppg++;
	router.pppg++ <--> Eth10G <--> pgw.filterGate;
	pgw.pppg++ <--> Eth10G <--> eNB0.ppp;
}