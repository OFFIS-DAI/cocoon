package networks;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.networklayer.ipv4.RoutingTableRecorder;
import inet.node.inet.Router;
import simu5g.nodes.Ue;
import inet.node.inet.StandardHost;
import simu5g.common.binder.Binder;
import simu5g.common.carrierAggregation.CarrierAggregation;
import simu5g.nodes.Ue;
import simu5g.nodes.eNodeB;
import simu5g.nodes.PgwStandard;
import simu5g.world.radio.LteChannelControl;
import MangoTimeAdvancer;
network EvaluationNetworkLTE_100
{
    parameters:
        @display("i=block/network2;bgb=2302.182115000207,1862.6272449996322");
    submodules:

        timeAdvancer: MangoTimeAdvancer {
            @display("p=337,40");
        }
        channelControl: LteChannelControl {
            @display("p=100,100;is=vl");
        }
        routingRecorder: RoutingTableRecorder {
            @display("p=100,100;is=vl");
        }
        configurator: Ipv4NetworkConfigurator {
            @display("p=100,100;is=vl");
        }
        binder: Binder {
            @display("p=100,100;is=vl");
        }
        carrierAggregation: CarrierAggregation {
            @display("p=100,100;is=vl");
        }
        server: StandardHost {
            @display("p=100,100;is=vl");
        }
        pgw: PgwStandard {
            @display("p=100,100;is=n");
        }
        eNB0: eNodeB {
            @display("p=189,1000;is=vl");
        }
        router_central: Router {
            @display("p=904,754");
        }
        router_0: Router {
            @display("p=778,423");
        }
        router_1: Router {
            @display("p=1947,1621");
        }
        router_2: Router {
            @display("p=365,410");
        }
        node1: Ue {
            @display("p=1826,1862");
        }
        node2: Ue {
            @display("p=365,410");
        }
        node3: Ue {
            @display("p=10,377");
        }
        node4: Ue {
            @display("p=147,557");
        }
        node5: Ue {
            @display("p=778,423");
        }
        node6: Ue {
            @display("p=2302,1517");
        }
        node7: Ue {
            @display("p=381,10");
        }
        node8: Ue {
            @display("p=381,10");
        }
        node9: Ue {
            @display("p=1947,1621");
        }
        node10: Ue {@display("p=1654,133");}
	node11: Ue {@display("p=3636,1621");}
	node12: Ue {@display("p=1654,133");}
	node13: Ue {@display("p=1699,377");}
	node14: Ue {@display("p=3098,1466");}
	node15: Ue {@display("p=2070,10");}
	node16: Ue {@display("p=1258,1331");}
	node17: Ue {@display("p=10,2220");}
	node18: Ue {@display("p=1837,557");}
	node19: Ue {@display("p=3098,1466");}
	
	node20: Ue {@display("p=2138,7539");}
	node21: Ue {@display("p=3125,1466");}
	node22: Ue {@display("p=3542,1862");}
	node23: Ue {@display("p=1712,4955");}
	node24: Ue {@display("p=3636,9520");}
	node25: Ue {@display("p=2082,410");}
	node26: Ue {@display("p=2770,7765");}
	node27: Ue {@display("p=1987,5654");}
	node28: Ue {@display("p=7440,3407");}
	node29: Ue {@display("p=2997,9137");}
	node30: Ue {@display("p=2628,7566");}
	node31: Ue {@display("p=3663,1621");}
	node32: Ue {@display("p=2770,7765");}
	node33: Ue {@display("p=7498,3089");}
	node34: Ue {@display("p=6941,8285");}
	node35: Ue {@display("p=7258,7656");}
	node36: Ue {@display("p=1787,837");}
	node37: Ue {@display("p=996,9159");}
	node38: Ue {@display("p=2186,4882");}
	node39: Ue {@display("p=2098,10");}
	node40: Ue {@display("p=3955,2188");}
	node41: Ue {@display("p=6706,8764");}
	node42: Ue {@display("p=10,8513");}
	node43: Ue {@display("p=3499,9147");}
	node44: Ue {@display("p=7287,8187");}
	node45: Ue {@display("p=2380,9074");}
	node46: Ue {@display("p=1733,5644");}
	node47: Ue {@display("p=1712,4955");}
	node48: Ue {@display("p=2082,410");}
	node49: Ue {@display("p=1987,5654");}
	node50: Ue {@display("p=7417,5525");}
	node51: Ue {@display("p=7258,7812");}
	node52: Ue {@display("p=947,9570");}
	node53: Ue {@display("p=7083,8846");}
	node54: Ue {@display("p=2138,7694");}
	node55: Ue {@display("p=2400,7640");}
	node56: Ue {@display("p=1827,6059");}
	node57: Ue {@display("p=6688,2457");}
	node58: Ue {@display("p=3542,2017");}
	node59: Ue {@display("p=3636,9676");}
	node60: Ue {@display("p=1712,5111");}
	node61: Ue {@display("p=4018,1672");}
	node62: Ue {@display("p=1535,9243");}
	node63: Ue {@display("p=1787,993");}
	node64: Ue {@display("p=3542,2017");}
	node65: Ue {@display("p=2082,565");}
	node66: Ue {@display("p=1787,993");}
	node67: Ue {@display("p=1681,288");}
	node68: Ue {@display("p=3663,1777");}
	node69: Ue {@display("p=2219,5561");}
	node70: Ue {@display("p=6663,10167");}
	node71: Ue {@display("p=2098,165");}
	node72: Ue {@display("p=1979,7628");}
	node73: Ue {@display("p=7212,8974");}
	node74: Ue {@display("p=10,8668");}
	node75: Ue {@display("p=1864,712");}
	node76: Ue {@display("p=6906,8753");}
	node77: Ue {@display("p=7493,8060");}
	node78: Ue {@display("p=6961,9208");}
	node79: Ue {@display("p=7440,3562");}
	node80: Ue {@display("p=6895,9624");}
	node81: Ue {@display("p=7493,8060");}
	node82: Ue {@display("p=1375,5926");}
	node83: Ue {@display("p=1266,5540");}
	node84: Ue {@display("p=7212,8974");}
	node85: Ue {@display("p=1429,6165");}
	node86: Ue {@display("p=2138,7694");}
	node87: Ue {@display("p=7440,3562");}
	node88: Ue {@display("p=2495,578");}
	node89: Ue {@display("p=2186,5037");}
	node90: Ue {@display("p=1375,5926");}
	node91: Ue {@display("p=6633,7025");}
	node92: Ue {@display("p=2495,578");}
	node93: Ue {@display("p=7143,8659");}
	node94: Ue {@display("p=3636,9676");}
	node95: Ue {@display("p=1712,5111");}
	node96: Ue {@display("p=1733,5799");}
	node97: Ue {@display("p=738,9260");}
	node98: Ue {@display("p=2628,7721");}
	node99: Ue {@display("p=2082,565");}
        node0: Ue {
            @display("p=904,754");
        }
    connections:
        router_0.pppg++ <--> C1Gbps <--> router_central.pppg++;
		router_1.pppg++ <--> C1Gbps <--> router_central.pppg++;
		router_2.pppg++ <--> C1Gbps <--> router_central.pppg++;
        server.pppg++ <--> C1Gbps <--> router_central.pppg++;
        router_central.pppg++ <--> C1Gbps <--> pgw.filterGate;
        pgw.pppg++ <--> C1Gbps <--> eNB0.ppp;
}
