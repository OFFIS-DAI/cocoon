package networks;

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.networklayer.ipv4.RoutingTableRecorder;
import inet.node.ethernet.Eth10G;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import simu5g.common.binder.Binder;
import simu5g.common.carrierAggregation.CarrierAggregation;
import simu5g.nodes.Upf;
import simu5g.nodes.NR.gNodeB;
import simu5g.nodes.NR.NRUe;
import simu5g.world.radio.LteChannelControl;
import MangoTimeAdvancer;

network EvaluationNetwork5G_50
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
        
        upf: Upf {
            @display("p=100,100;is=l");
        }
        gNB0: gNodeB {
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
        node1: NRUe {
            @display("p=1826,1862");
        }
        node2: NRUe {
            @display("p=365,410");
        }
        node3: NRUe {
            @display("p=10,377");
        }
        node4: NRUe {
            @display("p=147,557");
        }
        node5: NRUe {
            @display("p=778,423");
        }
        node6: NRUe {
            @display("p=2302,1517");
        }
        node7: NRUe {
            @display("p=381,10");
        }
        node8: NRUe {
            @display("p=381,10");
        }
        node9: NRUe {
            @display("p=1947,1621");
        }
        node10: NRUe {
            @display("p=1654,133");
        }
        node11: NRUe {
            @display("p=3636,1621");
        }
        node12: NRUe {
            @display("p=1654,133");
        }
        node13: NRUe {
            @display("p=1699,377");
        }
        node14: NRUe {
            @display("p=3098,1466");
        }
        node15: NRUe {
            @display("p=2070,10");
        }
        node16: NRUe {
            @display("p=1258,1331");
        }
        node17: NRUe {
            @display("p=10,2220");
        }
        node18: NRUe {
            @display("p=1837,557");
        }
        node19: NRUe {
            @display("p=3098,1466");
        }
        node20: NRUe {
            @display("p=2138,7539");
        }
        node21: NRUe {
            @display("p=3125,1466");
        }
        node22: NRUe {
            @display("p=3542,1862");
        }
        node23: NRUe {
            @display("p=1712,4955");
        }
        node24: NRUe {
            @display("p=3636,9520");
        }
        node25: NRUe {
            @display("p=2082,410");
        }
        node26: NRUe {
            @display("p=2770,7765");
        }
        node27: NRUe {
            @display("p=1987,5654");
        }
        node28: NRUe {
            @display("p=7440,3407");
        }
        node29: NRUe {
            @display("p=2997,9137");
        }
        node30: NRUe {
            @display("p=2628,7566");
        }
        node31: NRUe {
            @display("p=3663,1621");
        }
        node32: NRUe {
            @display("p=2770,7765");
        }
        node33: NRUe {
            @display("p=7498,3089");
        }
        node34: NRUe {
            @display("p=6941,8285");
        }
        node35: NRUe {
            @display("p=7258,7656");
        }
        node36: NRUe {
            @display("p=1787,837");
        }
        node37: NRUe {
            @display("p=996,9159");
        }
        node38: NRUe {
            @display("p=2186,4882");
        }
        node39: NRUe {
            @display("p=2098,10");
        }
        node40: NRUe {
            @display("p=3955,2188");
        }
        node41: NRUe {
            @display("p=6706,8764");
        }
        node42: NRUe {
            @display("p=10,8513");
        }
        node43: NRUe {
            @display("p=3499,9147");
        }
        node44: NRUe {
            @display("p=7287,8187");
        }
        node45: NRUe {
            @display("p=2380,9074");
        }
        node46: NRUe {
            @display("p=1733,5644");
        }
        node47: NRUe {
            @display("p=1712,4955");
        }
        node48: NRUe {
            @display("p=2082,410");
        }
        node49: NRUe {
            @display("p=1987,5654");
        }
        node0: NRUe {
            @display("p=904,754");
        }
    connections:
        router_0.pppg++ <--> C1Gbps <--> router_central.pppg++;
		router_1.pppg++ <--> C1Gbps <--> router_central.pppg++;
		router_2.pppg++ <--> C1Gbps <--> router_central.pppg++;
        server.pppg++ <--> Eth10G <--> router_central.pppg++;
    	router_central.pppg++ <--> Eth10G <--> upf.filterGate;
    	upf.pppg++ <--> Eth10G <--> gNB0.ppp;
}
