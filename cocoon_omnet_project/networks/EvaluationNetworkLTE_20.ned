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
network EvaluationNetworkLTE_20
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
        node10: Ue {
            @display("p=1654,133");
        }
        node11: Ue {
            @display("p=3636,1621");
        }
        node12: Ue {
            @display("p=1654,133");
        }
        node13: Ue {
            @display("p=1699,377");
        }
        node14: Ue {
            @display("p=3098,1466");
        }
        node15: Ue {
            @display("p=2070,10");
        }
        node16: Ue {
            @display("p=1258,1331");
        }
        node17: Ue {
            @display("p=10,2220");
        }
        node18: Ue {
            @display("p=1837,557");
        }
        node19: Ue {
            @display("p=3098,1466");
        }
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
