package networks; 
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.networklayer.ipv4.RoutingTableRecorder;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import MangoTimeAdvancer;
network EvaluationNetworkEthernet_10 {
parameters: @display("i=block/network2;bgb=2302.182115000207,1862.6272449996322");
submodules:

	timeAdvancer: MangoTimeAdvancer {@display("p=337,40");}
	router_central: Router{@display("p=904,754");}
	configurator: Ipv4NetworkConfigurator{@display("p=100,100");}
	router_0: Router{@display("p=778,423");}
	router_1: Router{@display("p=1947,1621");}
	router_2: Router{@display("p=365,410");}
	node1: StandardHost {@display("p=1826,1862");}
	node2: StandardHost {@display("p=365,410");}
	node3: StandardHost {@display("p=10,377");}
	node4: StandardHost {@display("p=147,557");}
	node5: StandardHost {@display("p=778,423");}
	node6: StandardHost {@display("p=2302,1517");}
	node7: StandardHost {@display("p=381,10");}
	node8: StandardHost {@display("p=381,10");}
	node9: StandardHost {@display("p=1947,1621");}
	node0: StandardHost {@display("p=904,754");}
connections:
	node0.pppg++ <--> C1Gbps <--> router_central.pppg++;
	router_0.pppg++ <--> C1Gbps <--> router_central.pppg++;
	router_1.pppg++ <--> C1Gbps <--> router_central.pppg++;
	router_2.pppg++ <--> C1Gbps <--> router_central.pppg++;
	node1.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node2.pppg++ <--> C100kbps <--> router_2.pppg++;
	node3.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node4.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node5.pppg++ <--> C100kbps <--> router_0.pppg++;
	node6.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node7.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node8.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node9.pppg++ <--> C100kbps <--> router_1.pppg++;
}