package networks; 
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.networklayer.ipv4.RoutingTableRecorder;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import MangoTimeAdvancer;
network EvaluationNetworkEthernet_20 {
parameters: @display("i=block/network2;bgb=3991.5072550000623,2220.3622749997303");
submodules:

	timeAdvancer: MangoTimeAdvancer {@display("p=337,40");}
	router_central: Router{@display("p=2508,1050");}
	configurator: Ipv4NetworkConfigurator{@display("p=100,100");}
	router_0: Router{@display("p=3636,1621");}
	router_1: Router{@display("p=2055,410");}
	router_2: Router{@display("p=1258,1331");}
	node1: StandardHost {@display("p=2070,10");}
	node2: StandardHost {@display("p=3991,1517");}
	node3: StandardHost {@display("p=2055,410");}
	node4: StandardHost {@display("p=3928,2188");}
	node5: StandardHost {@display("p=3928,2188");}
	node6: StandardHost {@display("p=3515,1862");}
	node7: StandardHost {@display("p=2468,423");}
	node8: StandardHost {@display("p=3636,1621");}
	node9: StandardHost {@display("p=2055,410");}
	node10: StandardHost {@display("p=1654,133");}
	node11: StandardHost {@display("p=3636,1621");}
	node12: StandardHost {@display("p=1654,133");}
	node13: StandardHost {@display("p=1699,377");}
	node14: StandardHost {@display("p=3098,1466");}
	node15: StandardHost {@display("p=2070,10");}
	node16: StandardHost {@display("p=1258,1331");}
	node17: StandardHost {@display("p=10,2220");}
	node18: StandardHost {@display("p=1837,557");}
	node19: StandardHost {@display("p=3098,1466");}
	node0: StandardHost {@display("p=2508,1050");}
connections:
	node0.pppg++ <--> C1Gbps <--> router_central.pppg++;
	router_0.pppg++ <--> C1Gbps <--> router_central.pppg++;
	router_1.pppg++ <--> C1Gbps <--> router_central.pppg++;
	router_2.pppg++ <--> C1Gbps <--> router_central.pppg++;
	node1.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node2.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node3.pppg++ <--> C100kbps <--> router_1.pppg++;
	node4.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node5.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node6.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node7.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node8.pppg++ <--> C100kbps <--> router_0.pppg++;
	node9.pppg++ <--> C10kbps <--> router_1.pppg++;
	node10.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node11.pppg++ <--> C10kbps <--> router_0.pppg++;
	node12.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node13.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node14.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node15.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node16.pppg++ <--> C100kbps <--> router_2.pppg++;
	node17.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node18.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node19.pppg++ <--> C1Mbps <--> router_0.pppg++;
}