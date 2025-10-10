package networks; 
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.networklayer.ipv4.RoutingTableRecorder;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import MangoTimeAdvancer;
network EvaluationNetworkEthernet_5 {
parameters: @display("i=block/network2;bgb=2283.8949850001372,2344.164245000109");
submodules:

	timeAdvancer: MangoTimeAdvancer {@display("p=337,40");}
	router_central: Router{@display("p=758,702");}
	configurator: Ipv4NetworkConfigurator{@display("p=100,100");}
	router_0: Router{@display("p=2283,2344");}
	router_1: Router{@display("p=426,165");}
	router_2: Router{@display("p=10,288");}
	node1: StandardHost {@display("p=426,165");}
	node2: StandardHost {@display("p=10,288");}
	node3: StandardHost {@display("p=2283,2344");}
	node4: StandardHost {@display("p=311,10");}
	node0: StandardHost {@display("p=758,702");}
connections:
	node0.pppg++ <--> C1Gbps <--> router_central.pppg++;
	router_0.pppg++ <--> C1Gbps <--> router_central.pppg++;
	router_1.pppg++ <--> C1Gbps <--> router_central.pppg++;
	router_2.pppg++ <--> C1Gbps <--> router_central.pppg++;
	node1.pppg++ <--> C100kbps <--> router_1.pppg++;
	node2.pppg++ <--> C100kbps <--> router_2.pppg++;
	node3.pppg++ <--> C100kbps <--> router_0.pppg++;
	node4.pppg++ <--> C1Mbps <--> router_1.pppg++;
}