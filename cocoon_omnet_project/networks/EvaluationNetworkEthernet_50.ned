package networks; 
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.networklayer.ipv4.RoutingTableRecorder;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import MangoTimeAdvancer;
network EvaluationNetworkEthernet_50 {
parameters: @display("i=block/network2;bgb=7498.455814999994,9520.953449999914");
submodules:

	timeAdvancer: MangoTimeAdvancer {@display("p=337,40");}
	router_central: Router{@display("p=3479,5460");}
	configurator: Ipv4NetworkConfigurator{@display("p=100,100");}
	router_0: Router{@display("p=1979,7473");}
	router_1: Router{@display("p=6633,6870");}
	router_2: Router{@display("p=1787,837");}
	node1: StandardHost {@display("p=6581,6470");}
	node2: StandardHost {@display("p=3125,1466");}
	node3: StandardHost {@display("p=1979,7473");}
	node4: StandardHost {@display("p=6633,6870");}
	node5: StandardHost {@display("p=7493,7905");}
	node6: StandardHost {@display("p=1827,5903");}
	node7: StandardHost {@display("p=1535,9088");}
	node8: StandardHost {@display("p=1903,5284");}
	node9: StandardHost {@display("p=1979,7473");}
	node10: StandardHost {@display("p=37,2220");}
	node11: StandardHost {@display("p=2098,10");}
	node12: StandardHost {@display("p=7430,2812");}
	node13: StandardHost {@display("p=7287,8187");}
	node14: StandardHost {@display("p=6335,8136");}
	node15: StandardHost {@display("p=1429,6010");}
	node16: StandardHost {@display("p=2186,4882");}
	node17: StandardHost {@display("p=1733,5644");}
	node18: StandardHost {@display("p=6581,6470");}
	node19: StandardHost {@display("p=1681,133");}
	node20: StandardHost {@display("p=2138,7539");}
	node21: StandardHost {@display("p=3125,1466");}
	node22: StandardHost {@display("p=3542,1862");}
	node23: StandardHost {@display("p=1712,4955");}
	node24: StandardHost {@display("p=3636,9520");}
	node25: StandardHost {@display("p=2082,410");}
	node26: StandardHost {@display("p=2770,7765");}
	node27: StandardHost {@display("p=1987,5654");}
	node28: StandardHost {@display("p=7440,3407");}
	node29: StandardHost {@display("p=2997,9137");}
	node30: StandardHost {@display("p=2628,7566");}
	node31: StandardHost {@display("p=3663,1621");}
	node32: StandardHost {@display("p=2770,7765");}
	node33: StandardHost {@display("p=7498,3089");}
	node34: StandardHost {@display("p=6941,8285");}
	node35: StandardHost {@display("p=7258,7656");}
	node36: StandardHost {@display("p=1787,837");}
	node37: StandardHost {@display("p=996,9159");}
	node38: StandardHost {@display("p=2186,4882");}
	node39: StandardHost {@display("p=2098,10");}
	node40: StandardHost {@display("p=3955,2188");}
	node41: StandardHost {@display("p=6706,8764");}
	node42: StandardHost {@display("p=10,8513");}
	node43: StandardHost {@display("p=3499,9147");}
	node44: StandardHost {@display("p=7287,8187");}
	node45: StandardHost {@display("p=2380,9074");}
	node46: StandardHost {@display("p=1733,5644");}
	node47: StandardHost {@display("p=1712,4955");}
	node48: StandardHost {@display("p=2082,410");}
	node49: StandardHost {@display("p=1987,5654");}
	node0: StandardHost {@display("p=3479,5460");}
connections:
	node0.pppg++ <--> C1Gbps <--> router_central.pppg++;
	router_0.pppg++ <--> C1Gbps <--> router_central.pppg++;
	router_1.pppg++ <--> C1Gbps <--> router_central.pppg++;
	router_2.pppg++ <--> C1Gbps <--> router_central.pppg++;
	node1.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node2.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node3.pppg++ <--> C100kbps <--> router_0.pppg++;
	node4.pppg++ <--> C100kbps <--> router_1.pppg++;
	node5.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node6.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node7.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node8.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node9.pppg++ <--> C10kbps <--> router_0.pppg++;
	node10.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node11.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node12.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node13.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node14.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node15.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node16.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node17.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node18.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node19.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node20.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node21.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node22.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node23.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node24.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node25.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node26.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node27.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node28.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node29.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node30.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node31.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node32.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node33.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node34.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node35.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node36.pppg++ <--> C100kbps <--> router_2.pppg++;
	node37.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node38.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node39.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node40.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node41.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node42.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node43.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node44.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node45.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node46.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node47.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node48.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node49.pppg++ <--> C1Mbps <--> router_0.pppg++;
}