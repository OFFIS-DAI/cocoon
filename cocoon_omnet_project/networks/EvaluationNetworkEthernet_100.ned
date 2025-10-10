package networks; 
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.networklayer.ipv4.RoutingTableRecorder;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import MangoTimeAdvancer;
network EvaluationNetworkEthernet_100 {
parameters: @display("i=block/network2;bgb=7498.455814999994,10167.105360000394");
submodules:

	timeAdvancer: MangoTimeAdvancer {@display("p=337,40");}
	router_central: Router{@display("p=3821,5920");}
	configurator: Ipv4NetworkConfigurator{@display("p=100,100");}
	router_0: Router{@display("p=1979,7628");}
	router_1: Router{@display("p=3125,1621");}
	router_2: Router{@display("p=6941,8441");}
	node1: StandardHost {@display("p=37,2375");}
	node2: StandardHost {@display("p=1733,5799");}
	node3: StandardHost {@display("p=7483,7649");}
	node4: StandardHost {@display("p=1285,1486");}
	node5: StandardHost {@display("p=6533,8667");}
	node6: StandardHost {@display("p=3955,2344");}
	node7: StandardHost {@display("p=7498,3245");}
	node8: StandardHost {@display("p=6533,8667");}
	node9: StandardHost {@display("p=6712,8197");}
	node10: StandardHost {@display("p=7483,7649");}
	node11: StandardHost {@display("p=3125,1621");}
	node12: StandardHost {@display("p=2219,5561");}
	node13: StandardHost {@display("p=2380,9230");}
	node14: StandardHost {@display("p=6941,8441");}
	node15: StandardHost {@display("p=1815,9272");}
	node16: StandardHost {@display("p=2400,7640");}
	node17: StandardHost {@display("p=6941,8441");}
	node18: StandardHost {@display("p=947,9570");}
	node19: StandardHost {@display("p=894,8920");}
	node20: StandardHost {@display("p=6581,6625");}
	node21: StandardHost {@display("p=1987,5809");}
	node22: StandardHost {@display("p=7083,8846");}
	node23: StandardHost {@display("p=1987,5809");}
	node24: StandardHost {@display("p=1266,5540");}
	node25: StandardHost {@display("p=6690,9071");}
	node26: StandardHost {@display("p=1726,533");}
	node27: StandardHost {@display("p=1535,9243");}
	node28: StandardHost {@display("p=1983,10");}
	node29: StandardHost {@display("p=7287,8342");}
	node30: StandardHost {@display("p=738,9260");}
	node31: StandardHost {@display("p=1864,712");}
	node32: StandardHost {@display("p=7498,3245");}
	node33: StandardHost {@display("p=2098,165");}
	node34: StandardHost {@display("p=6581,6625");}
	node35: StandardHost {@display("p=6712,8197");}
	node36: StandardHost {@display("p=2153,8981");}
	node37: StandardHost {@display("p=3125,1621");}
	node38: StandardHost {@display("p=1903,5439");}
	node39: StandardHost {@display("p=3663,1777");}
	node40: StandardHost {@display("p=3955,2344");}
	node41: StandardHost {@display("p=2770,7920");}
	node42: StandardHost {@display("p=6706,8920");}
	node43: StandardHost {@display("p=1345,8926");}
	node44: StandardHost {@display("p=3499,9302");}
	node45: StandardHost {@display("p=4263,4432");}
	node46: StandardHost {@display("p=7143,8659");}
	node47: StandardHost {@display("p=6687,9339");}
	node48: StandardHost {@display("p=2186,5037");}
	node49: StandardHost {@display("p=2628,7721");}
	node50: StandardHost {@display("p=7417,5525");}
	node51: StandardHost {@display("p=7258,7812");}
	node52: StandardHost {@display("p=947,9570");}
	node53: StandardHost {@display("p=7083,8846");}
	node54: StandardHost {@display("p=2138,7694");}
	node55: StandardHost {@display("p=2400,7640");}
	node56: StandardHost {@display("p=1827,6059");}
	node57: StandardHost {@display("p=6688,2457");}
	node58: StandardHost {@display("p=3542,2017");}
	node59: StandardHost {@display("p=3636,9676");}
	node60: StandardHost {@display("p=1712,5111");}
	node61: StandardHost {@display("p=4018,1672");}
	node62: StandardHost {@display("p=1535,9243");}
	node63: StandardHost {@display("p=1787,993");}
	node64: StandardHost {@display("p=3542,2017");}
	node65: StandardHost {@display("p=2082,565");}
	node66: StandardHost {@display("p=1787,993");}
	node67: StandardHost {@display("p=1681,288");}
	node68: StandardHost {@display("p=3663,1777");}
	node69: StandardHost {@display("p=2219,5561");}
	node70: StandardHost {@display("p=6663,10167");}
	node71: StandardHost {@display("p=2098,165");}
	node72: StandardHost {@display("p=1979,7628");}
	node73: StandardHost {@display("p=7212,8974");}
	node74: StandardHost {@display("p=10,8668");}
	node75: StandardHost {@display("p=1864,712");}
	node76: StandardHost {@display("p=6906,8753");}
	node77: StandardHost {@display("p=7493,8060");}
	node78: StandardHost {@display("p=6961,9208");}
	node79: StandardHost {@display("p=7440,3562");}
	node80: StandardHost {@display("p=6895,9624");}
	node81: StandardHost {@display("p=7493,8060");}
	node82: StandardHost {@display("p=1375,5926");}
	node83: StandardHost {@display("p=1266,5540");}
	node84: StandardHost {@display("p=7212,8974");}
	node85: StandardHost {@display("p=1429,6165");}
	node86: StandardHost {@display("p=2138,7694");}
	node87: StandardHost {@display("p=7440,3562");}
	node88: StandardHost {@display("p=2495,578");}
	node89: StandardHost {@display("p=2186,5037");}
	node90: StandardHost {@display("p=1375,5926");}
	node91: StandardHost {@display("p=6633,7025");}
	node92: StandardHost {@display("p=2495,578");}
	node93: StandardHost {@display("p=7143,8659");}
	node94: StandardHost {@display("p=3636,9676");}
	node95: StandardHost {@display("p=1712,5111");}
	node96: StandardHost {@display("p=1733,5799");}
	node97: StandardHost {@display("p=738,9260");}
	node98: StandardHost {@display("p=2628,7721");}
	node99: StandardHost {@display("p=2082,565");}
	node0: StandardHost {@display("p=3821,5920");}
connections:
	node0.pppg++ <--> C1Gbps <--> router_central.pppg++;
	router_0.pppg++ <--> C1Gbps <--> router_central.pppg++;
	router_1.pppg++ <--> C1Gbps <--> router_central.pppg++;
	router_2.pppg++ <--> C1Gbps <--> router_central.pppg++;
	node1.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node2.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node3.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node4.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node5.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node6.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node7.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node8.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node9.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node10.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node11.pppg++ <--> C100kbps <--> router_1.pppg++;
	node12.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node13.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node14.pppg++ <--> C100kbps <--> router_2.pppg++;
	node15.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node16.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node17.pppg++ <--> C10kbps <--> router_2.pppg++;
	node18.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node19.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node20.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node21.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node22.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node23.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node24.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node25.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node26.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node27.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node28.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node29.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node30.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node31.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node32.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node33.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node34.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node35.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node36.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node37.pppg++ <--> C10kbps <--> router_1.pppg++;
	node38.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node39.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node40.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node41.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node42.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node43.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node44.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node45.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node46.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node47.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node48.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node49.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node50.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node51.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node52.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node53.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node54.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node55.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node56.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node57.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node58.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node59.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node60.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node61.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node62.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node63.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node64.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node65.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node66.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node67.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node68.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node69.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node70.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node71.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node72.pppg++ <--> C100kbps <--> router_0.pppg++;
	node73.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node74.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node75.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node76.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node77.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node78.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node79.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node80.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node81.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node82.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node83.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node84.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node85.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node86.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node87.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node88.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node89.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node90.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node91.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node92.pppg++ <--> C1Mbps <--> router_1.pppg++;
	node93.pppg++ <--> C1Mbps <--> router_2.pppg++;
	node94.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node95.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node96.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node97.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node98.pppg++ <--> C1Mbps <--> router_0.pppg++;
	node99.pppg++ <--> C1Mbps <--> router_1.pppg++;
}