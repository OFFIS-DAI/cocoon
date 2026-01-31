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
        @display("i=block/network2;bgb=400, 400");
    submodules:

        timeAdvancer: MangoTimeAdvancer {
            @display("p=369,29");
        }
        carrierAggregation: CarrierAggregation {
            @display("p=369,160");
        }

        channelControl: LteChannelControl {
            @display("p=369,293");
        }
        configurator: Ipv4NetworkConfigurator {
            @display("p=369,231");
        }
        binder: Binder {
            @display("p=369,88");
        }
        server: StandardHost {
            @display("p=275,361");
        }
        router: Router {
            @display("p=248,298");
        }
        upf: Upf {
            @display("p=214,347");
        }
        gNB0: gNodeB {
            @display("p=200,250");
        }
        node0: StandardHost {
            // central entity 
            @display("p=330,330");
        }
        node1: NRUe {
            //  Buchenweg 2 
            @display("p= 136.19808178127278 , 266.7808129210025 ");
        }
        node2: NRUe {
            //  Buchenweg 4 
            @display("p= 135.34602346195607 , 258.4243683256209 ");
        }
        node3: NRUe {
            //  Buchenweg 1 
            @display("p= 109.55600467708427 , 279.1108704926446 ");
        }
        node4: NRUe {
            //  Buchenweg 3 
            @display("p= 107.35014683264308 , 263.0050863940269 ");
        }
        node5: NRUe {
            //  Buchenweg 5 
            @display("p= 103.98684812395368 , 247.6105312211439 ");
        }
        node6: NRUe {
            //  Buchenweg 6 
            @display("p= 134.07065122557105 , 236.87388666719198 ");
        }
        node7: NRUe {
            //  Buchenweg 7 
            @display("p= 109.70507018500939 , 231.05967176519334 ");
        }
        node8: NRUe {
            //  Buchenweg 9 
            @display("p= 106.0909309007111 , 215.9685475854203 ");
        }
        node9: NRUe {
            //  Buchenweg 11a 
            @display("p= 100.0 , 190.81692989636213 ");
        }
        node10: NRUe {
            //  Buchenweg 11 
            @display("p= 116.87517556815874 , 193.14509646128863 ");
        }
        node11: NRUe {
            //  Buchenweg 8 
            @display("p= 141.10889204329578 , 212.15646655112505 ");
        }
        node12: NRUe {
            //  Buchenweg 13 
            @display("p= 124.6468965325621 , 173.46035425085574 ");
        }
        node13: NRUe {
            //  Buchenweg 15 
            @display("p= 140.71447496837936 , 156.75369920022786 ");
        }
        node14: NRUe {
            //  Buchenweg 17 
            @display("p= 158.76687453780323 , 143.00827796570957 ");
        }
        node15: NRUe {
            //  Buchenweg 19 
            @display("p= 173.0003695912892 , 142.22780754975975 ");
        }
        node16: NRUe {
            //  Buchenweg 21 
            @display("p= 187.83278567378875 , 140.1346326135099 ");
        }
        node17: NRUe {
            //  Buchenweg 23 
            @display("p= 203.3587229428813 , 135.51941210310906 ");
        }
        node18: NRUe {
            //  Buchenweg 25 
            @display("p= 217.60022531513823 , 136.25009013991803 ");
        }
        node19: NRUe {
            //  Buchenweg 25a 
            @display("p= 214.6824416151503 , 100.0 ");
        }
        node20: NRUe {
            //  Buchenweg 27 
            @display("p= 231.66422947729006 , 141.6156514696777 ");
        }
        node21: NRUe {
            //  Buchenweg 29 
            @display("p= 246.33032117126277 , 146.27291464153677 ");
        }
        node22: NRUe {
            //  Buchenweg 29a 
            @display("p= 249.82400997559307 , 138.59848055802286 ");
        }
        node23: NRUe {
            //  Buchenweg 31 
            @display("p= 264.681847693515 , 160.48077862523496 ");
        }
        node24: NRUe {
            //  Buchenweg 33 
            @display("p= 283.37649809534196 , 182.3428132813424 ");
        }
        node25: NRUe {
            //  Buchenweg 35 
            @display("p= 295.98880094324704 , 199.7037243079394 ");
        }
        node26: NRUe {
            //  Buchenweg 37 
            @display("p= 300.5195698341704 , 216.40198030788451 ");
        }
        node27: NRUe {
            //  Buchenweg 39 
            @display("p= 304.4349385824753 , 231.29018875397742 ");
        }
        node28: NRUe {
            //  Buchenweg 41 
            @display("p= 307.13492097333074 , 245.4796044435352 ");
        }
        node29: NRUe {
            //  Buchenweg 10 
            @display("p= 147.43338752392447 , 195.65268810186535 ");
        }
        node30: NRUe {
            //  Buchenweg 12 
            @display("p= 161.3608236007858 , 175.23027331009507 ");
        }
        node31: NRUe {
            //  Buchenweg 14 
            @display("p= 187.3803180376999 , 169.1492010038346 ");
        }
        node32: NRUe {
            //  Buchenweg 16 
            @display("p= 219.28502094047144 , 168.4769586417824 ");
        }
        node33: NRUe {
            //  Buchenweg 18 
            @display("p= 242.42795590654714 , 172.18280640617013 ");
        }
        node34: NRUe {
            //  Buchenweg 20 
            @display("p= 261.9007081936579 , 188.39938639476895 ");
        }
        node35: NRUe {
            //  Buchenweg 22 
            @display("p= 275.12898854282685 , 207.67094972450286 ");
        }
        node36: NRUe {
            //  Buchenweg 24 
            @display("p= 277.7864681102801 , 232.94151109457016 ");
        }
        node37: NRUe {
            //  Buchenweg 26 
            @display("p= 279.5533846797771 , 242.5020223800093 ");
        }
        node38: NRUe {
            //  GEN-WIND-1 
            @display("p= 181.0 , 22.0 ");
        }
        node39: NRUe {
            //  GEN-PV-1 
            @display("p= 167.0 , 65.0 ");
        }
        node40: NRUe {
            //  GEN-CHP-1 
            @display("p= 220.0 , 84.0 ");
        }
        node41: NRUe {
            //  GEN-WIND-2 
            @display("p= 100.0 , 26.0 ");
        }
        node42: NRUe {
            //  GEN-PV-2 
            @display("p= 99.0 , 75.0 ");
        }
        node43: NRUe {
            //  GEN-BESS-1 
            @display("p= 148.0 , 29.0 ");
        }
        node44: NRUe {
            //  LOAD-IND-1 
            @display("p= 23.0 , 88.0 ");
        }
        node45: NRUe {
            //  LOAD-DATA-1 
            @display("p= 248.0 , 54.0 ");
        }
        node46: NRUe {
            //  LOAD-TRAM-1 
            @display("p= 236.0 , 83.0 ");
        }
        node47: NRUe {
            //  LOAD-EV-1 
            @display("p= 203.0 , 67.0 ");
        }
        node48: NRUe {
            //  LOAD-RES-1 
            @display("p= 134.0 , 6.0 ");
        }
        node49: NRUe {
            //  LOAD-COM-1 
            @display("p= 79.0 , 65.0 ");
        }

    connections:
        server.pppg++ <--> Eth10G <--> router.pppg++;
        node0.pppg++ <--> Eth10G <--> router.pppg++;
        router.pppg++ <--> Eth10G <--> upf.filterGate;
        upf.pppg++ <--> Eth10G <--> gNB0.ppp;
}

network EvaluationNetwork5G_20
{
    parameters:
        @display("i=block/network2;bgb=400, 400");
    submodules:

        timeAdvancer: MangoTimeAdvancer {
            @display("p=369,29");
        }
        carrierAggregation: CarrierAggregation {
            @display("p=369,160");
        }

        channelControl: LteChannelControl {
            @display("p=369,293");
        }
        configurator: Ipv4NetworkConfigurator {
            @display("p=369,231");
        }
        binder: Binder {
            @display("p=369,88");
        }
        server: StandardHost {
            @display("p=275,361");
        }
        router: Router {
            @display("p=248,298");
        }
        upf: Upf {
            @display("p=214,347");
        }
        gNB0: gNodeB {
            @display("p=200,250;is=vl");
        }
        node0: StandardHost {
            // central entity 
            @display("p=330,330");
        }
        node1: NRUe {
            //  Buchenweg 2 
            @display("p= 136.19808178127278 , 266.7808129210025 ");
        }
        node2: NRUe {
            //  Buchenweg 4 
            @display("p= 135.34602346195607 , 258.4243683256209 ");
        }
        node3: NRUe {
            //  Buchenweg 1 
            @display("p= 109.55600467708427 , 279.1108704926446 ");
        }
        node4: NRUe {
            //  Buchenweg 3 
            @display("p= 107.35014683264308 , 263.0050863940269 ");
        }
        node5: NRUe {
            //  Buchenweg 5 
            @display("p= 103.98684812395368 , 247.6105312211439 ");
        }
        node6: NRUe {
            //  Buchenweg 6 
            @display("p= 134.07065122557105 , 236.87388666719198 ");
        }
        node7: NRUe {
            //  Buchenweg 7 
            @display("p= 109.70507018500939 , 231.05967176519334 ");
        }
        node8: NRUe {
            //  Buchenweg 9 
            @display("p= 106.0909309007111 , 215.9685475854203 ");
        }
        node9: NRUe {
            //  Buchenweg 11a 
            @display("p= 100.0 , 190.81692989636213 ");
        }
        node10: NRUe {
            //  Buchenweg 11 
            @display("p= 116.87517556815874 , 193.14509646128863 ");
        }
        node11: NRUe {
            //  Buchenweg 8 
            @display("p= 141.10889204329578 , 212.15646655112505 ");
        }
        node12: NRUe {
            //  Buchenweg 13 
            @display("p= 124.6468965325621 , 173.46035425085574 ");
        }
        node13: NRUe {
            //  Buchenweg 15 
            @display("p= 140.71447496837936 , 156.75369920022786 ");
        }
        node14: NRUe {
            //  Buchenweg 17 
            @display("p= 158.76687453780323 , 143.00827796570957 ");
        }
        node15: NRUe {
            //  Buchenweg 19 
            @display("p= 173.0003695912892 , 142.22780754975975 ");
        }
        node16: NRUe {
            //  Buchenweg 21 
            @display("p= 187.83278567378875 , 140.1346326135099 ");
        }
        node17: NRUe {
            //  Buchenweg 23 
            @display("p= 203.3587229428813 , 135.51941210310906 ");
        }
        node18: NRUe {
            //  Buchenweg 25 
            @display("p= 217.60022531513823 , 136.25009013991803 ");
        }
        node19: NRUe {
            //  Buchenweg 25a 
            @display("p= 214.6824416151503 , 100.0 ");
        }
    connections:
        server.pppg++ <--> Eth10G <--> router.pppg++;
        node0.pppg++ <--> Eth10G <--> router.pppg++;
        router.pppg++ <--> Eth10G <--> upf.filterGate;
        upf.pppg++ <--> Eth10G <--> gNB0.ppp;
}

network EvaluationNetwork5G_10
{
    parameters:
        @display("i=block/network2;bgb=400, 400");
    submodules:

        timeAdvancer: MangoTimeAdvancer {
            @display("p=369,29");
        }
        carrierAggregation: CarrierAggregation {
            @display("p=369,160");
        }

        channelControl: LteChannelControl {
            @display("p=369,293");
        }
        configurator: Ipv4NetworkConfigurator {
            @display("p=369,231");
        }
        binder: Binder {
            @display("p=369,88");
        }
        server: StandardHost {
            @display("p=275,361");
        }
        router: Router {
            @display("p=248,298");
        }
        upf: Upf {
            @display("p=214,347");
        }
        gNB0: gNodeB {
            @display("p=200,250");
        }
        node0: StandardHost {
            // central entity 
            @display("p=330,330");
        }
        node1: NRUe {
            //  Buchenweg 2 
            @display("p= 136.19808178127278 , 266.7808129210025 ");
        }
        node2: NRUe {
            //  Buchenweg 4 
            @display("p= 135.34602346195607 , 258.4243683256209 ");
        }
        node3: NRUe {
            //  Buchenweg 1 
            @display("p= 109.55600467708427 , 279.1108704926446 ");
        }
        node4: NRUe {
            //  Buchenweg 3 
            @display("p= 107.35014683264308 , 263.0050863940269 ");
        }
        node5: NRUe {
            //  Buchenweg 5 
            @display("p= 103.98684812395368 , 247.6105312211439 ");
        }
        node6: NRUe {
            //  Buchenweg 6 
            @display("p= 134.07065122557105 , 236.87388666719198 ");
        }
        node7: NRUe {
            //  Buchenweg 7 
            @display("p= 109.70507018500939 , 231.05967176519334 ");
        }
        node8: NRUe {
            //  Buchenweg 9 
            @display("p= 106.0909309007111 , 215.9685475854203 ");
        }
        node9: NRUe {
            //  Buchenweg 11a 
            @display("p= 100.0 , 190.81692989636213 ");
        }

    connections:
        server.pppg++ <--> Eth10G <--> router.pppg++;
        node0.pppg++ <--> Eth10G <--> router.pppg++;
        router.pppg++ <--> Eth10G <--> upf.filterGate;
        upf.pppg++ <--> Eth10G <--> gNB0.ppp;
}

network EvaluationNetwork5G_5
{
    parameters:
        @display("i=block/network2;bgb=400, 400");
    submodules:

        timeAdvancer: MangoTimeAdvancer {
            @display("p=369,29");
        }
        carrierAggregation: CarrierAggregation {
            @display("p=369,160");
        }

        channelControl: LteChannelControl {
            @display("p=369,293");
        }
        configurator: Ipv4NetworkConfigurator {
            @display("p=369,231");
        }
        binder: Binder {
            @display("p=369,88");
        }
        server: StandardHost {
            @display("p=275,361");
        }
        router: Router {
            @display("p=248,298");
        }
        upf: Upf {
            @display("p=214,347");
        }
        gNB0: gNodeB {
            @display("p=200,250");
        }
        node0: StandardHost {
            // central entity 
            @display("p=330,330");
        }
        node1: NRUe {
            //  Buchenweg 2 
            @display("p= 136.19808178127278 , 266.7808129210025 ");
        }
        node2: NRUe {
            //  Buchenweg 4 
            @display("p= 135.34602346195607 , 258.4243683256209 ");
        }
        node3: NRUe {
            //  Buchenweg 1 
            @display("p= 109.55600467708427 , 279.1108704926446 ");
        }
        node4: NRUe {
            //  Buchenweg 3 
            @display("p= 107.35014683264308 , 263.0050863940269 ");
        }
    connections:
        server.pppg++ <--> Eth10G <--> router.pppg++;
        node0.pppg++ <--> Eth10G <--> router.pppg++;
        router.pppg++ <--> Eth10G <--> upf.filterGate;
        upf.pppg++ <--> Eth10G <--> gNB0.ppp;
}

