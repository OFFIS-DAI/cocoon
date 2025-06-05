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
import simu5g.nodes.backgroundCell.BackgroundCell;
import simu5g.world.radio.LteChannelControl;

import MangoTimeAdvancer;

network SimbenchNetwork5G
{
    parameters:
        @display("i=block/network2;bgb=379.35363110608887,1756.448845718056");
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
        router: Router {
            @display("p=100,100;is=vl");
        }
        upf: Upf {
            @display("p=100,100;is=l");
        }
        iUpf: Upf {
            @display("p=100,100;is=s");
        }
        gNB0: gNodeB {
            @display("p=189,1000;is=vl");
        }
        bgCell: BackgroundCell {
            @display("p=179,868;is=vl");
        }
        node0: NRUe {
            @display("p=62,1246");
        }
        node1: NRUe {
            @display("p=62,1246");
        }
        node2: NRUe {
            @display("p=306,1655");
        }
        node3: NRUe {
            @display("p=310,1543");
        }
        node4: NRUe {
            @display("p=336,976");
        }
        node5: NRUe {
            @display("p=336,976");
        }
        node6: NRUe {
            @display("p=221,1262");
        }
        node7: NRUe {
            @display("p=221,1262");
        }
        node8: NRUe {
            @display("p=221,1262");
        }
        node9: NRUe {
            @display("p=316,1377");
        }
        node10: NRUe {
            @display("p=336,208");
        }
        node11: NRUe {
            @display("p=339,687");
        }
        node12: NRUe {
            @display("p=339,687");
        }
        node13: NRUe {
            @display("p=312,1488");
        }
        node14: NRUe {
            @display("p=350,387");
        }
        node15: NRUe {
            @display("p=36,1634");
        }
        node16: NRUe {
            @display("p=64,1190");
        }
        node17: NRUe {
            @display("p=215,1429");
        }
        node18: NRUe {
            @display("p=338,921");
        }
        node19: NRUe {
            @display("p=338,921");
        }
        node20: NRUe {
            @display("p=127,1303");
        }
        node21: NRUe {
            @display("p=389,210");
        }
        node22: NRUe {
            @display("p=344,732");
        }
        node23: NRUe {
            @display("p=344,732");
        }
        node24: NRUe {
            @display("p=117,1581");
        }
        node25: NRUe {
            @display("p=128,1248");
        }
        node26: NRUe {
            @display("p=13,1144");
        }
        node27: NRUe {
            @display("p=220,1485");
        }
        node28: NRUe {
            @display("p=344,732");
        }
        node29: NRUe {
            @display("p=344,732");
        }
        node30: NRUe {
            @display("p=328,1021");
        }
        node31: NRUe {
            @display("p=345,531");
        }
        node32: NRUe {
            @display("p=345,531");
        }
        node33: NRUe {
            @display("p=308,1599");
        }
        node34: NRUe {
            @display("p=334,832");
        }
        node35: NRUe {
            @display("p=111,1748");
        }
        node36: NRUe {
            @display("p=324,1332");
        }
        node37: NRUe {
            @display("p=324,1332");
        }
        node38: NRUe {
            @display("p=228,1040");
        }
        node39: NRUe {
            @display("p=228,1040");
        }
        node40: NRUe {
            @display("p=123,1415");
        }
        node41: NRUe {
            @display("p=123,1415");
        }
        node42: NRUe {
            @display("p=230,995");
        }
        node43: NRUe {
            @display("p=230,995");
        }
        node44: NRUe {
            @display("p=10,1244");
        }
        node45: NRUe {
            @display("p=314,1432");
        }
        node46: NRUe {
            @display("p=121,1470");
        }
        node47: NRUe {
            @display("p=19,955");
        }
        node48: NRUe {
            @display("p=19,955");
        }
        node49: NRUe {
            @display("p=349,587");
        }
        node50: NRUe {
            @display("p=29,855");
        }
        node51: NRUe {
            @display("p=33,1523");
        }
        node52: NRUe {
            @display("p=33,1523");
        }
        node53: NRUe {
            @display("p=33,1523");
        }
        node54: NRUe {
            @display("p=16,1044");
        }
        node55: NRUe {
            @display("p=208,1640");
        }
        node56: NRUe {
            @display("p=208,1640");
        }
        node57: NRUe {
            @display("p=208,1640");
        }
        node58: NRUe {
            @display("p=332,308");
        }
        node59: NRUe {
            @display("p=341,631");
        }
        node60: NRUe {
            @display("p=383,176");
        }
        node61: NRUe {
            @display("p=337,164");
        }
        node62: NRUe {
            @display("p=122,1637");
        }
        node63: NRUe {
            @display("p=11,1188");
        }
        node64: NRUe {
            @display("p=210,1585");
        }
        node65: NRUe {
            @display("p=385,132");
        }
        node66: NRUe {
            @display("p=136,1036");
        }
        node67: NRUe {
            @display("p=31,1579");
        }
        node68: NRUe {
            @display("p=21,899");
        }
        node69: NRUe {
            @display("p=324,1332");
        }
        node70: NRUe {
            @display("p=324,1332");
        }
        node71: NRUe {
            @display("p=338,921");
        }
        node72: NRUe {
            @display("p=338,921");
        }
        node73: NRUe {
            @display("p=238,940");
        }
        node74: NRUe {
            @display("p=346,487");
        }
        node75: NRUe {
            @display("p=340,86");
        }
        node76: NRUe {
            @display("p=340,86");
        }
        node77: NRUe {
            @display("p=340,86");
        }
        node78: NRUe {
            @display("p=43,1412");
        }
        node79: NRUe {
            @display("p=43,1412");
        }
        node80: NRUe {
            @display("p=327,264");
        }
        node81: NRUe {
            @display("p=326,1076");
        }
        node82: NRUe {
            @display("p=211,1540");
        }
        node83: NRUe {
            @display("p=211,1540");
        }
        node84: NRUe {
            @display("p=332,130");
        }
        node85: NRUe {
            @display("p=219,1318");
        }
        node86: NRUe {
            @display("p=224,1151");
        }
        node87: NRUe {
            @display("p=389,10");
        }
        node88: NRUe {
            @display("p=134,1081");
        }
        node89: NRUe {
            @display("p=321,1221");
        }
        node90: NRUe {
            @display("p=386,87");
        }
        node91: NRUe {
            @display("p=380,265");
        }
        node92: NRUe {
            @display("p=323,1176");
        }
        node93: NRUe {
            @display("p=14,1099");
        }
        node94: NRUe {
            @display("p=132,1137");
        }
        node95: NRUe {
            @display("p=60,1301");
        }
        node96: NRUe {
            @display("p=348,442");
        }
        node97: NRUe {
            @display("p=348,442");
        }
        node98: NRUe {
            @display("p=348,442");
        }
        node99: NRUe {
            @display("p=336,776");
        }

    connections:
        server.pppg++ <--> Eth10G <--> router.pppg++;
        router.pppg++ <--> Eth10G <--> upf.filterGate;
        upf.pppg++ <--> Eth10G <--> iUpf.pppg++;
        iUpf.pppg++ <--> Eth10G <--> gNB0.ppp;
}
