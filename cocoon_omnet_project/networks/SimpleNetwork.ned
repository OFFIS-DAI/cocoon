package networks;

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.ethernet.EthernetSwitch;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import ned.DatarateChannel;

import MangoTimeAdvancer;


network SimpleNetwork
{
    types:
        channel fiberline extends DatarateChannel
        {
            delay = 1us;
            datarate = 512Mbps;
        }
        channel ethline extends DatarateChannel
        {
            delay = 5ms;
            datarate = 100Mbps;
        }
    submodules:
        timeAdvancer: MangoTimeAdvancer {
            @display("p=337,40");
        }
        client: StandardHost {
            @display("p=250,100;i=device/laptop");
        }
        node1: StandardHost {
            @display("p=250,200;i=device/pc");
        }
        node2: StandardHost {
            @display("p=250,300;i=device/pc");
        }
        switch: EthernetSwitch {
            @display("p=400,200");
        }
        net: Router {
            @display("p=700,200");
        }
        router: Router {
            @display("p=550,200");
        }
        server: StandardHost {
            @display("p=850,200;i=device/server");
        }
        configurator: Ipv4NetworkConfigurator {
            @display("p=100,100;is=s");
        }
        node3: StandardHost {
            @display("p=327,363;i=device/pc");
        }
    connections:
        client.ethg++ <--> ethline <--> switch.ethg++;
        switch.ethg++ <--> ethline <--> node1.ethg++;
        switch.ethg++ <--> ethline <--> node2.ethg++;
        switch.ethg++ <--> ethline <--> node3.ethg++;
        router.ethg++ <--> ethline <--> switch.ethg++;
        router.pppg++ <--> fiberline <--> net.pppg++;
        server.pppg++ <--> fiberline <--> net.pppg++;
}
