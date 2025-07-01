// MangoTcpApp.ned

import inet.applications.contract.IApp;

//
// TCP application for the Mango system that interfaces with MangoScheduler
//
simple MangoTcpApp like IApp
{
    parameters:
        @class(MangoTcpApp);
        @display("i=block/app");
        
        // Server parameters
        string localAddress = default("");  // Local IP address
        int localPort = default(8345);      // Port to listen on
        
        // Client parameters
        int connectPort = default(8345);    // Port to connect to
        int timeToLive = default(-1);       // Socket TTL (-1: use OS default)
        int dscp = default(-1);             // DSCP for sent packets
        int tos = default(-1);              // TOS for sent packets
        
        // Lifecycle
        double stopOperationExtraTime @unit(s) = default(2s);
        double stopOperationTimeout @unit(s) = default(1s);
        
        // Statistics
        @signal[packetSent](type=inet::Packet);
        @signal[packetReceived](type=inet::Packet);
        @statistic[packetSent](title="packets sent"; source=packetSent; record=count,"sum(packetBytes)","vector(packetBytes)"; interpolationmode=none);
        @statistic[packetReceived](title="packets received"; source=packetReceived; record=count,"sum(packetBytes)","vector(packetBytes)"; interpolationmode=none);
        
    gates:
        input socketIn @labels(TcpCommand/up);
        output socketOut @labels(TcpCommand/down);
}