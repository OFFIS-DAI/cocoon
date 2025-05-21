/*
 * MangoTcpApp.cc
 *
 * Created on: May 19, 2025
 * Author: malin
 */

#include <cmath>
#include <omnetpp.h>
#include <string.h>
#include <algorithm>

#include "inet/applications/base/ApplicationPacket_m.h"
#include "inet/common/ModuleAccess.h"
#include "inet/common/TagBase_m.h"
#include "inet/common/TimeTag_m.h"
#include "inet/common/lifecycle/ModuleOperations.h"
#include "inet/common/packet/Packet.h"
#include "inet/common/packet/chunk/ByteCountChunk.h"
#include "inet/common/packet/chunk/BytesChunk.h"
#include "inet/networklayer/common/L3AddressResolver.h"

#include "MangoTcpApp.h"
#include "messages/MangoMessage_m.h"  // Include the generated message class
#include "mango_scheduler.h"

Define_Module(MangoTcpApp);

MangoTcpApp::MangoTcpApp() {
}

MangoTcpApp::~MangoTcpApp() {
    // Clean up any packets still in the message map
    for (auto& portEntry : messageMap) {
        for (auto& timeEntry : portEntry.second) {
            delete timeEntry.second;
        }
    }
}

void MangoTcpApp::initialize(int stage) {
    if (stage == inet::INITSTAGE_LOCAL) {
        // Initialize statistics
        numMessagesSent = 0;
        numMessagesReceived = 0;

        // Get module name for logging
        moduleName = getParentModule()->getFullName();

        // Register signals for statistics
        WATCH(numMessagesSent);
        WATCH(numMessagesReceived);
    }
    else if (stage == inet::INITSTAGE_APPLICATION_LAYER) {
        const char *localAddress = par("localAddress");
        int localPort = par("localPort");

        // Set up server socket for incoming connections
        serverSocket.setOutputGate(gate("socketOut"));
        serverSocket.setCallback(this);

        serverSocket.bind(
            localAddress[0] ? inet::L3Address(localAddress) : inet::L3Address(),
            localPort);
        serverSocket.listen();

        MangoScheduler *scheduler = check_and_cast<MangoScheduler *>(getSimulation()->getScheduler());
        scheduler->registerApp(this);


        EV_INFO << moduleName << " initialized as TCP server at " << localAddress << ":" << localPort << std::endl;
    }
    else {
        TcpAppBase::initialize(stage);
    }

}

void MangoTcpApp::handleMessageWhenUp(cMessage *msg) {
    std::cout << "received message in TCP App: " << msg << endl;
    if (typeid(*msg) == typeid(Timer)) {
        // Timer message for scheduled events
        handleTimer(msg);
    }
    else if (msg->arrivedOn("socketIn")) {
        // Message from socket (network)
        socket.processMessage(msg);
    }
    else {
        // TODO: handle message from scheduler.
        std::cout << "received message from scheduler." << endl;
    }
}

void MangoTcpApp::handleTimer(cMessage *msg) {
    std::cout << "handle timer in TCP app" << endl;
    if (msg != nullptr && typeid(*msg) == typeid(Timer)) {
        Timer *timer = dynamic_cast<Timer*>(msg);

        switch (timer->getTimerType()) {
            case 0:  // Connect timer
                connect();
                break;

            case 1:  // Send data timer
                sendData(timer->getReceiverPort());
                break;

            case 2:  // Close connection timer
                close();
                break;

            default:
                EV_WARN << moduleName << ": Unknown timer type " << timer->getTimerType() << std::endl;
                break;
        }
    }
    else {
        EV_ERROR << moduleName << ": Called handleTimer with non-Timer object" << std::endl;
    }

    delete msg;
}

void MangoTcpApp::connect() {
    int currentSimTime = simTime().inUnit(SIMTIME_MS);

    // Find connections scheduled for current time
    auto it = connectToTimeToPort.find(currentSimTime);
    if (it != connectToTimeToPort.end()) {
        std::list<int> ports = connectToTimeToPort[currentSimTime];

        for (auto const& port : ports) {
            EV_INFO << moduleName << " (" << currentSimTime << "): Connecting to port " << port << std::endl;

            inet::TcpSocket clientSocket;
            if (clientSockets.find(port) != clientSockets.end()) {
                clientSocket = clientSockets[port];
            }
            else {
                const char *localAddress = par("localAddress");
                int localPort = par("localPort");

                clientSocket.setOutputGate(gate("socketOut"));
                clientSocket.setCallback(this);

                try {
                    clientSocket.bind(
                        localAddress[0] ? inet::L3Address(localAddress) : inet::L3Address(),
                        localPort);
                }
                catch (...) {
                    EV_ERROR << moduleName << " (" << currentSimTime << "): Socket already bound" << std::endl;
                }
            }

            // Renew socket for new connection
            clientSocket.renewSocket();

            // Set socket parameters
            int timeToLive = par("timeToLive");
            if (timeToLive != -1)
                clientSocket.setTimeToLive(timeToLive);

            int dscp = par("dscp");
            if (dscp != -1)
                clientSocket.setDscp(dscp);

            int tos = par("tos");
            if (tos != -1)
                clientSocket.setTos(tos);

            // Resolve destination address
            inet::L3Address destination;
            try {
                const char *connectAddress = portToName[port].c_str();

                inet::L3AddressResolver().tryResolve(connectAddress, destination);
                if (destination.isUnspecified()) {
                    EV_ERROR << moduleName << " (" << currentSimTime << "): Cannot resolve destination address "
                             << connectAddress << std::endl;
                }
                else {
                    EV_INFO << moduleName << " (" << currentSimTime << "): Connecting to "
                            << connectAddress << "(" << destination.str() << ") and port " << port << std::endl;

                    clientSocket.connect(destination, port);
                    emit(connectSignal, 1L);
                }

                clientSockets[port] = clientSocket;
            }
            catch (...) {
                EV_ERROR << moduleName << " (" << currentSimTime << "): Error resolving L3 address" << std::endl;
            }
        }

        // Remove processed entry
        connectToTimeToPort.erase(it);
    }

    // Schedule next connection timer
    auto minElement = std::min_element(connectToTimeToPort.begin(), connectToTimeToPort.end(),
        [](const auto& a, const auto& b) {
            return a.first < b.first && !b.second.empty();
        });

    if (minElement != connectToTimeToPort.end()) {
        Timer *timer = new Timer();
        timer->setTimerType(0);
        simtime_t nextTime = SimTime(minElement->first, SIMTIME_MS);

        EV_INFO << moduleName << " (" << currentSimTime << "): Schedule next connect timer for "
                << minElement->first << std::endl;

        if (nextTime < simTime()) {
            EV_ERROR << moduleName << " (" << currentSimTime << "): Timer scheduled in the past ("
                     << nextTime.str() << ")" << std::endl;
        }
        else {
            scheduleAt(nextTime, timer);
        }
    }
}

void MangoTcpApp::socketEstablished(inet::TcpSocket *socket) {
    int port = socket->getRemotePort();

    if (clientSockets.find(port) == clientSockets.end()) {
        EV_ERROR << moduleName << " (" << simTime().inUnit(SIMTIME_MS)
                 << "): Established socket not saved in map: " << port << std::endl;
    }
    else {
        // Schedule send timer
        Timer *sendTimer = new Timer();
        sendTimer->setTimerType(1);  // Send timer
        sendTimer->setReceiverPort(port);
        scheduleAt(simTime(), sendTimer);
    }
}

void MangoTcpApp::socketDataArrived(inet::TcpSocket *socket, inet::Packet *msg, bool urgent) {
    EV_INFO << moduleName << " (" << simTime().inUnit(SIMTIME_MS)
            << "): Data arrived from " << socket->getRemoteAddress() << ":" << socket->getRemotePort() << std::endl;

    int delay = simTime().inUnit(SIMTIME_MS) - msg->getSendingTime().inUnit(SIMTIME_MS);
    int bytes = msg->getByteLength();

    numMessagesReceived++;
    emit(inet::packetReceivedSignal, msg);

    // Process packet to extract MangoMessage
    inet::Packet *packet = dynamic_cast<inet::Packet*>(msg);

    // TODO: handle message

    // Let the base class handle the socket
    TcpAppBase::socketDataArrived(socket, msg, urgent);

    // Close the socket after processing
    socket->close();
}

void MangoTcpApp::socketClosed(inet::TcpSocket *socket) {
    TcpAppBase::socketClosed(socket);

    if (operationalState == State::STOPPING_OPERATION && !this->socket.isOpen())
        startActiveOperationExtraTimeOrFinish(par("stopOperationExtraTime"));
}

void MangoTcpApp::socketFailure(inet::TcpSocket *socket, int code) {
    TcpAppBase::socketFailure(socket, code);

    // Handle any cleanup needed after failure
    EV_ERROR << moduleName << " (" << simTime().inUnit(SIMTIME_MS)
             << "): Socket failure with code " << code << std::endl;
}

void MangoTcpApp::sendData(int receiverPort) {
    if (clientSockets.find(receiverPort) == clientSockets.end()) {
        EV_ERROR << moduleName << " (" << simTime().inUnit(SIMTIME_MS)
                 << "): Receiver port " << receiverPort << " not found in sockets" << std::endl;
        return;
    }

    inet::TcpSocket& clientSocket = clientSockets[receiverPort];

    // Find earliest message for this port
    int currentTime = simTime().inUnit(SIMTIME_MS);
    int earliestTime = INT_MAX;

    for (const auto& elem : messageMap[receiverPort]) {
        if (elem.first < earliestTime) {
            earliestTime = elem.first;
        }
    }

    if (earliestTime > currentTime) {
        EV_INFO << moduleName << " (" << currentTime << "): Will send packet later at "
                << earliestTime << std::endl;
        return;
    }

    if (messageMap[receiverPort].count(earliestTime)) {
        inet::Packet *packet = messageMap[receiverPort][earliestTime];

        EV_INFO << moduleName << " (" << currentTime << "): Sending data packet" << std::endl;

        int numBytes = packet->getByteLength();
        emit(inet::packetSentSignal, packet);

        clientSocket.send(packet);

        numMessagesSent++;

        // Clean up after sending
        messageMap[receiverPort].erase(earliestTime);
    }
    else {
        EV_ERROR << moduleName << " (" << currentTime
                 << "): Earliest time not in message map" << std::endl;
    }
}

void MangoTcpApp::handleStartOperation(inet::LifecycleOperation *operation) {
    const char *localAddress = par("localAddress");
    int localPort = par("localPort");

    serverSocket.setOutputGate(gate("socketOut"));
    serverSocket.setCallback(this);

    serverSocket.bind(
        localAddress[0] ? inet::L3Address(localAddress) : inet::L3Address(),
        localPort);
    serverSocket.listen();

    // Schedule initial connect timer if needed
    if (!connectToTimeToPort.empty()) {
        Timer *timer = new Timer();
        timer->setTimerType(0);  // Connect timer
        scheduleAt(simTime(), timer);
    }
}

void MangoTcpApp::handleStopOperation(inet::LifecycleOperation *operation) {
    if (socket.isOpen())
        close();

    delayActiveOperationFinish(par("stopOperationTimeout"));
}

void MangoTcpApp::handleCrashOperation(inet::LifecycleOperation *operation) {
    // Nothing special needed for crash
}


void MangoTcpApp::finish() {
    EV_INFO << moduleName << " (" << simTime().inUnit(SIMTIME_MS)
            << "): Received " << numMessagesReceived << " messages and sent "
            << numMessagesSent << " messages" << std::endl;
}
