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
#include "inet/common/TimeTag_m.h"
#include "inet/common/TagBase_m.h"
#include "inet/common/IdentityTag_m.h"

#include "MangoTcpApp.h"
#include "messages/MangoMsgTag_m.h"
#include "MangoScheduler.h"

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
        EV << moduleName << " initialized  first stage" << std::endl;
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


        EV << moduleName << " initialized as TCP server at " << localAddress << ":" << localPort << std::endl;
    }
    else {
        TcpAppBase::initialize(stage);
    }

}

void MangoTcpApp::connect() {
    EV << moduleName << " in connect" << endl;
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
void MangoTcpApp::handleMessage(cMessage *msg) {
    // Special case for MangoMessage - handle it directly regardless of module state
    if (strcmp(msg->getName(), "MangoMessage") == 0) {
        // Get the message fields directly
        std::string msgId = check_and_cast<MangoMessage *>(msg)->getMessageId();
        std::string senderId = check_and_cast<MangoMessage *>(msg)->getSenderId();
        std::string receiverId = check_and_cast<MangoMessage *>(msg)->getReceiverId();
        int64_t msgSize = check_and_cast<MangoMessage *>(msg)->getMessageSize();
        int receiverPort = check_and_cast<MangoMessage *>(msg)->getReceiverPort();


        EV << "  Message ID: " << msgId << std::endl;
        EV << "  Sender: " << senderId << std::endl;
        EV << "  Receiver: " << receiverId << " with port number: " << receiverPort << std::endl;
        EV << "  Size: " << msgSize << " bytes" << std::endl;

        if (getParentModule()->getFullName() == senderId) {
            // We are the sender - let's handle everything directly here
            // Set up direct connection to receiver
            const char *localAddress = par("localAddress");
            int localPort = par("localPort");

            // Create a socket for connection
            inet::TcpSocket clientSocket;
            clientSocket.setOutputGate(gate("socketOut"));
            clientSocket.setCallback(this);

            try {
                clientSocket.bind(
                        localAddress[0] ? inet::L3Address(localAddress) : inet::L3Address(),
                                localPort);
            }
            catch (...) {
                EV << "Socket already bound" << std::endl;
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

            // Resolve destination address and connect
            inet::L3Address destination;
            try {
                EV << "Resolving address for " << receiverId << std::endl;
                inet::L3AddressResolver().tryResolve(receiverId.c_str(), destination);

                if (destination.isUnspecified()) {
                    EV << "Cannot resolve address for " << receiverId << std::endl;
                }
                else {
                    EV << "Connecting to " << receiverId << " (" << destination.str() << ":" << receiverPort << ")" << std::endl;

                    // Store socket
                    clientSockets[receiverPort] = clientSocket;

                    // Store mapping
                    portToName[receiverPort] = receiverId;

                    // Connect
                    clientSocket.connect(destination, receiverPort);

                    // Create packet
                    auto data = inet::makeShared<inet::ByteCountChunk>(inet::B(msgSize));
                    auto msgTag = data->addTag<inet::MangoMsgTag>();
                    msgTag->setMsgId(msgId.c_str());
                    msgTag->setMsgSize(msgSize);
                    messageSizeMap[msgId] = 0;
                    inet::Packet *packet = new inet::Packet("Mango Message", data);

                    clientSocket.send(packet);

                    numMessagesSent++;
                }
            }
            catch (std::exception& e) {
                EV << "Error connecting: " << e.what() << std::endl;
            }
        }
        else if (getParentModule()->getFullName() == receiverId) {
            // We are the receiver - expect TCP packets from sender
            EV << "I am the receiver, expecting TCP packets from " << senderId << std::endl;
        }

        // Delete the MangoMessage
        delete msg;
    }
    // Handle incoming data messages
    else if (msg->arrivedOn("socketIn") && strstr(msg->getName(), "data") != nullptr) {
        EV << "Received data message on socketIn: " << msg->getName() << std::endl;

        // This is a data message from a socket
        inet::Packet *packet = dynamic_cast<inet::Packet*>(msg);
        if (packet) {
            // Extract message ID from packet name (if available)
            std::string packetName = packet->getName();
            std::string msgId;
            int expectedMsgSize;

            EV << "total length: " << packet->getTotalLength() << endl;

            auto data = packet->peekData(); // get all data from the packet
            auto regions = data->getAllTags<inet::MangoMsgTag>(); // get all tag regions
            for (auto& region : regions) { // for each region do
                msgId = region.getTag()->getMsgId();
                expectedMsgSize = region.getTag()->getMsgSize();
            }

            // Record statistics
            int bytes = packet->getByteLength();
            numMessagesReceived++;

            EV << "Received packet " << msgId << ", size: " << bytes << " bytes, expected bytes: " << expectedMsgSize << std::endl;
            messageSizeMap[msgId] += bytes;
            EV << "updated to: " << messageSizeMap[msgId] << endl;

            if (messageSizeMap[msgId] == expectedMsgSize) {
                // Create JSON payload for the scheduler
                json responsePayload = {
                        {"msg_id", msgId},
                        {"receiver", moduleName},
                        {"size_B", bytes},
                        {"time_received", simTime().inUnit(SIMTIME_MS)}
                };
                // Format message for scheduler
                std::string responseMsg = "RECEIVED|" + responsePayload.dump();
                // Get scheduler and send the message
                MangoScheduler *scheduler = check_and_cast<MangoScheduler *>(getSimulation()->getScheduler());
                if (scheduler) {
                    scheduler->sendMessage(responseMsg);
                    EV << "Sent message receipt notification to scheduler: " << responseMsg << std::endl;
                } else {
                    std::cerr << "Could not get scheduler to notify of received message" << std::endl;
                }
            }

            // Signal reception
            emit(inet::packetReceivedSignal, packet);
        }

        // Pass to socket message handling
        bool processed = false;
        for (auto& socketPair : clientSockets) {
            if (socketPair.second.belongsToSocket(msg)) {
                socketPair.second.processMessage(msg);
                processed = true;
                break;
            }
        }

        if (!processed && serverSocket.belongsToSocket(msg)) {
            serverSocket.processMessage(msg);
            processed = true;
        }

        if (!processed) {
            delete msg;
        }
    }
    // Handle socket notifications directly - like "ESTABLISHED"
    else if (msg->arrivedOn("socketIn") ||
            strstr(msg->getName(), "ESTAB") ||
            strstr(msg->getName(), "CONN") ||
            strstr(msg->getName(), "DATA") ||
            strstr(msg->getName(), "SOCK")) {

        // Process socket message
        if (socket.belongsToSocket(msg)) {
            socket.processMessage(msg);
        }
        else {
            // Check if it belongs to one of our client sockets
            bool processed = false;
            for (auto& socketPair : clientSockets) {
                if (socketPair.second.belongsToSocket(msg)) {
                    socketPair.second.processMessage(msg);
                    processed = true;
                    break;
                }
            }

            // If server socket
            if (!processed && serverSocket.belongsToSocket(msg)) {
                serverSocket.processMessage(msg);
                processed = true;
            }

            // If not processed by any socket, delete it
            if (!processed) {
                EV << "Socket message not processed by any socket, deleting: " << msg->getName() << std::endl;
                delete msg;
            }
        }
    }
    // Handle Timer messages
    else if (msg != nullptr && (strcmp(msg->getName(), "ConnectTimer") == 0 || dynamic_cast<Timer*>(msg) != nullptr)) {
        Timer *timer = dynamic_cast<Timer*>(msg);
        if (timer != nullptr) {
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
                EV << "Unknown timer type: " << timer->getTimerType() << std::endl;
                break;
            }
        } else {
            // It's a ConnectTimer but not a Timer class
            connect();
        }

        delete msg;
    }
    else {
        // For all other messages, use the standard ApplicationBase handler with a try/catch
        try {
            ApplicationBase::handleMessage(msg);
        } catch (const std::exception& e) {
            // If there's an exception (like module is down), handle it here
            EV << "Exception in ApplicationBase::handleMessage: " << e.what() << std::endl;
            delete msg;
        } catch (...) {
            // Catch any other exceptions
            EV << "Unknown exception in ApplicationBase::handleMessage" << std::endl;
            delete msg;
        }
    }
}

void MangoTcpApp::socketClosed(inet::TcpSocket *socket) {
    TcpAppBase::socketClosed(socket);

    if (operationalState == State::STOPPING_OPERATION && !this->socket.isOpen())
        startActiveOperationExtraTimeOrFinish(par("stopOperationExtraTime"));
}

void MangoTcpApp::socketFailure(inet::TcpSocket *socket, int code) {
    TcpAppBase::socketFailure(socket, code);
}

void MangoTcpApp::finish() {
    EV << moduleName << " (" << simTime().inUnit(SIMTIME_MS)
                                                            << "): Received " << numMessagesReceived << " messages and sent "
                                                            << numMessagesSent << " messages" << std::endl;
}
