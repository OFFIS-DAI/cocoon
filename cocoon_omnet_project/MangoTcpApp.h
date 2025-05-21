/*
 * MangoTcpApp.h
 *
 * Created on: May 19, 2025
 * Author: malin
 *
 * The MangoTcpApp represents the application layer for end devices that
 * communicate with the MangoScheduler. It sends and receives messages over
 * TCP and notifies the scheduler about deliveries.
 */

#ifndef MANGOTCPAPP_H_
#define MANGOTCPAPP_H_

#include <omnetpp.h>
#include <json.hpp>
#include <map>
#include <string>

#include "inet/applications/tcpapp/TcpAppBase.h"
#include "inet/common/lifecycle/LifecycleOperation.h"
#include "inet/common/lifecycle/NodeStatus.h"
#include "inet/transportlayer/contract/tcp/TcpSocket.h"

// Include Timer message definition
#include "messages/Timer_m.h"

// Forward declaration for MangoMessage
class MangoMessage;

using namespace omnetpp;
using json = nlohmann::json;

class MangoTcpApp : public inet::TcpAppBase {
protected:
    // Statistics
    int numMessagesSent = 0;
    int numMessagesReceived = 0;

    // Module name for logging
    std::string moduleName;

    // Socket for server mode
    inet::TcpSocket serverSocket;

    // Map of client sockets for outgoing connections
    std::map<int, inet::TcpSocket> clientSockets;

    // Map of destination ports to node names
    std::map<int, std::string> portToName;

    // Map of scheduled messages by time and destination port
    std::map<int, std::map<int, inet::Packet*>> messageMap;

    // Map of connection times to ports
    std::map<int, std::list<int>> connectToTimeToPort;

protected:
    // Initialize the application
    virtual void initialize(int stage) override;
    int numInitStages() const override { return inet::NUM_INIT_STAGES; }

    // Message handling
    virtual void handleTimer(cMessage *msg) {};
    virtual void handleMessage(cMessage *msg) override;

    // Socket event handlers
    virtual void socketEstablished(inet::TcpSocket *socket) override;
    virtual void socketDataArrived(inet::TcpSocket *socket, inet::Packet *msg, bool urgent) override;
    virtual void socketClosed(inet::TcpSocket *socket) override;
    virtual void socketFailure(inet::TcpSocket *socket, int code) override;

    // Lifecycle methods
    virtual void handleStartOperation(inet::LifecycleOperation *operation) override {};
    virtual void handleStopOperation(inet::LifecycleOperation *operation) override {};
    virtual void handleCrashOperation(inet::LifecycleOperation *operation) override {};

    // Connection and sending methods
    virtual void connect() override;
    virtual void sendData(int receiverPort);

public:
    MangoTcpApp();
    virtual ~MangoTcpApp();

    // Finalize statistics
    virtual void finish() override;
};

#endif /* MANGOTCPAPP_H_ */
