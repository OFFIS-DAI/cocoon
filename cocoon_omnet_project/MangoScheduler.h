/*
 * mango_scheduler.h
 */
#ifndef __MANGO_SCHEDULER_H
#define __MANGO_SCHEDULER_H

#include <omnetpp.h>
#include <iostream>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>

using namespace omnetpp;

// Forward declare a NetworkMessage class for our simulation messages
class MangoMessage : public cMessage {
private:
    std::string messageId;
    std::string senderId;
    std::string receiverId;
    int64_t messageSize;
    simtime_t creationTime;
    int receiverPort;

public:
    MangoMessage(const char* name = nullptr) : cMessage(name) {}

    void setMessageId(const std::string& id) { messageId = id; }
    std::string getMessageId() const { return messageId; }

    void setSenderId(const std::string& id) { senderId = id; }
    std::string getSenderId() const { return senderId; }

    void setReceiverId(const std::string& id) { receiverId = id; }
    std::string getReceiverId() const { return receiverId; }

    void setMessageSize(int64_t size) { messageSize = size; }
    int64_t getMessageSize() const { return messageSize; }

    void setCreationTime(simtime_t time) { creationTime = time; }
    simtime_t getCreationTime() const { return creationTime; }

    void setReceiverPort(int port) { receiverPort = port; }
      int getReceiverPort() const { return receiverPort; }
};


class MangoScheduler : public cScheduler
{
private:
    // Socket related members
    int serverSocket = -1;
    int clientSocket = -1;
    std::thread* listenerThread = nullptr;
    bool running = false;

    // Message queue and synchronization
    std::queue<std::string> messageQueue;
    std::mutex queueMutex;
    std::condition_variable queueCondition;

    std::list<omnetpp::cModule*> modules = {};

    // Socket configuration
    const int PORT = 8345;
    const char* HOST = "127.0.0.1";

    // Helper methods for socket operations
    void setupServerSocket();
    void listenForMessages();
    void cleanup();

    // For tracking messages and time bounds
    simtime_t maxTimeAdvance = SIMTIME_MAX;

    // Flag to track if termination message from Python was received
    bool terminationReceived = false;

    // Add the message processing method
    void processMessage(const std::string& message);
    cModule *getReceiverModule(std::string module_name);

public:
    MangoScheduler();
    virtual ~MangoScheduler();

    // cScheduler methods
    virtual void startRun() override;
    virtual void endRun() override;
    virtual cEvent *guessNextEvent() override;
    virtual cEvent *takeNextEvent() override;
    virtual void putBackEvent(cEvent *event) override;

    void registerApp(cModule *mod);

    // Socket communication methods
    void sendMessage(const std::string& message);
    std::string receiveMessage(bool blocking = true);
    bool hasMessage();

    virtual std::string str() const override;
};

#endif // __MANGO_SCHEDULER_H
