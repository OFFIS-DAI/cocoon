/*
 * MangoScheduler.h
 */

#ifndef MANGOSCHEDULER_H_
#define MANGOSCHEDULER_H_

#include <omnetpp.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>
#include <vector>
#include <map>

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

    void setSimulationDuration(simtime_t duration);

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

// Structure to hold pending event data from listener thread
struct PendingEventData {
    std::string messageId;
    std::string senderId;
    std::string receiverId;
    int64_t messageSize;
    int receiverPort;
    double timeSendMs;
};

// Structure to hold pending configuration data
struct PendingConfigData {
    int simulationDuration;
};

// Structure to hold pending time advance data
struct PendingTimeAdvanceData {
    double maxAdvanceMs;
};

class MangoScheduler : public cScheduler
{
private:
    static const int PORT = 8345;

    // Socket management
    int serverSocket = -1;
    int clientSocket = -1;

    // Thread management
    std::thread* listenerThread = nullptr;
    std::atomic<bool> running{false};
    std::atomic<bool> terminationReceived{false};

    // Thread-safe communication between listener and main thread
    std::mutex pendingDataMutex;
    std::queue<PendingEventData> pendingEvents;
    std::queue<PendingConfigData> pendingConfigs;
    std::queue<PendingTimeAdvanceData> pendingTimeAdvances;
    std::atomic<bool> hasPendingData{false};

    // Legacy message queue (keeping for compatibility)
    std::mutex queueMutex;
    std::queue<std::string> messageQueue;
    std::condition_variable queueCondition;

    // Thread-safe socket sending
    std::mutex sendMutex;

    // Simulation control
    simtime_t maxTimeAdvance = 0;
    simtime_t simulationDuration = SimTime::ZERO;

    // Module management
    std::vector<cModule*> modules;

    // Helper methods
    void setupServerSocket();
    void listenForMessages();
    void processMessage(const std::string& message);
    void processPendingData(); // New method to process data from listener thread

    void cleanup();

public:
    MangoScheduler();
    virtual ~MangoScheduler();

    // cScheduler interface
    virtual std::string str() const override;
    virtual void startRun() override;
    virtual void endRun() override;
    virtual cEvent* guessNextEvent() override;
    virtual cEvent* takeNextEvent() override;
    virtual void putBackEvent(cEvent* event) override;


    // Module registration
    void registerApp(cModule *mod);
    cModule *getReceiverModule(std::string module_name);

    // Message handling (legacy)
    std::string receiveMessage(bool blocking = false);
    bool hasMessage();
    void sendMessage(const std::string& message);
};

#endif /* MANGOSCHEDULER_H_ */
