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

using namespace omnetpp;

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

    // Socket configuration
    const int PORT = 8345;
    const char* HOST = "127.0.0.1";

    // Helper methods for socket operations
    void setupServerSocket();
    void listenForMessages();

    // For tracking messages and time bounds
    simtime_t maxTimeAdvance = SIMTIME_MAX;

    // Add the message processing method
    void processMessage(const std::string& message);

public:
    MangoScheduler();
    virtual ~MangoScheduler();

    // cScheduler methods
    virtual void startRun() override;
    virtual void endRun() override;
    virtual cEvent *guessNextEvent() override;
    virtual cEvent *takeNextEvent() override;
    virtual void putBackEvent(cEvent *event) override;

    // Socket communication methods
    void sendMessage(const std::string& message);
    std::string receiveMessage(bool blocking = true);
    bool hasMessage();

    virtual std::string str() const override;
};

#endif // __MANGO_SCHEDULER_H
