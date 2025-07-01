/*
 * mango_scheduler.cc
 */
extern "C" {
#include <pthread.h>
#include <signal.h>
}
#include "MangoScheduler.h"

#include <json.hpp>
using json = nlohmann::json;

// Signal handler flag to detect interruptions
volatile sig_atomic_t sigintReceived = 0;

// Signal handler for SIGINT
void handleSignal(int signal) {
    if (signal == SIGINT) {
        sigintReceived = 1;
        std::cout << "SIGINT received, preparing for graceful shutdown..." << std::endl;
    }
}

class AdvanceTimeEvent : public cMessage {
public:
    AdvanceTimeEvent() : cMessage("AdvanceTimeEvent") {}
};

Register_Class(MangoScheduler);

MangoScheduler::MangoScheduler()
{
    std::cout << "MangoScheduler initialized." << std::endl;
    terminationReceived = false;

    // Set up signal handler for graceful interruption
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handleSignal;
    sigaction(SIGINT, &sa, NULL);
}

MangoScheduler::~MangoScheduler()
{
    cleanup();
}

void MangoScheduler::cleanup() {
    if (running) {
        running = false;
        if (listenerThread && listenerThread->joinable()) {
            listenerThread->join();
            delete listenerThread;
            listenerThread = nullptr;
        }
    }

    if (clientSocket >= 0) {
        close(clientSocket);
        clientSocket = -1;
    }

    if (serverSocket >= 0) {
        close(serverSocket);
        serverSocket = -1;
    }
}

void MangoScheduler::registerApp(cModule *mod) {
    std::cout << "App registered: " << mod->getFullPath() << std::endl;
    modules.push_back(mod);
}

cModule *MangoScheduler::getReceiverModule(std::string module_name) {
    cModule *matchingModule = nullptr;

    std::cout << "Looking for module: " << module_name << std::endl;
    std::cout << "Available modules: " << modules.size() << std::endl;

    for (auto const& module : modules) {
        if (!module) continue; // Safety check

        cModule* parent = module->getParentModule();
        if (!parent) continue; // Safety check

        std::string moduleNameCheck = std::string(parent->getName());
        std::cout << "Checking module: " << moduleNameCheck << std::endl;
        if (moduleNameCheck.compare(module_name) == 0) {
            matchingModule = module;
            std::cout << "Found matching module: " << module->getFullPath() << std::endl;
            break;
        }
    }
    return matchingModule;
}

std::string MangoScheduler::str() const
{
    return "MangoScheduler (Python integration)";
}

void MangoScheduler::setupServerSocket() {
    // Create socket
    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket < 0) {
        throw cRuntimeError("Failed to create server socket");
    }

    // Set socket options for reuse
    int opt = 1;
    if (setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        close(serverSocket);
        throw cRuntimeError("Failed to set socket options");
    }

    // Prepare server address
    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(PORT);

    // Bind socket
    if (bind(serverSocket, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0) {
        close(serverSocket);
        throw cRuntimeError("Failed to bind server socket");
    }

    // Listen for connections
    if (listen(serverSocket, 1) < 0) {
        close(serverSocket);
        throw cRuntimeError("Failed to listen on server socket");
    }

    std::cout << "Server socket set up, waiting for Python client to connect on port " << PORT << std::endl;

    // Set socket as non-blocking
    int flags = fcntl(serverSocket, F_GETFL, 0);
    fcntl(serverSocket, F_SETFL, flags | O_NONBLOCK);

    // Wait for connection with timeout
    int timeout = 30; // seconds
    time_t startTime = time(NULL);

    while (!sigintReceived && time(NULL) - startTime < timeout) {
        struct sockaddr_in clientAddr;
        socklen_t clientAddrLen = sizeof(clientAddr);
        clientSocket = accept(serverSocket, (struct sockaddr *)&clientAddr, &clientAddrLen);

        if (clientSocket >= 0) {
            flags = fcntl(clientSocket, F_GETFL, 0);
            fcntl(clientSocket, F_SETFL, flags | O_NONBLOCK);

            std::cout << "Python client connected" << std::endl;

            running = true;
            listenerThread = new std::thread(&MangoScheduler::listenForMessages, this);
            return;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (sigintReceived) {
            std::cout << "Received interrupt while waiting for client connection" << std::endl;
            break;
        }
    }

    if (clientSocket < 0) {
        if (time(NULL) - startTime >= timeout) {
            std::cout << "Timeout waiting for Python client connection" << std::endl;
        }
        close(serverSocket);
        serverSocket = -1;
        throw cRuntimeError("Failed to accept client connection");
    }
}

void MangoScheduler::listenForMessages() {
    char buffer[4096];

    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 100000; // 100ms
    setsockopt(clientSocket, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);

    while (running && !sigintReceived) {
        memset(buffer, 0, sizeof(buffer));

        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(clientSocket, &readfds);

        struct timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = 50000; // 50ms

        int activity = select(clientSocket + 1, &readfds, NULL, NULL, &timeout);

        if (activity < 0 && errno != EINTR) {
            std::cerr << "Error in select: " << strerror(errno) << std::endl;
            break;
        }

        if (activity > 0 && FD_ISSET(clientSocket, &readfds)) {
            ssize_t bytesRead = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);

            if (bytesRead > 0) {
                std::string message(buffer, bytesRead);
                std::cout << "Received message from Python: " << message << std::endl;

                // THREAD SAFETY FIX: Only parse and queue data, don't manipulate OMNeT++ objects
                processMessage(message);

                // Legacy message queue for compatibility
                {
                    std::lock_guard<std::mutex> lock(queueMutex);
                    messageQueue.push(message);
                }
                queueCondition.notify_one();
            }
            else if (bytesRead == 0) {
                std::cout << "Python client disconnected" << std::endl;
                break;
            }
            else {
                if (errno != EAGAIN && errno != EWOULDBLOCK) {
                    std::cerr << "Error reading from socket: " << strerror(errno) << std::endl;
                    break;
                }
            }
        }

        if (sigintReceived) {
            std::cout << "Listener thread detected interruption signal" << std::endl;
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "Listener thread exiting" << std::endl;
}

// THREAD SAFETY FIX: This method now only parses and queues data
void MangoScheduler::processMessage(const std::string& message) {
    try {
        size_t delimiterPos = message.find('|');
        if (delimiterPos == std::string::npos) {
            std::cerr << "Invalid message format: " << message << std::endl;
            return;
        }

        std::string type = message.substr(0, delimiterPos);
        std::string payload = message.substr(delimiterPos + 1);

        if (type == "CONFIG") {
            json data = json::parse(payload);
            if (data.contains("simulation_duration")) {
                PendingConfigData config;
                config.simulationDuration = data["simulation_duration"];

                {
                    std::lock_guard<std::mutex> lock(pendingDataMutex);
                    pendingConfigs.push(config);
                    hasPendingData = true;
                }
            }
            sendMessage("CONFIG_ACK|Configuration received");
        }
        else if (type == "MESSAGE") {
            json data = json::parse(payload);
            double max_advance = data["max_advance"];

            // Queue time advance update
            PendingTimeAdvanceData timeAdvance;
            timeAdvance.maxAdvanceMs = max_advance;

            json messages = data["messages"];

            {
                std::lock_guard<std::mutex> lock(pendingDataMutex);
                pendingTimeAdvances.push(timeAdvance);

                // Queue all message events
                for (const auto& msg : messages) {
                    PendingEventData eventData;
                    eventData.messageId = msg["msg_id"];
                    eventData.senderId = msg["sender"];
                    eventData.receiverId = msg["receiver"];
                    eventData.messageSize = msg["size_B"];
                    eventData.timeSendMs = msg["time_send_ms"];
                    eventData.receiverPort = 8345; // default

                    pendingEvents.push(eventData);
                }
                hasPendingData = true;
            }

            // For now, send immediate response (could be improved)
            json response;
            std::vector<std::string> scheduled_ids;
            for (const auto& msg : messages) {
                scheduled_ids.push_back(msg["msg_id"]);
            }
            response["scheduled"] = scheduled_ids;
            response["errors"] = json::array();
            sendMessage("SCHEDULED|" + response.dump());
        }
        else if (type == "TERMINATE") {
            std::cout << "Received termination signal from Python client." << std::endl;
            terminationReceived = true;
            sendMessage("TERM_ACK|Acknowledged termination request");
        }
        else if (type == "WAITING") {
            json data = json::parse(payload);
            double max_advance = data["max_advance"];

            PendingTimeAdvanceData timeAdvance;
            timeAdvance.maxAdvanceMs = max_advance;

            {
                std::lock_guard<std::mutex> lock(pendingDataMutex);
                pendingTimeAdvances.push(timeAdvance);
                hasPendingData = true;
            }

            sendMessage("WAITING_ACK|Time advance scheduled");
        }
        else {
            std::cout << "Unhandled message type: " << type << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error processing message: " << e.what() << std::endl;
        sendMessage("ERROR|" + std::string(e.what()));
    }
}

// NEW METHOD: Process pending data from listener thread in main thread
void MangoScheduler::processPendingData() {
    std::lock_guard<std::mutex> lock(pendingDataMutex);

    // Process pending configurations
    while (!pendingConfigs.empty()) {
        auto config = pendingConfigs.front();
        pendingConfigs.pop();

        simulationDuration = SimTime(config.simulationDuration, SIMTIME_MS);
        std::cout << "Applied simulation duration: " << simulationDuration.str() << std::endl;
    }

    // Process pending time advances
    while (!pendingTimeAdvances.empty()) {
        auto timeAdvance = pendingTimeAdvances.front();
        pendingTimeAdvances.pop();

        maxTimeAdvance = SimTime(timeAdvance.maxAdvanceMs, SIMTIME_MS);
        std::cout << "Applied max time advance: " << maxTimeAdvance.str() << std::endl;
    }

    // Process pending events
    while (!pendingEvents.empty()) {
        auto eventData = pendingEvents.front();
        pendingEvents.pop();

        // Now safely create OMNeT++ objects in main thread
        cModule* senderModule = getReceiverModule(eventData.senderId);
        cModule* receiverModule = getReceiverModule(eventData.receiverId);

        if (receiverModule) {
            try {
                cPar& portPar = receiverModule->par("localPort");
                if (portPar.isSet()) {
                    eventData.receiverPort = portPar.intValue();
                }
            } catch (...) {
                // Use default port if parameter access fails
            }
        }

        if (senderModule) {
            std::cout << "Creating MangoMessage for sender: " << senderModule->getFullPath() << std::endl;

            MangoMessage* mangoMsg = new MangoMessage("MangoMessage");
            mangoMsg->setMessageId(eventData.messageId);
            mangoMsg->setSenderId(eventData.senderId);
            mangoMsg->setReceiverId(eventData.receiverId);
            mangoMsg->setMessageSize(eventData.messageSize);
            mangoMsg->setReceiverPort(eventData.receiverPort);

            simtime_t eventTime = SimTime(eventData.timeSendMs, SIMTIME_MS);
            simtime_t currentTime = simTime();
            if (eventTime < currentTime) {
                std::cout << "Warning: Event time " << eventTime.str()
                         << " is in the past (current: " << currentTime.str()
                         << "). Adjusting to current time." << std::endl;
                eventTime = currentTime;
            }
            mangoMsg->setCreationTime(eventTime);

            mangoMsg->setSchedulingPriority(1);
            mangoMsg->setArrival(senderModule->getId(), -1, eventTime);

            // NOW SAFE: Insert into FES from main thread
            getSimulation()->getFES()->insert(mangoMsg);

            std::cout << "Scheduled message " << eventData.messageId << " at time "
                     << eventTime.str() << std::endl;
        }
        else {
            std::cerr << "Error: Could not find sender module: " << eventData.senderId << std::endl;
        }
    }

    hasPendingData = false;
}

void MangoScheduler::sendMessage(const std::string& message) {
    // Thread-safe socket access
    std::lock_guard<std::mutex> lock(sendMutex);

    // Update maxTimeAdvance only if called from main simulation thread
    // (simTime() is only valid in main thread)
    if (message.find("RECEIVED|") == 0) {
        try {
            // Only call simTime() if we're in the main simulation thread
            // This is a bit of a hack, but necessary since simTime() isn't thread-safe
            if (getSimulation() && getSimulation()->getContextModule()) {
                maxTimeAdvance = simTime();
            }
        } catch (...) {
            // If simTime() fails (called from wrong thread), ignore the time update
            // The main thread will handle time advancement properly
        }
    }

    if (clientSocket >= 0) {
        std::string delimitedMessage = message + "\n";

        fd_set writefds;
        FD_ZERO(&writefds);
        FD_SET(clientSocket, &writefds);

        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;

        int result = select(clientSocket + 1, NULL, &writefds, NULL, &timeout);

        if (result > 0 && FD_ISSET(clientSocket, &writefds)) {
            ssize_t bytesSent = send(clientSocket, delimitedMessage.c_str(), delimitedMessage.length(), 0);

            if (bytesSent < 0) {
                std::cerr << "Error sending message: " << strerror(errno) << std::endl;
            }
            else if (bytesSent < (ssize_t)delimitedMessage.length()) {
                std::cerr << "Warning: Only sent " << bytesSent << " of " << delimitedMessage.length() << " bytes" << std::endl;
            }
        }
        else if (result < 0) {
            std::cerr << "Error in select for sending: " << strerror(errno) << std::endl;
        }
        else {
            std::cerr << "Socket not ready for writing" << std::endl;
        }
    }
    else {
        std::cerr << "Cannot send message: Socket not connected" << std::endl;
    }
}

std::string MangoScheduler::receiveMessage(bool blocking) {
    std::unique_lock<std::mutex> lock(queueMutex);

    if (blocking) {
        queueCondition.wait_for(lock, std::chrono::seconds(1), [this]{
            return !messageQueue.empty() || sigintReceived;
        });

        if (sigintReceived) {
            return "";
        }
    }

    if (messageQueue.empty()) {
        return "";
    }

    std::string message = messageQueue.front();
    messageQueue.pop();
    return message;
}

bool MangoScheduler::hasMessage() {
    std::lock_guard<std::mutex> lock(queueMutex);
    return !messageQueue.empty();
}

void MangoScheduler::startRun() {
    std::cout << "Starting MangoScheduler run, setting up socket connection..." << std::endl;

    try {
        setupServerSocket();
        sendMessage("INIT");
    }
    catch (const std::exception& e) {
        std::cerr << "Error during startRun: " << e.what() << std::endl;
        cleanup();
        throw;
    }
}

void MangoScheduler::endRun() {
    std::cout << "Ending MangoScheduler run, cleaning up..." << std::endl;

    if (clientSocket >= 0) {
        sendMessage("TERM");
    }

    cleanup();
}

cEvent* MangoScheduler::guessNextEvent() {
    return sim->getFES()->peekFirst();
}

cEvent* MangoScheduler::takeNextEvent() {
    // THREAD SAFETY FIX: Process pending data from listener thread
    if (hasPendingData) {
        processPendingData();
    }

    if (sigintReceived) {
        std::cout << "Simulation interrupted by signal, ending gracefully" << std::endl;
        throw cTerminationException(SA_INTERRUPT);
    }

    if (simTime() >= simulationDuration) {
        std::cout << "Simulation duration reached: " << simulationDuration.str() << std::endl;
        throw cTerminationException(E_ENDEDOK);
        return nullptr;
    }

    cEvent* event = sim->getFES()->peekFirst();

    if (event) {
        AdvanceTimeEvent* advanceEvent = dynamic_cast<AdvanceTimeEvent*>(event);
        if (advanceEvent) {
            sendMessage("WAITING_COMPLETE|Time advance completed");
        } else {
            simtime_t currentTime = simTime();
            simtime_t eventTime = event->getArrivalTime();

            if (eventTime > maxTimeAdvance) {
                sendMessage("WAITING");
                std::cout << "Next event at " << eventTime.str()
                         << " exceeds max advance limit of " << maxTimeAdvance.str()
                         << " from current time " << currentTime.str()
                         << ". Waiting for Python..." << std::endl;
                event = nullptr;
            }
        }
    }

    if (!event) {
        if (!terminationReceived) {
            sendMessage("WAITING");

            int waitAttempts = 0;
            const int maxWaitAttempts = 1000;

            while (waitAttempts < maxWaitAttempts && !terminationReceived && !sigintReceived) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                waitAttempts++;

                // Check for pending data and process it
                if (hasPendingData) {
                    processPendingData();
                }

                event = sim->getFES()->peekFirst();
                if (event) {
                    simtime_t currentTime = simTime();
                    simtime_t eventTime = event->getArrivalTime();

                    if (eventTime <= maxTimeAdvance) {
                        std::cout << "Found event within max advance limit after waiting" << std::endl;
                        break;
                    } else {
                        event = nullptr;
                    }
                }

                if (waitAttempts % 10 == 0) {
                    std::cout << "Still waiting for Python messages... ("
                             << waitAttempts / 10 << " seconds)" << " at time " << simTime() << std::endl;
                }
            }

            if (waitAttempts >= maxWaitAttempts && !event) {
                std::cout << "Timeout waiting for Python messages. Current max advance: "
                         << maxTimeAdvance.str() << std::endl;

                event = sim->getFES()->peekFirst();
                if (event) {
                    simtime_t timeAdvance = event->getArrivalTime() - simTime();
                    if (timeAdvance > maxTimeAdvance) {
                        std::cerr << "ERROR: No events within max advance limit and Python not responding" << std::endl;
                        throw cRuntimeError("Simulation deadlock: No events within max advance limit");
                    }
                }
            }

            if (sigintReceived) {
                std::cout << "Simulation interrupted while waiting for events" << std::endl;
                throw cTerminationException(SA_INTERRUPT);
            }

            if (!event) {
                return nullptr;
            }
        }
        else {
            std::cout << "No more events and termination received, ending simulation" << std::endl;
            throw cTerminationException(E_ENDEDOK);
        }
    }

    cEvent* tmp = sim->getFES()->removeFirst();
    ASSERT(tmp == event);

    return event;
}

void MangoScheduler::putBackEvent(cEvent* event) {
    sim->getFES()->insert(event);
}
