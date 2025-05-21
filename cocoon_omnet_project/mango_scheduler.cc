/*
 * mango_scheduler.cc
 */
extern "C" {
#include <pthread.h>
#include <signal.h>
}
#include "mango_scheduler.h"

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

// Forward declare a NetworkMessage class for our simulation messages
class MangoMessage : public cMessage {
private:
    std::string messageId;
    std::string senderId;
    std::string receiverId;
    int64_t messageSize;
    simtime_t creationTime;

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
    // Clean up resources if they haven't been cleaned up yet
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
    // Get corresponding module to server name
    cModule *matchingModule = nullptr;

    std::cout << "Looking for module: " << module_name << std::endl;
    std::cout << "Available modules: " << modules.size() << std::endl;

    for (auto const& module : modules) {
        std::string moduleNameCheck = std::string(module->getParentModule()->getName());
        std::cout << "Checking module: " << moduleNameCheck << std::endl;
        if (moduleNameCheck.compare(module_name) == 0) {
            matchingModule = module;
            std::cout << "Found matching module: " << module->getFullPath() << std::endl;
            break;
        }
    }
    return (matchingModule);
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
        // Check for incoming connections (non-blocking)
        struct sockaddr_in clientAddr;
        socklen_t clientAddrLen = sizeof(clientAddr);
        clientSocket = accept(serverSocket, (struct sockaddr *)&clientAddr, &clientAddrLen);

        if (clientSocket >= 0) {
            // Set client socket to non-blocking
            flags = fcntl(clientSocket, F_GETFL, 0);
            fcntl(clientSocket, F_SETFL, flags | O_NONBLOCK);

            std::cout << "Python client connected" << std::endl;

            // Start listener thread for incoming messages
            running = true;
            listenerThread = new std::thread(&MangoScheduler::listenForMessages, this);

            // Successfully connected
            return;
        }

        // Short delay before retrying
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

    // Set a smaller receive timeout to be more responsive to shutdown
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 100000; // 100ms
    setsockopt(clientSocket, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);

    while (running && !sigintReceived) {
        memset(buffer, 0, sizeof(buffer));

        // Check if socket is ready to read
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

                // Process the message immediately
                processMessage(message);

                // Add message to queue for other components that might need it
                {
                    std::lock_guard<std::mutex> lock(queueMutex);
                    messageQueue.push(message);
                }

                // Notify waiting threads
                queueCondition.notify_one();
            }
            else if (bytesRead == 0) {
                std::cout << "Python client disconnected" << std::endl;
                break;
            }
            else {
                // Error occurred, check if we should continue
                if (errno != EAGAIN && errno != EWOULDBLOCK) {
                    std::cerr << "Error reading from socket: " << strerror(errno) << std::endl;
                    break;
                }
            }
        }

        // Check for interruption
        if (sigintReceived) {
            std::cout << "Listener thread detected interruption signal" << std::endl;
            break;
        }

        // Short sleep to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "Listener thread exiting" << std::endl;
}

// Updated processMessage method for MangoScheduler to ensure modules are up

void MangoScheduler::processMessage(const std::string& message) {
    try {
        // Split the message into type and payload
        size_t delimiterPos = message.find('|');
        if (delimiterPos == std::string::npos) {
            std::cerr << "Invalid message format: " << message << std::endl;
            return;
        }

        std::string type = message.substr(0, delimiterPos);
        std::string payload = message.substr(delimiterPos + 1);

        if (type == "MESSAGE") {
            // Parse the JSON payload
            json data = json::parse(payload);

            // Extract the fields
            std::string sender = data["sender"];
            std::string receiver = data["receiver"];
            int64_t size_B = data["size_B"];
            double time_send_ms = data["time_send_ms"];
            std::string msg_id = data["msg_id"];
            double max_advance = data["max_advance"];

            // Update max time advance
            maxTimeAdvance = max_advance;

            // Log the extracted information
            std::cout << "Processing message: " << std::endl;
            std::cout << "  Sender: " << sender << std::endl;
            std::cout << "  Receiver: " << receiver << std::endl;
            std::cout << "  Size: " << size_B << " bytes" << std::endl;
            std::cout << "  Send time: " << time_send_ms << " ms" << std::endl;
            std::cout << "  Message ID: " << msg_id << std::endl;
            std::cout << "  Max advance: " << max_advance << std::endl;

            // Find modules
            cModule* senderModule = getReceiverModule(sender);

            if (senderModule) {
                std::cout << "sender module: " << senderModule->getFullPath() << std::endl;

                // Check if the module is up by checking for NodeStatus
                cModule* senderNode = senderModule->getParentModule();
                cModule* statusModule = senderNode->getSubmodule("status");
                bool isUp = true;

                if (statusModule) {
                    std::string statusString = statusModule->par("state").stringValue();
                    isUp = (statusString == "UP");
                    std::cout << "Module status: " << statusString << std::endl;
                }

                if (!isUp) {
                    std::cerr << "Warning: Sender module is not UP, message may be dropped." << std::endl;
                    // Still try to send the message - the module will handle it appropriately
                }

                // Create and schedule a network message
                MangoMessage* mangoMsg = new MangoMessage("MangoMessage");
                mangoMsg->setMessageId(msg_id);
                mangoMsg->setSenderId(sender);
                mangoMsg->setReceiverId(receiver);
                mangoMsg->setMessageSize(size_B);

                // Calculate absolute simulation time for the event
                simtime_t eventTime = simTime() + SimTime(time_send_ms, SIMTIME_MS);
                mangoMsg->setCreationTime(eventTime);

                // Schedule the message to the sender module
                mangoMsg->setSchedulingPriority(1); // Higher priority
                mangoMsg->setArrival(senderModule->getId(), -1, eventTime);

                // Insert into future event set
                getSimulation()->getFES()->insert(mangoMsg);

                std::cout << "Scheduled message " << msg_id << " at time (seconds) "
                          << eventTime.str() << std::endl;

                // Send a confirmation back to Python
                sendMessage("SCHEDULED|" + msg_id);
            }
            else {
                std::cerr << "Error: Could not find sender module: " << sender << std::endl;

                // Send an error message back to Python
                sendMessage("ERROR|Could not find sender module: " + sender);
            }
        }
        else if (type == "TERMINATE") {
            // Set termination flag when TERMINATE message received
            std::cout << "Received termination signal from Python client." << std::endl;
            terminationReceived = true;

            // Send acknowledgment back to Python
            sendMessage("TERM_ACK|Acknowledged termination request");
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

void MangoScheduler::sendMessage(const std::string& message) {
    if (clientSocket >= 0) {
        // Use select to check if the socket is writable
        fd_set writefds;
        FD_ZERO(&writefds);
        FD_SET(clientSocket, &writefds);

        struct timeval timeout;
        timeout.tv_sec = 1;  // 1 second timeout
        timeout.tv_usec = 0;

        int result = select(clientSocket + 1, NULL, &writefds, NULL, &timeout);

        if (result > 0 && FD_ISSET(clientSocket, &writefds)) {
            // Socket is ready for writing
            ssize_t bytesSent = send(clientSocket, message.c_str(), message.length(), 0);

            if (bytesSent < 0) {
                std::cerr << "Error sending message: " << strerror(errno) << std::endl;
            }
            else if (bytesSent < (ssize_t)message.length()) {
                std::cerr << "Warning: Only sent " << bytesSent << " of " << message.length() << " bytes" << std::endl;
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
        // Wait until a message is available or interrupted
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

        // Send initialization message to Python client
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

    // Send termination message to Python client
    if (clientSocket >= 0) {
        sendMessage("TERM");
    }

    // Clean up resources
    cleanup();
}

cEvent* MangoScheduler::guessNextEvent() {
    return sim->getFES()->peekFirst();
}

cEvent* MangoScheduler::takeNextEvent() {
    // First check if we've been interrupted
    if (sigintReceived) {
        std::cout << "Simulation interrupted by signal, ending gracefully" << std::endl;
        throw cTerminationException(SA_INTERRUPT);
    }

    // Look for the next event in the FES
    cEvent* event = sim->getFES()->peekFirst();

    if (!event) {
        // No events in FES - but we should wait for Python messages if not terminated
        if (!terminationReceived) {
            std::cout << "No more events but waiting for Python messages..." << std::endl;

            // Wait for a short time to allow messages to arrive
            // This will also check for interruptions periodically
            for (int i = 0; i < 10 && !terminationReceived && !sigintReceived; i++) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                // Check for new events after waiting
                event = sim->getFES()->peekFirst();
                if (event) {
                    break; // Found an event, process it
                }
            }

            // If interrupted during wait, end simulation
            if (sigintReceived) {
                std::cout << "Simulation interrupted while waiting for events" << std::endl;
                throw cTerminationException(SA_INTERRUPT);
            }

            // If still no events, return nullptr to continue polling
            if (!event) {
                return nullptr;
            }
        }
        else {
            // We've received termination signal and there are no more events
            std::cout << "No more events and termination received, ending simulation" << std::endl;
            throw cTerminationException(E_ENDEDOK);
        }
    }

    // We have an event - remove it from FES and return it
    cEvent* tmp = sim->getFES()->removeFirst();
    ASSERT(tmp == event);

    return event;
}

void MangoScheduler::putBackEvent(cEvent* event) {
    sim->getFES()->insert(event);
}
