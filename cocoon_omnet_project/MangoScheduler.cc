
/*
 * mango_scheduler.cc
 */
extern "C" {
#include <pthread.h>
#include <signal.h>
}
#include "MangoScheduler.h"

#include <json.hpp>
using json = nlohmann::json;// Forward declare a NetworkMessage class for our simulation messages

// Signal handler flag to detect interruptions
volatile sig_atomic_t sigintReceived = 0;

// Signal handler for SIGINT
void handleSignal(int signal) {
    if (signal == SIGINT) {
        sigintReceived = 1;
        EV << "SIGINT received, preparing for graceful shutdown..." << std::endl;
    }
}

class AdvanceTimeEvent : public cMessage {
public:
    AdvanceTimeEvent() : cMessage("AdvanceTimeEvent") {}
};


Register_Class(MangoScheduler);

MangoScheduler::MangoScheduler()
{
    EV << "MangoScheduler initialized." << std::endl;
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
    EV << "App registered: " << mod->getFullPath() << std::endl;
    modules.push_back(mod);
}

cModule *MangoScheduler::getReceiverModule(std::string module_name) {
    // Get corresponding module to server name
    cModule *matchingModule = nullptr;

    EV << "Looking for module: " << module_name << std::endl;
    EV << "Available modules: " << modules.size() << std::endl;

    for (auto const& module : modules) {
        std::string moduleNameCheck = std::string(module->getParentModule()->getName());
        EV << "Checking module: " << moduleNameCheck << std::endl;
        if (moduleNameCheck.compare(module_name) == 0) {
            matchingModule = module;
            EV << "Found matching module: " << module->getFullPath() << std::endl;

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

    EV << "Server socket set up, waiting for Python client to connect on port " << PORT << std::endl;

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

            EV << "Python client connected" << std::endl;

            // Start listener thread for incoming messages
            running = true;
            listenerThread = new std::thread(&MangoScheduler::listenForMessages, this);

            // Successfully connected
            return;
        }

        // Short delay before retrying
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (sigintReceived) {
            EV << "Received interrupt while waiting for client connection" << std::endl;
            break;
        }
    }

    if (clientSocket < 0) {
        if (time(NULL) - startTime >= timeout) {
            EV << "Timeout waiting for Python client connection" << std::endl;
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
                EV << "Received message from Python: " << message << std::endl;

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
                EV << "Python client disconnected" << std::endl;
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
            EV << "Listener thread detected interruption signal" << std::endl;
            break;
        }

        // Short sleep to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    EV << "Listener thread exiting" << std::endl;
}

// Updated processMessage method to handle multiple messages in one payload
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

        if (type == "CONFIG") {
            // Handle configuration message with simulation duration
            json data = json::parse(payload);

            if (data.contains("simulation_duration")) {
                int duration = data["simulation_duration"];
                simulationDuration = SimTime(duration, SIMTIME_MS);

                std::cout << "Scheduled simulation termination at " << simulationDuration.str() << std::endl;
            }

            // Send acknowledgment
            sendMessage("CONFIG_ACK|Configuration received");
        }

        if (type == "MESSAGE") {
            // Parse the JSON payload
            json data = json::parse(payload);

            // Extract max_advance (shared for all messages in this batch)
            double max_advance = data["max_advance"];
            maxTimeAdvance = max_advance / 1000; // convert from ms to s

            EV << "Processing message batch with max advance: " << max_advance << " ms" << std::endl;

            // Extract the messages array
            json messages = data["messages"];
            std::vector<std::string> scheduled_msg_ids;
            std::vector<std::string> error_msg_ids;

            // Process each message in the batch
            for (const auto& msg : messages) {
                std::string sender = msg["sender"];
                std::string receiver = msg["receiver"];
                int64_t size_B = msg["size_B"];

                double time_send_ms = msg["time_send_ms"];
                std::string msg_id = msg["msg_id"];

                // Log the extracted information
                EV << "Processing individual message: " << std::endl;
                EV << "  Sender: " << sender << std::endl;
                EV << "  Receiver: " << receiver << std::endl;
                EV << "  Size: " << size_B << " bytes" << std::endl;
                EV << "  Send time: " << time_send_ms << " ms" << std::endl;
                EV << "  Message ID: " << msg_id << std::endl;

                // Find sender module
                cModule* senderModule = getReceiverModule(sender);
                cModule* receiverModule = getReceiverModule(receiver);
                int receiverPort = 8345; // default

                // Try to get the localPort parameter from the receiver's TCP app
                if (receiverModule) {
                    cPar& portPar = receiverModule->par("localPort");
                    if (portPar.isSet()) {
                        receiverPort = portPar.intValue();
                        EV << "Found receiver port: " << receiverPort << std::endl;
                    }
                }

                if (senderModule) {
                    EV << "sender module: " << senderModule->getFullPath() << std::endl;

                    // Create and schedule a network message
                    MangoMessage* mangoMsg = new MangoMessage("MangoMessage");
                    mangoMsg->setMessageId(msg_id);
                    mangoMsg->setSenderId(sender);
                    mangoMsg->setReceiverId(receiver);
                    mangoMsg->setMessageSize(size_B);
                    mangoMsg->setReceiverPort(receiverPort);

                    // Calculate absolute simulation time for the event
                    simtime_t eventTime = SimTime(time_send_ms, SIMTIME_MS);
                    simtime_t currentTime = simTime();
                    if (eventTime < currentTime) {
                        EV << "Warning: Event time " << eventTime.str()
                                                           << " is in the past (current: " << currentTime.str()
                                                           << "). Adjusting to current time." << std::endl;
                        eventTime = currentTime;
                    }
                    mangoMsg->setCreationTime(eventTime);

                    // Schedule the message to the sender module
                    mangoMsg->setSchedulingPriority(1); // Higher priority
                    mangoMsg->setArrival(senderModule->getId(), -1, eventTime);

                    // Insert into future event set
                    getSimulation()->getFES()->insert(mangoMsg);

                    EV << "Scheduled message " << msg_id << " at time (seconds) "
                            << eventTime.str() << std::endl;

                    scheduled_msg_ids.push_back(msg_id);
                }
                else {
                    std::cerr << "Error: Could not find sender module: " << sender << std::endl;
                    error_msg_ids.push_back(msg_id);
                }
            }

            // Send batch response back to Python
            json response;
            response["scheduled"] = scheduled_msg_ids;
            response["errors"] = error_msg_ids;

            if (error_msg_ids.empty()) {
                sendMessage("SCHEDULED|" + response.dump());
            } else {
                sendMessage("PARTIAL|" + response.dump());
            }
        }
        else if (type == "TERMINATE") {
            // Set termination flag when TERMINATE message received
            EV << "Received termination signal from Python client." << std::endl;
            terminationReceived = true;

            // Send acknowledgment back to Python
            sendMessage("TERM_ACK|Acknowledged termination request");
        }
        else if (type == "WAITING") {
            EV << "Received waiting message" << endl;

            try {
                json data = json::parse(payload);

                // Extract max_advance (shared for all messages in this batch)
                double max_advance = data["max_advance"];
                maxTimeAdvance = SimTime(max_advance, SIMTIME_MS); // convert from ms to s

                // Create a special event that will send acknowledgment when processed
                AdvanceTimeEvent *dummyEvent = new AdvanceTimeEvent();
                dummyEvent->setSchedulingPriority(1);

                cModule* advancer = getSimulation()->getModuleByPath("timeAdvancer");
                if (!advancer) {
                    EV << "Warning: timeAdvancer module not found, using system module" << std::endl;
                    advancer = getSimulation()->getSystemModule();
                }

                if (advancer) {
                    dummyEvent->setArrival(advancer->getId(), -1, maxTimeAdvance);

                    // Thread-safe FES insert with termination check
                    {
                        std::lock_guard<std::mutex> lock(fesMutex);
                        if (!simulationTerminating && getSimulation() && getSimulation()->getFES()) {
                            getSimulation()->getFES()->insert(dummyEvent);
                            // Send immediate acknowledgment that we've scheduled the time advance
                            sendMessage("WAITING_ACK|Time advance scheduled");
                        } else {
                            EV << "Simulation terminating, discarding dummy event" << std::endl;
                            delete dummyEvent; // Clean up the event
                            sendMessage("ERROR|Simulation terminating");
                        }
                    }
                } else {
                    delete dummyEvent;
                    sendMessage("ERROR|No valid module found for time advance");
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Error processing WAITING message: " << e.what() << std::endl;
                sendMessage("ERROR|" + std::string(e.what()));
            }
        }
        else {
            EV << "Unhandled message type: " << type << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error processing message: " << e.what() << std::endl;
        sendMessage("ERROR|" + std::string(e.what()));
    }
}

void MangoScheduler::sendMessage(const std::string& message) {
    //maxTimeAdvance = simTime();
    if (clientSocket >= 0) {
        // Add newline delimiter to separate messages
        std::string delimitedMessage = message + "\n";
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
            ssize_t bytesSent = send(clientSocket, delimitedMessage.c_str(), delimitedMessage.length(), 0);

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
    EV << "Starting MangoScheduler run, setting up socket connection..." << std::endl;

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
    EV << "Ending MangoScheduler run, cleaning up..." << std::endl;

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
        EV << "Simulation interrupted by signal, ending gracefully" << std::endl;
        throw cTerminationException(SA_INTERRUPT);
    }
    // Check if simulation duration has been reached
    if (simTime() >= simulationDuration) {
        EV << "Simulation duration reached: " << simulationDuration.str() << std::endl;
        throw cTerminationException(E_ENDEDOK);
        return nullptr;
    }
    // Look for the next event in the FES
    cEvent* event = sim->getFES()->peekFirst();

    // Check if the next event would exceed max time advance
    if (event) {
        // Check if this is our dummy time advance event
        AdvanceTimeEvent* advanceEvent = dynamic_cast<AdvanceTimeEvent*>(event);
        if (advanceEvent) {
            // Send completion message to Python
            sendMessage("WAITING_COMPLETE|Time advance completed");
        } else {
            simtime_t currentTime = simTime();
            simtime_t eventTime = event->getArrivalTime();

            if (eventTime > maxTimeAdvance) {
                sendMessage("WAITING");
                // Next event is beyond max advance - we need to wait for Python
                EV << "Next event at " << eventTime.str()
                                                                                  << " exceeds max advance limit of " << maxTimeAdvance.str()
                                                                                  << " from current time " << currentTime.str()
                                                                                  << ". Waiting for Python..." << std::endl;

                // Don't process this event yet - wait for Python messages
                event = nullptr;
            }
        }
    }

    if (!event) {
        // No events in FES within allowed time range - wait for Python messages
        if (!terminationReceived) {
            sendMessage("WAITING");

            // Wait for messages with periodic checks
            int waitAttempts = 0;
            const int maxWaitAttempts = 1000; // 100 seconds total (100 * 100ms)

            while (waitAttempts < maxWaitAttempts && !terminationReceived && !sigintReceived) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                waitAttempts++;

                // Check if we received new messages that might have updated maxTimeAdvance
                // or scheduled new events within the allowed range
                event = sim->getFES()->peekFirst();
                if (event) {
                    simtime_t currentTime = simTime();
                    simtime_t eventTime = event->getArrivalTime();
                    simtime_t timeAdvance = eventTime - currentTime;

                    if (timeAdvance <= maxTimeAdvance) {
                        // Found an event within allowed range
                        EV << "Found event within max advance limit after waiting" << std::endl;
                        break;
                    } else {
                        // Still beyond limit
                        event = nullptr;
                    }
                }

                // Periodically log waiting status
                if (waitAttempts % 10 == 0) {
                    EV << "Still waiting for Python messages... ("
                            << waitAttempts / 10 << " seconds)" << " at time " << simTime()  << std::endl;
                }
            }

            // If we've waited too long without receiving messages
            if (waitAttempts >= maxWaitAttempts && !event) {
                EV << "Timeout waiting for Python messages. Current max advance: "
                        << maxTimeAdvance.str() << std::endl;

                // Check one more time if there's an event we can process
                event = sim->getFES()->peekFirst();
                if (event) {
                    simtime_t timeAdvance = event->getArrivalTime() - simTime();
                    if (timeAdvance > maxTimeAdvance) {
                        std::cerr << "ERROR: No events within max advance limit and Python not responding" << std::endl;
                        throw cRuntimeError("Simulation deadlock: No events within max advance limit");
                    }
                }
            }

            // If interrupted during wait, end simulation
            if (sigintReceived) {
                EV << "Simulation interrupted while waiting for events" << std::endl;
                throw cTerminationException(SA_INTERRUPT);
            }

            // If still no events within range, return nullptr to continue polling
            if (!event) {
                return nullptr;
            }
        }
        else {
            // We've received termination signal and there are no more events
            EV << "No more events and termination received, ending simulation" << std::endl;
            throw cTerminationException(E_ENDEDOK);
        }
    }

    // We have an event within the allowed time range - remove it from FES and return it
    cEvent* tmp = sim->getFES()->removeFirst();
    ASSERT(tmp == event);

    return event;
}
void MangoScheduler::putBackEvent(cEvent* event) {
    sim->getFES()->insert(event);
}
