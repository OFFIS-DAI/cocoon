/*
 * mango_scheduler.cc
 */
// Force pthread linkage
extern "C" {
    #include <pthread.h>
}
#include "mango_scheduler.h"

Register_Class(MangoScheduler);

MangoScheduler::MangoScheduler()
{
    std::cout << "MangoScheduler initialized." << std::endl;
}

MangoScheduler::~MangoScheduler()
{
    // Clean up resources if they haven't been cleaned up yet
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

    // Accept connection (blocking)
    struct sockaddr_in clientAddr;
    socklen_t clientAddrLen = sizeof(clientAddr);
    clientSocket = accept(serverSocket, (struct sockaddr *)&clientAddr, &clientAddrLen);

    if (clientSocket < 0) {
        close(serverSocket);
        throw cRuntimeError("Failed to accept client connection");
    }

    std::cout << "Python client connected" << std::endl;

    // Start listener thread for incoming messages
    running = true;
    listenerThread = new std::thread(&MangoScheduler::listenForMessages, this);
}

void MangoScheduler::listenForMessages() {
    char buffer[4096];

    while (running) {
        memset(buffer, 0, sizeof(buffer));
        ssize_t bytesRead = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);

        if (bytesRead > 0) {
            std::string message(buffer, bytesRead);

            // Add message to queue
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                messageQueue.push(message);
            }

            // Notify waiting threads
            queueCondition.notify_one();

            std::cout << "Received message from Python: " << message << std::endl;
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

        // Sleep to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void MangoScheduler::sendMessage(const std::string& message) {
    if (clientSocket >= 0) {
        send(clientSocket, message.c_str(), message.length(), 0);
    }
}

std::string MangoScheduler::receiveMessage(bool blocking) {
    std::unique_lock<std::mutex> lock(queueMutex);

    if (blocking) {
        // Wait until a message is available
        queueCondition.wait(lock, [this]{ return !messageQueue.empty(); });
    }
    else if (messageQueue.empty()) {
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
    setupServerSocket();

    // Send initialization message to Python client
    sendMessage("INIT");
}

void MangoScheduler::endRun() {
    std::cout << "Ending MangoScheduler run, cleaning up..." << std::endl;

    // Send termination message to Python client
    sendMessage("TERM");

    // Stop listener thread
    running = false;
    if (listenerThread && listenerThread->joinable()) {
        listenerThread->join();
        delete listenerThread;
        listenerThread = nullptr;
    }

    // Close sockets
    if (clientSocket >= 0) {
        close(clientSocket);
        clientSocket = -1;
    }

    if (serverSocket >= 0) {
        close(serverSocket);
        serverSocket = -1;
    }
}

cEvent* MangoScheduler::guessNextEvent() {
    return sim->getFES()->peekFirst();
}

cEvent* MangoScheduler::takeNextEvent() {
    cEvent* event = sim->getFES()->peekFirst();
    if (!event)
        throw cTerminationException(E_ENDEDOK);

    // Remove and return the event
    cEvent* tmp = sim->getFES()->removeFirst();
    ASSERT(tmp == event);

    // Notify Python about event processing
    std::string eventInfo = "EVENT:" + std::to_string(event->getArrivalTime().dbl());
    sendMessage(eventInfo);

    return event;
}

void MangoScheduler::putBackEvent(cEvent* event) {
    sim->getFES()->insert(event);
}
