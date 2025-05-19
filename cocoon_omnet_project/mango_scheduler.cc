/*
 * mango_scheduler.cc
 *
 *  Created on: May 16, 2025
 *      Author: malin
 */


#include "mango_scheduler.h"

Register_Class(MangoScheduler);

MangoScheduler::MangoScheduler()
{
    std::cout << "scheduler initialized. " << std::endl;
}


MangoScheduler::~MangoScheduler()
{
}

std::string MangoScheduler::str() const
{
    return "MangoScheduler (Python integration)";
}

void MangoScheduler::startRun() {

}

void MangoScheduler::endRun() {

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
    std::cout << "take next event" << std::endl;
    return event;
}

void MangoScheduler::putBackEvent(cEvent* event) {

}
