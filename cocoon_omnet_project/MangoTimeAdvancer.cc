/*
 * MangoTimeAdvancer.cc
 *
 *  Created on: May 23, 2025
 *      Author: malin
 */

#include "MangoTimeAdvancer.h"

Define_Module(MangoTimeAdvancer);

void MangoTimeAdvancer::handleMessage(cMessage *msg) {
    // Just discard the dummy event
    EV << "MangoTimeAdvancer received dummy time advance event: " << msg->getName() << endl;
    delete msg;
}


