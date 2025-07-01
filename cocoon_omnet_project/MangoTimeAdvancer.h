/*
 * MangoTimeAdvancer.h
 *
 *  Created on: May 23, 2025
 *      Author: malin
 */

#ifndef MANGOTIMEADVANCER_H_
#define MANGOTIMEADVANCER_H_

#include <omnetpp.h>
using namespace omnetpp;

class MangoTimeAdvancer : public cSimpleModule {
  protected:
    virtual void handleMessage(cMessage *msg) override;
};


#endif /* MANGOTIMEADVANCER_H_ */
