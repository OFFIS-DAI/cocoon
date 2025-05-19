/*
 * mango_scheduler.h
 *
 *  Created on: May 16, 2025
 *      Author: malin
 */

#ifndef MODULES_MANGO_SCHEDULER_H_
#define MODULES_MANGO_SCHEDULER_H_

#include <omnetpp.h>

using namespace omnetpp;

class MangoScheduler : public cScheduler{
public:
    MangoScheduler();

    virtual ~MangoScheduler();

    // Overridden scheduler methods
    virtual std::string str() const override;
    virtual void startRun() override;
    virtual void endRun() override;
    virtual cEvent* guessNextEvent() override;
    virtual cEvent* takeNextEvent() override;
    virtual void putBackEvent(cEvent* event) override;

};




#endif /* MODULES_MANGO_SCHEDULER_H_ */
