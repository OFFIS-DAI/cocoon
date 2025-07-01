cplusplus {{
#include "TimerType_m.h"
}}

enum TimerType;

packet Timer {
    int timerType @enum(TimerType);
    int receiverPort;
}