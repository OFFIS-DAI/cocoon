import inet.common.INETDefs;
import inet.common.TagBase;
import inet.linklayer.common.MacAddress;

namespace inet;

class MangoMsgTag extends TagBase
{
    string msgId;
    int msgSize;
}