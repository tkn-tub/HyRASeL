// Receiver-local timing metadata attached by TimestampedApskRadio.
//
// The start time is deliberately carried with the successfully received
// packet. This avoids matching a later packet against a separate application
// side registry when several PHY receptions overlap.

import inet.common.INETDefs;
import inet.common.TagBase;

namespace localization;

class ReceptionTimingInd extends inet::TagBase
{
    simtime_t startRxTime;
}
