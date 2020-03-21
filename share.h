// Author: sumingrui

#ifndef SHARE_H_
#define SHARE_H_

#include <string>

using std::string;

enum LOGLEVEL
{
    info,
    warnning,
    error
};

extern string logTimestamp;
string GetTimeCurrnet();
void log(LOGLEVEL level, string content);

#endif //SHARE_H_