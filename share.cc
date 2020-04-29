#include <time.h>
#include <mutex>
#include <stdio.h>
#include <iostream>
#include "share.h"

string logTimestamp = GetTimeCurrnet();
std::mutex m_mutex;

string GetTimeCurrnet()
{
    char tmp[20];
    time_t timestamp;
    time(&timestamp);
    strftime(tmp, sizeof(tmp), "%Y-%m-%d_%H-%M-%S", localtime(&timestamp));
    return tmp;
}

void log(LOGLEVEL level, string content)
{
    string strLoglevel;
    if (level == info)
    {
        strLoglevel = ",[INFO]: ";
    }
    else if (level == warnning)
    {
        strLoglevel = ",[WARRNING]: ";
    }
    else
    {
        strLoglevel = ",[ERROR]: ";
    }

    FILE *logfile;
    std::unique_lock<std::mutex> locker(m_mutex);
    logfile = fopen(("./results/" + logTimestamp + ".log").c_str(), "at+");
    if (logfile != NULL)
    {
        char timeBuffer[20];
        time_t current;
        time(&current);
        strftime(timeBuffer, sizeof(timeBuffer), "%Y-%m-%d %H:%M:%S", localtime(&current));
        string singleLog = string(timeBuffer) + strLoglevel + content + "\r\n";
        fwrite(singleLog.c_str(), singleLog.size(), 1, logfile);
        fclose(logfile);
        locker.unlock();
    }
    else
    {
        //debug
        std::cout << "Log wrong." << std::endl;
    }
}