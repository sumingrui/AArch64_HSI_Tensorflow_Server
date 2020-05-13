#ifndef MYSTUBSERVER_H_
#define MYSTUBSERVER_H_

#include "abstractstubserver.h"
#include "tf.h"
#include <unistd.h>
#include <stdio.h>
#include <string>

using namespace jsonrpc;
using namespace std;

class MyStubServer : public AbstractStubServer
{
public:
    MyStubServer(AbstractServerConnector &connector);

    virtual string sayHello();
    virtual int exeAlgorithm(const string &algorithm, double computeRatio, const string &rawfile);
    virtual int stopListening();
};

MyStubServer::MyStubServer(AbstractServerConnector &connector) : AbstractStubServer(connector)
{
}

string MyStubServer::sayHello()
{
    return "Server has Connected";
}

int MyStubServer::exeAlgorithm(const string &algorithm, double computeRatio, const string &rawfile)
{
    string taskfile = "/repos/tf_server/AArch64_HSI_Tensorflow_Server/dataset/" + rawfile;
    if (!access(taskfile.c_str(), F_OK | R_OK))
    {
        //tf_2dcnn
        if (algorithm.compare("tf_2dcnn") == 0)
        {
            TF_2dcnn("/repos/tf_server/AArch64_HSI_Tensorflow_Server/dataset/", rawfile.substr(0, rawfile.length() - 4).c_str(), computeRatio);
            return 1;
        }
    }
}

int MyStubServer::stopListening()
{
    //stop
    return 1;
}

#endif //MYSTUBSERVER_H_
