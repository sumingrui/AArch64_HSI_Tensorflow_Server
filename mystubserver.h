#ifndef MYSTUBSERVER_H_
#define MYSTUBSERVER_H_

#define USE_CXX11_ABI 1

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
    virtual int dataTransfer(const string &data);
    virtual double requestRatio();
    virtual string requestResult(const string &data);
    virtual int stopListening();
};

MyStubServer::MyStubServer(AbstractServerConnector &connector) : AbstractStubServer(connector)
{
}

string MyStubServer::sayHello()
{
    return "Server has Connected";
}

int MyStubServer::dataTransfer(const string &data)
{
    //检测data文件是否存在
    if (!access(data.c_str(),F_OK | R_OK))
    {
        //tf_cnn
        const char* sendImgPath;
        TF_2dcnn("/repos/tf_server/AArch64_HSI_Tensorflow_Server/dataset/","newrawSinglefile20190711140909", sendImgPath);
        //log(info, "Result Image path: " + sendImgPath);
        return 1;
    }
}

double MyStubServer::requestRatio()
{
    //返回识别比率
    return 0.8;
}

string MyStubServer::requestResult(const string &data)
{
    //scp发送图片，返回图片名称
    return "image.jpg";
}

int MyStubServer::stopListening()
{
    //stop
    return 1;
}

#endif //MYSTUBSERVER_H_
