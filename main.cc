#include <jsonrpccpp/server/connectors/httpserver.h>
#include "mystubserver.h"
int main()
{
    HttpServer httpserver(8383);
    MyStubServer s(httpserver);
    s.StartListening();
    getchar();
    s.StopListening();
    return 0;
}
