#pragma once

#include <Ice/Ice.h>
#include <memory>
#include "utils.h"
#include "msp_log.h"
class RaceServer;
typedef struct InitParams {
    uint32_t maxBotCount = 10;
    std::string ice_host = getEnvString("ICE_LISTEN_HOST", "127.0.0.1");
    int ice_port = getEnvUInt("ICE_LISTEN_PORT", 30000);
    int ice_timeout = getEnvInt("ICE_TIMEOUT", 3000);
}InitParams;

class IceEnv {
public:
    IceEnv();
    int init(const InitParams & params);
    void close();
    std::shared_ptr<RaceServer> & getRaceServer() { return m_server; };
    Ice::CommunicatorPtr getIC() const { return m_ic; };
private:
    std::shared_ptr<RaceServer> m_server;
    Ice::CommunicatorPtr m_ic;
    Ice::ObjectAdapterPtr m_adapter;
};

extern std::shared_ptr<IceEnv> iceEnv;
std::shared_ptr<IceEnv> getIceEnv();

//extern void onReInitCars(tRmInfo * ReInfo);
//extern void onReBeforeOneStep(tRmInfo * ReInfo);
//extern void onReAfterOneStep(tRmInfo * ReInfo);