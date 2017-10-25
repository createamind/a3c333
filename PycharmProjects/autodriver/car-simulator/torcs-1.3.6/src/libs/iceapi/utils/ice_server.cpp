#include "stdafx.h"
#include "ice_server.h"
#include "race_server.h"
#include "msp_log.h"
#include "utils.h"



MSP_MODULE_DECLARE("ice_server");

IceEnv::IceEnv() {
    MSP_INFO("create IceEnv");
}

int IceEnv::init(const InitParams &params) {
    std::string adpname = "RaceServer";

    Ice::PropertiesPtr props = Ice::createProperties();
    props->setProperty("Ice.MessageSizeMax", "0");
    props->setProperty("Ice.ThreadPool.Server.SizeMax", "16");
    props->setProperty("Ice.ThreadPool.Server.Size", "8");
    props->setProperty("Ice.ThreadPool.Client.Size", "8");
    props->setProperty("Ice.ThreadPool.Client.SizeMax", "16");
//    props->setProperty("Ice.Trace.ThreadPool", "1");
//    props->setProperty("Ice.Trace.Network", "1");
//    props->setProperty("Ice.Trace.Protocol", "1");

    int thread_pool_max_size = 64;

    props->setProperty(stdsprintf("%s.ThreadPool.SizeMax", adpname.c_str()) , stdsprintf("%d", thread_pool_max_size));
    props->setProperty(stdsprintf("%s.ThreadPool.Size", adpname.c_str()) , stdsprintf("%d", thread_pool_max_size/4));
    Ice::InitializationData data;
    data.properties = props;
    Ice::CommunicatorPtr ic = Ice::initialize(data);
    if(!ic)
    {
        MSP_ERROR("can not init ice communicator ptr");
        return -1;
    }
    int port = params.ice_port;
    std::string host = params.ice_host;
    int timeout = params.ice_timeout;
    assert(port > 0 && port < 65535);
    // TODO: 此处尽量不要使用0.0.0.0，因为Ice会选择所有interface，导致新的连接会连接到其他地址去，然后长时间等待而超时
    // 一定注意，被坑过两次了
    std::string endpoint = stdsprintf("tcp -h %s -p %d -t %u", host.c_str(), port, timeout, host.c_str(), port+1, timeout);
    Ice::ObjectAdapterPtr adapter;
    for(int retry = 0; retry < 10; retry++)
    {
        try
        {
            adapter = ic->createObjectAdapterWithEndpoints(adpname, endpoint);
            if(adapter)
            {
                break;
            }
        }
        catch(Ice::Exception & e)
        {
            MSP_ERROR("createObjectAdapterWithEndpoints: %s failed, exception=%s, param=%s", adpname.c_str(), e.what(), endpoint.c_str());
        }
        catch(...)
        {
            MSP_ERROR("createObjectAdapterWithEndpoints: %s failed, unknow exception, param=%s", adpname.c_str(), endpoint.c_str());
        }
        sleep(1);
    }
    if(!adapter)
    {
        MSP_ERROR("can not init ice adapter");
        return -1;
    }
    m_server = std::make_shared<RaceServer>(params.maxBotCount);
    adapter->add(m_server, Ice::stringToIdentity(adpname));
    adapter->activate();
    m_ic = ic;
    m_adapter = adapter;

    std::cout << "ice initialized: " << endpoint << std::endl;

    MSP_INFO("ice initialized %s", endpoint.c_str());
    return 0;
}

void IceEnv::close() {
    MSP_NOTICE("close adapter and ic");
    m_server->close();
    if (m_adapter) {
        MSP_INFO("deactivate adapter");
        m_adapter->deactivate();
    }
    if (m_ic) {
        m_ic->destroy();
    }
    m_server = nullptr;
}




std::shared_ptr<IceEnv> iceEnv;

std::shared_ptr<IceEnv> getIceEnv() {
    return iceEnv;
}

