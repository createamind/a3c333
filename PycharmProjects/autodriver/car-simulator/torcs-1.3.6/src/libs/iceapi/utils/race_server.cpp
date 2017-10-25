//
// Created by Robin Huang on 8/11/17.
//
#include "stdafx.h"
#include "ice_server.h"
#include "race_server.h"
MSP_MODULE_DECLARE("server");
RaceServer::RaceServer(uint32_t maxBotCount) : m_queueStatusList(4) {
    for(int i = 0;i < maxBotCount; i++) {
        m_bots.push_back(nullptr);
    }
    m_serverInitParam = std::make_shared<Race::ServerInitParam>();
}

void RaceServer::setBot(uint32_t botIdx, std::shared_ptr<RaceBot> bot) {
    assert(botIdx >= 0 && botIdx < m_bots.size());
    if (bot) {
        assert(botIdx == bot->getIdx());
        MSP_NOTICE("set bot %d: %s", bot->getIdx(), bot->getName().c_str());
    }
    else if (m_bots[botIdx]) {
        MSP_NOTICE("remove bot %d: %s", botIdx, m_bots[botIdx]->getName().c_str());
    }
    m_bots[botIdx] = bot;
}

//void RaceServer::init(::Ice::Int botIdx, const ::Race::BotInitParamPtr& params, const ::Ice::Current &) {
//    if (botIdx  < 0 || botIdx >= m_bots.size()) {
//        throw std::invalid_argument(stdsprintf("invalid botIdx %d", botIdx));
//    }
//    m_bots[botIdx]->init(params);
//}

::Race::BotInfoList RaceServer::getBots(const ::Ice::Current &) {
    while(!m_flagEnd && !m_flagCarInited) {
        usleep(1000*10);
    }
    Race::BotInfoList ret;
    for(auto a: m_bots) {
        if (a) {
            ret.push_back(a->getBotInfo());
        }
    }
    return ret;
}

extern int RESTART;
::std::shared_ptr<::Race::StepResult> RaceServer::reset(::std::shared_ptr<::Race::ResetParam> params,
                                                        const ::Ice::Current &) {
    MSP_NOTICE("client request reset: count = %lu", m_resetCount);
    if(m_resetCount > 0) {
        MSP_NOTICE("real reset, count = %d", m_resetCount);
        m_flagCarInited = false;
        m_flagReset = true;
        RESTART = 1;

        while (m_queueStatusList.size() > 0) {
            auto p = m_queueStatusList.pop();
            p->recycle();
        }
        for(auto bot: m_bots) {
            if (!bot) continue;
            auto bot_hook = dynamic_cast<RaceBotHook*>(bot.get());
            bot->close();
        }
    }

    m_stepEpisode = 0;
    m_resetCount++;
    m_semCarsInited.acquire();

    MSP_DEBUG("reset: cars inited");

    for(auto p: params->raceInit) {
        auto botIdx = p->ident;
        if (p->screenCapture && params->raceInit.size() > 1) {
            throw std::invalid_argument(stdsprintf("Screen capture mode only allow one bot per instance"));
        }
        if(botIdx < 0 || botIdx > (m_bots.size() - 1)) {
            throw std::invalid_argument(stdsprintf("invalid botIdx %d", botIdx));
        }
        auto bot = m_bots[botIdx];
        if (!bot) {
            throw std::invalid_argument(stdsprintf("botIdx %d not exist", botIdx));
        }
        bot->setRaceInitParam(p);
    }
    m_semResetSignalAvaiable.release();
    Race::StepResultPtr ret = std::make_shared<Race::StepResult>();
    auto buf = m_queueStatusList.pop();
    if (!buf) {
        if (!m_flagEnd) {
            MSP_WARNING("reset: queue status pop nullptr");
        }
        return ret;
    }
    for(auto s: buf->statusList) {
        if (s->driveInfo) {
            MSP_INFO("")
        }
        assert(!s->driveInfo);
    }
    ret->statusList = buf->statusList;
    buf->recycle();
    return ret;
}

::std::shared_ptr<::Race::StepResult> RaceServer::step(::std::shared_ptr<::Race::StepParam> params, const ::Ice::Current &) {
    for(auto di: params->driveInfos) {
        auto botIdx = di->ident;
        if (botIdx  < 0 || botIdx >= m_bots.size()) {
            throw std::invalid_argument(stdsprintf("invalid botIdx %d", botIdx));
        }
        auto bot = m_bots[botIdx];
        if (!bot) {
            throw std::invalid_argument(stdsprintf("botIdx %d not exist, pool=%s", botIdx, m_serverInitParam->name.c_str()));
        }
        auto bot_hook = dynamic_cast<RaceBotHook*>(bot.get());
        if (bot_hook) {
            throw std::invalid_argument(stdsprintf("botIdx %d is hook bot, should not has stepParam", botIdx));
        }
        auto botCtrl = dynamic_cast<RaceBotCtrl*>(bot.get());
        auto buf = botCtrl->m_queueDriveInfo.getEmpty();
        if (buf) {
            *buf->driveInfo = *di;
            buf->push2queue();
        }
    }
    Race::StepResultPtr ret = std::make_shared<Race::StepResult>();
    auto buf = m_queueStatusList.pop();
    if (!buf) {
        if (!m_flagEnd) {
            MSP_WARNING("step: queue status pop failed")
        }
        return ret;
    }
    ret->statusList = buf->statusList;
    buf->recycle();
    if(m_flagEnd) {
        ret->isAllFinished = true;
        MSP_NOTICE("send all finish flag")
    }
//    MSP_INFO("step %ld, queue status size=%d", m_stepTotal, m_queueStatusList.size());
    return ret;
}

void RaceServer::shutdown(const ::Ice::Current &) {
    MSP_NOTICE("client request shutdown");
}

void RaceServer::init(::Ice::Identity ident, ::std::shared_ptr<::Race::ServerInitParam> param, const ::Ice::Current & current) {
    MSP_INFO("client register %s", ident.name.c_str());
    m_currentClient = Ice::uncheckedCast<Race::PoolPrx>(current.con->createProxy(ident));
    m_serverInitParam = std::move(param);
//    *m_serverInitParam = *param;
//    if (!param->dataioHost.empty() && param->dataioPort > 0) {
//        m_dataserverPrx = nullptr;
//        auto iceEnv = getIceEnv();
//        std::string ep = stdsprintf("CDataIO:tcp -h %s -p %d -t 3000", param->dataioHost.c_str(), param->dataioPort);
//        for (int retry = 0; retry < 3; retry++) {
//            m_dataserverPrx = Ice::checkedCast<DataFlow::DataServerPrx>(iceEnv->getIC()->stringToProxy(ep));
//            if (m_dataserverPrx) break;
//        }
//    }
}




Race::PoolPrx RaceServer::getCurrentPool() {
    assert(0);
//    if (m_currentClient)
//        return m_currentClient;
//    if (wait) {
//        while (!m_flagEnd) {
//            m_semClientAvaiable.acquire();
//            if (m_currentClient)
//                return m_currentClient;
//        }
//    }
//    return Race::ClientPrx();
}

void RaceServer::close() {
    m_flagEnd = true;
    m_queueStatusList.close();
    for(auto bot: m_bots) {
        if (bot) {
            bot->close();
        }
    }
}

std::shared_ptr<RaceBot> RaceServer::getBot(uint32_t idx) {
    assert(idx < m_bots.size());
    return m_bots[idx];
}

void RaceServer::onReInitCars(tRmInfo *ReInfo) {
    auto nCars = ReInfo->s->_ncars;

    for (int i = 0; i < nCars; i++) {
        auto car = ReInfo->s->cars[i];
        auto robot = car->robot;
        if (robot != nullptr) {
            int index = robot->index;
            if (robot->rbDrive == RaceBotHook::cbRbDrive) {
                MSP_WARNING("robot %d already hooked, ignore", index);
                continue;
            }
            if (m_bots[index]) {
                RaceBotHook * hook = dynamic_cast<RaceBotHook*>(m_bots[index].get());
                if (hook == nullptr) {
                    MSP_INFO("ignore no hook bot %d [%s]", index, m_bots[index]->getName().c_str());
                    continue;
                }
                MSP_INFO("clear boot hook %d %s", index, m_bots[index]->getName().c_str());
                m_bots[index] = nullptr;
            }
            if (!m_bots[index]) {
                auto bot = std::make_shared<RaceBotHook>(*this, index, robot, ReInfo->track);
                m_bots[index] = bot;

            }
        }
    }
    ReInfo->_reTimeMult = m_serverInitParam->timeMult;
    m_flagCarInited = true;
    m_stepEpisode = 0;
    MSP_INFO("init %d cars, status queue size = %d, timeScale=%.2f", nCars, m_queueStatusList.size(), ReInfo->_reTimeMult);
    while(m_queueStatusList.size() > 0) {
        auto buf = m_queueStatusList.pop();
        buf->recycle();
    }
    m_semCarsInited.release();
    m_semResetSignalAvaiable.acquire();
}

void RaceServer::onReBeforeOneStep(tRmInfo *ReInfo) {
    ReInfo->_reTimeMult = m_serverInitParam->timeMult;
}

void RaceServer::onReAfterOneStep(tRmInfo *ReInfo) {
    if (m_flagReset) {
        m_flagReset = false;
//        ReInfo->_reState = RE_STATE_RACE_STOP;
    }
    if (RESTART) {
        ReInfo->_reState = RE_STATE_RACE_STOP;
        while (m_queueStatusList.size() > 0) {
            auto buf = m_queueStatusList.pop();
            buf->statusList.clear();
            buf->recycle();
        }
        m_stepEpisode = 0;
    }
//    m_semStatusAvaiable.release();
}

void RaceServer::onReBeforeDrive(tRmInfo * ReInfo) {
    if (m_flagReset) {
        MSP_NOTICE("need reset, skip drive info got");
        return;
    }
    // 此处采集最新的状态和上次的动作，放入队列
//    if (m_queueStatusList.sizeEmpty() == 0) {
//        for (int retry = 0; retry < 10 && !m_flagEnd; retry ++) {
//            if(m_queueStatusList.sizeEmpty() > 0) break;
//            usleep(1000*10);
//        }
//    }
//    if (m_flagEnd) return;
//    if (m_queueStatusList.sizeEmpty() == 0) {
//        auto p = m_queueStatusList.pop();
//        p->recycle();
//        MSP_WARNING("queue status is full[%d], pop the first one", m_queueStatusList.size());
//    }
    auto buf = m_queueStatusList.getEmpty();
    if (!buf) {
        assert(m_flagEnd);
        return;
    }
    buf->statusList.clear();
    auto s = ReInfo->s;
    for (int i = 0; i < s->_ncars; i++) {
        if ((s->cars[i]->_state & RM_CAR_STATE_NO_SIMU) == 0) {
            auto robot = s->cars[i]->robot;
            auto bot = m_bots[robot->index];
            bot->fillStatus(s->cars[i], s);
            Race::StatusPtr raceStatus = std::make_shared<Race::Status>();
            *raceStatus = *bot->getRaceStatus();
            raceStatus->seqIdx = m_stepEpisode;
            raceStatus->ident = bot->getIdx();
            auto bot_hook = dynamic_cast<RaceBotHook*>(bot.get());
            if (m_stepEpisode > 0 && bot_hook) {
                raceStatus->driveInfo = std::make_shared<Race::DriveInfo>();
                *raceStatus->driveInfo = *bot->getDriveInfo();
                raceStatus->driveInfo->seqIdx = (Ice::Long)m_stepEpisode;
            }
            buf->statusList.push_back(raceStatus);
        }
    }
//    if (m_currentClient != nullptr) {
//
//    }
//    MSP_INFO("before drive: push data to status queue, size=%d", m_queueStatusList.size());
    buf->push2queue();
}

void RaceServer::onReAfterDrive(tRmInfo *ReInfo) {
    m_stepTotal ++;
    m_stepEpisode ++;
}

void onReInitCars(tRmInfo * ReInfo) {
    MSP_INFO("onReInitCars: %p", ReInfo);
    if (iceEnv && iceEnv->getRaceServer()) {
        iceEnv->getRaceServer()->onReInitCars(ReInfo);
    }
}

void onReBeforeOneStep(tRmInfo * ReInfo) {
    if (iceEnv && iceEnv->getRaceServer()) {
        iceEnv->getRaceServer()->onReBeforeOneStep(ReInfo);
    }
}

void onReAfterOneStep(tRmInfo * ReInfo) {
    if (iceEnv && iceEnv->getRaceServer()) {
        iceEnv->getRaceServer()->onReAfterOneStep(ReInfo);
    }
}

void onReBeforeDrive(tRmInfo * ReInfo) {
    if (iceEnv && iceEnv->getRaceServer()) {
        iceEnv->getRaceServer()->onReBeforeDrive(ReInfo);
    }
}

void onReAfterDrive(tRmInfo * ReInfo) {
    if (iceEnv && iceEnv->getRaceServer()) {
        iceEnv->getRaceServer()->onReAfterDrive(ReInfo);
    }
}
//void RaceBot::init(const ::Race::BotInitParamPtr &params) {
//    *m_initParams = *params;
//}
