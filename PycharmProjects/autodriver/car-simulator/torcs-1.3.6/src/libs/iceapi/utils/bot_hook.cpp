//
// Created by Robin Huang on 8/9/17.
//
#include "stdafx.h"
#include "ice_server.h"
#include "race_server.h"
#include "robot.h"
#include "robottools.h"
MSP_MODULE_DECLARE("bot_hook");

RaceBotHook::RaceBotHook(RaceServer &server, int idx, struct RobotItf * robot, tTrack * track) : RaceBot(server, idx) {
    m_botInfo->name = m_name = robot->name;
    assert(robot->rbDrive != RaceBotHook::cbRbDrive);
    m_cbDrive = robot->rbDrive;
    robot->rbDrive = RaceBotHook::cbRbDrive;
    m_cbNewTrack = robot->rbNewTrack;
    robot->rbNewTrack = RaceBotHook::cbRbNewTrack;
    m_cbNewRace = robot->rbNewRace;
    robot->rbNewRace = RaceBotHook::cbRbNewRace;
    m_cbEndRace = robot->rbEndRace;
    robot->rbEndRace = RaceBotHook::cbRbEndRace;
    m_cbShutdown = robot->rbShutdown;
    robot->rbShutdown = RaceBotHook::cbRbShutdown;
    m_cbPitCmd = robot->rbPitCmd;
    robot->rbPitCmd = RaceBotHook::cbRbPitCmd;

    m_tTrack = track;
    MSP_INFO("bot%02d %s replace hook func", idx, m_name.c_str());
}

void RaceBotHook::cbRbNewTrack(int index, tTrack *track, void *carHandle, void **myCarSettings, tSituation *s) {
    std::shared_ptr<RaceBot> bot = iceEnv->getRaceServer()->getBot(index);
    RaceBotHook * hook = dynamic_cast<RaceBotHook*>(bot.get());
    hook->newTrack(track, carHandle, myCarSettings, s);
}

void RaceBotHook::newTrack(tTrack *track, void *car, void **carParam, tSituation *s) {
    // will never called
    RaceBot::newTrack(track, car, carParam, s);
    MSP_INFO("bot%02d: newTrack");
    m_cbNewTrack(m_idx, track, car, carParam, s);
}

void RaceBotHook::cbRbNewRace(int index, tCarElt *car, tSituation *s) {
    std::shared_ptr<RaceBot> bot = iceEnv->getRaceServer()->getBot(index);
    RaceBotHook * hook = dynamic_cast<RaceBotHook*>(bot.get());
    hook->newRace(car, s);
}

void RaceBotHook::newRace(tCarElt *car, tSituation *s) {
    RaceBot::newRace(car, s);
    m_cbNewRace(m_idx, car, s);
}

void RaceBotHook::cbRbEndRace(int index, tCarElt *car, tSituation *s) {
    std::shared_ptr<RaceBot> bot = iceEnv->getRaceServer()->getBot(index);
    RaceBotHook * hook = dynamic_cast<RaceBotHook*>(bot.get());
    hook->endRace(car, s);
}

void RaceBotHook::endRace(tCarElt *car, tSituation *s) {
    MSP_INFO("bot%02d: endRace", m_idx);
    m_cbEndRace(m_idx, car, s);
}

void RaceBotHook::cbRbDrive(int index, tCarElt *car, tSituation *s) {
    std::shared_ptr<RaceBot> bot = iceEnv->getRaceServer()->getBot(index);
    RaceBotHook * hook = dynamic_cast<RaceBotHook*>(bot.get());
    hook->drive(car, s);
}

extern int RESTART;

void RaceBotHook::fillStatus(const tCarElt *car, const tSituation *s) {
    auto vsaved = car->_focusCD;
    RaceBot::fillStatus(car, s);
    ((tCarElt*)car)->_focusCD = vsaved;
}

void RaceBotHook::drive(tCarElt *car, tSituation *s) {
    if (m_total_tics == 0) {
        MSP_NOTICE("bot%02d: %s start drive", m_idx, m_name.c_str());
    }
//    tdble yaw_rel = RtTrackSideTgAngleL(&(car->_trkPos)) - car->_yaw; // relative yaw
//    MSP_DEBUG("Bot %d:\tSpeed : %0.2f \tYaw : %0.2f\tRPM : %0.2f\tGear : %d\tFuel : %0.2f", m_idx,
//              3.6*car->_speed_x,yaw_rel, car->_enginerpm, car->_gear, car->_fuel);


    m_cbDrive(m_idx, car, s);

    auto driveInfo = m_driveInfo;
    driveInfo->acceleration = car->_accelCmd;
    driveInfo->brake= car->_brakeCmd;
    driveInfo->gear = car->_gearCmd;
    driveInfo->steering = car->_steerCmd;
    driveInfo->clutch = car->_clutchCmd;
    driveInfo->focus = car->_focusCmd;
    driveInfo->seqIdx++;
    if (m_total_tics == 1) {
        MSP_NOTICE("bot%02d: first drive call", m_idx, m_name.c_str());
    }
}

void RaceBotHook::cbRbShutdown(int index) {
    std::shared_ptr<RaceBot> bot = iceEnv->getRaceServer()->getBot(index);
    RaceBotHook * hook = dynamic_cast<RaceBotHook*>(bot.get());
    hook->shutdown();
}

void RaceBotHook::shutdown() {
    MSP_INFO("bot%02d: shutdown", m_idx);
    m_cbShutdown(m_idx);
}

int RaceBotHook::cbRbPitCmd(int index, tCarElt *car, tSituation *s) {
    std::shared_ptr<RaceBot> bot = iceEnv->getRaceServer()->getBot(index);
    RaceBotHook * hook = dynamic_cast<RaceBotHook*>(bot.get());
    return hook->pitCmd(car, s);
}

int RaceBotHook::pitCmd(tCarElt *car, tSituation *s) {
    return m_cbPitCmd(m_idx, car, s);
}
