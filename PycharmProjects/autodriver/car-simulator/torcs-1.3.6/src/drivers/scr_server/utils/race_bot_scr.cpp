#include <stdafx.h>
#include "ice_server.h"
#include "race_server.h"
#include "race_bot_scr.h"
#include <tgf.h>
#include <track.h>
#include <car.h>
#include <raceman.h>
#include <robottools.h>
#include <robot.h>
#include "utils/sensors.h"
#include "SimpleParser.h"
#include "CarControl.h"
#include "utils/ObstacleSensors.h"


MSP_MODULE_DECLARE("bot_scr");

RaceBot_SCR::RaceBot_SCR(RaceServer &server, int idx) : RaceBotCtrl(server, idx) {
    m_botInfo->name = m_name = stdsprintf("scr_server %d", m_idx);
}

static double normRand(double avg,double std)
{
    double x1, x2, w, y1, y2;

    do {
        x1 = 2.0 * rand()/(double(RAND_MAX)) - 1.0;
        x2 = 2.0 * rand()/(double(RAND_MAX)) - 1.0;
        w = x1 * x1 + x2 * x2;
    } while ( w >= 1.0 );

    w = sqrt( (-2.0 * log( w ) ) / w );
    y1 = x1 * w;
    y2 = x2 * w;
    return y1*std + avg;
}

#define __NOISE_STD__ 0.1
#define __OPP_NOISE_STD__ 0.02
#define __FOCUS_NOISE_STD__ 0.01

void RaceBot_SCR::drive(tCarElt *car, tSituation *s) {
    if (m_total_tics == 0) {
        MSP_NOTICE("bot%02d: %s start drive", m_idx, m_name.c_str());
    }
    fillStatus(car, s);
    if (m_total_tics == 1) {
        MSP_NOTICE("bot%02d: %s, first drive call", m_idx, m_name.c_str());
    }
//        MSP_INFO("bot%02d: got status, step=%lu", m_idx, m_total_tics);
//        MSP_INFO("bot%02d: wait for driveInfo, step=%lu", m_idx, m_total_tics);
    auto bufdriveInfo = m_queueDriveInfo.pop();
    if (!bufdriveInfo) {
        MSP_INFO("bot%02d: queue pop return null, maybe closed", m_idx);
        return;
    }
//        MSP_INFO("bot%02d: got driveInfo, step=%lu", m_idx, m_total_tics);
    auto driveInfo = bufdriveInfo->driveInfo;
    car->_accelCmd = driveInfo->acceleration;
    car->_brakeCmd = driveInfo->brake;
    car->_gearCmd = driveInfo->gear;
    car->_steerCmd = driveInfo->steering;
    car->_clutchCmd = driveInfo->clutch;
    car->_focusCmd = driveInfo->focus;
    *m_driveInfo = *driveInfo;
    bufdriveInfo->recycle();
}

void RaceBot_SCR::close() {
    MSP_NOTICE("bot%02d: close", m_idx);
    RaceBotCtrl::close();
}