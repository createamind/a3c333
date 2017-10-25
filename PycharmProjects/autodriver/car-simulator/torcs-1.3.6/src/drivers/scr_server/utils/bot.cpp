#include "stdafx.h"
#include "ice_server.h"
#include "race_server.h"
#include "sensors.h"
#include "ObstacleSensors.h"

MSP_MODULE_DECLARE("bot");
RaceBot::RaceBot(RaceServer &server, int idx)  : m_server(server), m_idx(idx) {
    m_name = stdsprintf("racebot %d", m_idx);
    m_initParams = std::make_shared<Race::BotInitParam>();
    m_raceInitParams = std::make_shared<Race::BotRaceInitParam>();
    m_botInfo = std::make_shared<Race::BotInfo>();
    m_botInfo->name = m_name;
    m_botInfo->idx = m_idx;

    m_raceStatus = std::make_shared<Race::Status>();
    m_raceStatus->ident = m_idx;
    m_driveInfo = std::make_shared<Race::DriveInfo>();
    m_driveInfo->ident = m_idx;
}

RaceBot::~RaceBot() {

}

void RaceBot::close() {
}

void RaceBot::newTrack(tTrack *_track, void *car, void **carParam, tSituation *s) {
    assert(_track);
    m_tTrack = _track;
    auto * track = (tTrack*)_track;
    *carParam = NULL;
    auto trackName = strrchr(track->filename, '/') + 1;
    MSP_INFO("bot%02d: %s init track %s", m_idx, m_name.c_str(), trackName);
}

void RaceBot::newRace(tCarElt *car, tSituation *s) {
    MSP_INFO("bot%02d: %s newRace", m_idx, m_name.c_str());
    if (strcmp(getVersion(),"2009")==0)
    {
        __SENSORS_RANGE__ = 100;
        MSP_INFO("*****2009*****");
    }
    else if (strcmp(getVersion(),"2010")==0 || strcmp(getVersion(),"2011")==0 || strcmp(getVersion(),"2012")==0 || strcmp(getVersion(),"2013")==0)
        __SENSORS_RANGE__ = 200;
    else
    {
        MSP_ERROR("%s is not a recognized version",getVersion());
        exit(0);
    }

//    MSP_INFO("[%d]: wait for client connect", m_idx);
//    m_semInit.acquire();
//    MSP_INFO("[%d]: client connected");

    srand(time(NULL));
    {
        if (m_raceInitParams->angles.size() > 0) {
            std::stringstream strm;
            assert(m_raceInitParams->angles.size() == 19);
            for (int i = 0; i < 19; i++) {
                m_trackSensAngle[i] = m_raceInitParams->angles[i];
                strm << stdsprintf("%.4f,", m_trackSensAngle[i]);
//            MSP_DEBUG("trackSensAngle[%d] %.4f", i, m_trackSensAngle[i]);
            }
            MSP_INFO("init TrackSensorAngle: %s", strm.str().c_str());
        }
        else {
            std::stringstream strm;
            for (int i = 0; i < 19; ++i) {
                m_trackSensAngle[i] = -90.f + 10.0f * i;
                strm << stdsprintf("%.4f,", m_trackSensAngle[i]);
            }
            MSP_INFO("init default TrackSensorAngle: %s", strm.str().c_str());
        }
        m_trackSensor = std::make_shared<Sensors>(car, 19);
        for (int i = 0; i < 19; ++i) {
            m_trackSensor->setSensor(i,m_trackSensAngle[i],__SENSORS_RANGE__);
        }
    }
    {
        std::stringstream strm;
        m_focusSensor = std::make_shared<Sensors>(car, 5);
        for (int i = 0; i < 5; ++i) {//ML
            strm << stdsprintf("%.4f,", m_trackSensAngle[i]);
            m_focusSensor->setSensor(i, (car->_focusCmd) + i - 2.0, 200);//ML
        }
        MSP_INFO("init focus sensor %s", strm.str().c_str());
    }
    m_oppSensor = std::make_shared<ObstacleSensors>(36, (tTrack*)m_tTrack, car, s, (int) __SENSORS_RANGE__);
    m_prevDist = -1.f;
    m_total_tics = 0;
    m_distRaced = 0.0f;
}

void RaceBot::endRace(tCarElt *car, tSituation *s) {
    m_trackSensor = nullptr;
    m_focusSensor = nullptr;
    m_oppSensor = nullptr;
}
#include "robot.h"
#include "robottools.h"
void RaceBot::fillStatus(const tCarElt *car, const tSituation *s) {

    tdble angle = RtTrackSideTgAngleL(&(((tCarElt*)car)->_trkPos)) - car->_yaw; // relative yaw
    MSP_DEBUG("[%d]: Bot :\tSpeed : %0.2f \tYaw : %0.2f\tRPM : %0.2f\tGear : %d\tFuel : %0.2f", m_idx,
              3.6*car->_speed_x,angle, car->_enginerpm, car->_gear, car->_fuel);

    m_total_tics ++;

    // computing distance to middle
    float dist_to_middle = 2*car->_trkPos.toMiddle/(car->_trkPos.seg->width);
    // computing the car angle wrt the track axis
//    float angle =  RtTrackSideTgAngleL(&(car->_trkPos)) - car->_yaw;
    bool internalError = false;
    NORM_PI_PI(angle); // normalize the angle between -PI and + PI
    if (std::isnan(angle)) {
        MSP_ERROR("bot%02d: angle is nan, trackPos = %.2f, _yaw = %.2f", m_idx, car->_trkPos, car->_yaw);
        internalError = true;
    }
    //Update focus sensors' angle
    for (int i = 0; i < 5; ++i) {
        m_focusSensor->setSensor(i,(car->_focusCmd)+i-2.0,200);
    }

    // update the value of track sensors only as long as the car is inside the track
    float trackSensorOut[19];
    float focusSensorOut[5];//ML
    if (dist_to_middle<=1.0 && dist_to_middle >=-1.0 )
    {
        m_trackSensor->sensors_update();
        for (int i = 0; i < 19; ++i)
        {
            trackSensorOut[i] = m_trackSensor->getSensorOut(i);
//            if (getNoisy())
//                trackSensorOut[i] *= normRand(1,__NOISE_STD__);
        }
        m_focusSensor->sensors_update();//ML
//        if ((car->_focusCD <= car->_curLapTime + car->_curTime)//ML Only send focus sensor reading if cooldown is over
//            && (car->_focusCmd != 360))//ML Only send focus reading if requested by client
        {//ML
            for (int i = 0; i < 5; ++i)
            {
                focusSensorOut[i] = m_focusSensor->getSensorOut(i);
//                if (getNoisy())
//                    focusSensorOut[i] *= normRand(1,__FOCUS_NOISE_STD__);
            }
//            ((tCarElt*)car)->_focusCD = car->_curLapTime + car->_curTime + 1.0;//ML Add cooldown [seconds]
        }//ML
//        else//ML
//        {//ML
//            for (int i = 0; i < 5; ++i)//ML
//                focusSensorOut[i] = -1;//ML During cooldown send invalid focus reading
//        }//ML
    }
    else
    {
        for (int i = 0; i < 19; ++i)
        {
            trackSensorOut[i] = -1;
        }
        for (int i = 0; i < 5; ++i)
        {
            focusSensorOut[i] = -1;
        }
    }

    // update the value of opponent sensors
    float oppSensorOut[36];
    m_oppSensor->sensors_update(s);
    for (int i = 0; i < 36; ++i)
    {
        oppSensorOut[i] = m_oppSensor->getObstacleSensorOut(i);
//        if (getNoisy())
//            oppSensorOut[i] *= normRand(1,__OPP_NOISE_STD__);
    }

    float wheelSpinVel[4];
    for (int i=0; i<4; ++i) {
        wheelSpinVel[i] = car->_wheelSpinVel(i);
    }

    if (m_prevDist<0) {
        m_prevDist = car->race.distFromStartLine;
    }
    float curDistRaced = car->race.distFromStartLine - m_prevDist;
    m_prevDist = car->race.distFromStartLine;
    auto curTrack = (tTrack*)m_tTrack;
    if (curDistRaced>100) {
        curDistRaced -= curTrack->length;
    }
    if (curDistRaced<-100) {
        curDistRaced += curTrack->length;
    }

    m_distRaced += curDistRaced;
    auto raceStatus = m_raceStatus;
    raceStatus->angle = angle;
    raceStatus->curLapTime = float(car->_curLapTime);
    raceStatus->damage = getDamageLimit() ? car->_dammage : car->_fakeDammage;
    raceStatus->distFromStart = car->race.distFromStartLine;
    raceStatus->distRaced = m_distRaced;
    raceStatus->fuel = car->_fuel;
    raceStatus->gear = car->_gear;
    raceStatus->lastLapTime = float(car->_lastLapTime);
    raceStatus->opponents.resize(36);
    for(int i = 0; i < 36; i++)
        raceStatus->opponents[i] = oppSensorOut[i];
    raceStatus->racePos = car->race.pos;
    raceStatus->rpm = car->_enginerpm*10;
    raceStatus->speedX = float(car->_speed_x  * 3.6);
    raceStatus->speedY = float(car->_speed_y  * 3.6);
    raceStatus->speedZ = float(car->_speed_z  * 3.6);
    raceStatus->track.resize(19);
    for(int i = 0; i < 19; i++)
        raceStatus->track[i] = trackSensorOut[i];

    raceStatus->trackPos = dist_to_middle;
    raceStatus->wheelSpinVel.resize(4);
    for(int i = 0; i < 4; i++)
        raceStatus->wheelSpinVel[i] = wheelSpinVel[i];
    raceStatus->z = car->_pos_Z  - RtTrackHeightL(&(((tCarElt*)car)->_trkPos));
    raceStatus->focus.resize(5);
    for(int i = 0; i < 5; i++)
        raceStatus->focus[i] = focusSensorOut[i];

    raceStatus->laps = car->race.laps;

    if (internalError) {
        raceStatus->speedX = 0.;
        raceStatus->angle = 0.;
        raceStatus->speedY = 0.;
        raceStatus->speedZ = 0.;
        raceStatus->trackPos = 0.;
        raceStatus->isOver = true;
        raceStatus->rpm = 0.;
        for (int i = 0;i < raceStatus->wheelSpinVel.size(); i++) {
            raceStatus->wheelSpinVel[i] = 0.;
        }
    }
    raceStatus->seqIdx++;
}
