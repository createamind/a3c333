//
// Created by Robin Huang on 8/11/17.
//

#ifndef TORCS_1_3_6_RACE_SERVER_H
#define TORCS_1_3_6_RACE_SERVER_H


#include <memory>
#include "race.h"
#include "msp_log.h"
#include "utils.h"
#include <QtCore/QtCore>
#include "recycle_queue.h"
#include "robot.h"
class RaceServer;
#include "data.h"
class _BufStatus : public RecycleQueueDataBase {
public:
    _BufStatus() {
        raceStatus = std::make_shared<Race::Status>();
    }
    Race::StatusPtr raceStatus;
};
class _BufStatusList : public RecycleQueueDataBase {
public:
    Race::StatusList statusList;
};

class _BufDriveInfo : public RecycleQueueDataBase {
public:
    _BufDriveInfo() {
        driveInfo = std::make_shared<Race::DriveInfo>();
    }
    Race::DriveInfoPtr driveInfo;
};

class _BufDriveInfoList : public RecycleQueueDataBase {
public:
    Race::DriveInfoList driveInfoList;
};

class Sensors;
class ObstacleSensors;
class RaceBot {
protected:
    RaceServer & m_server;
    int m_idx;
    std::string m_name;
//    QSemaphore m_semInit;
    Race::StatusPtr m_raceStatus;
    Race::DriveInfoPtr m_driveInfo;
    Race::BotInitParamPtr m_initParams;
    Race::BotRaceInitParamPtr m_raceInitParams;

    Race::BotInfoPtr m_botInfo;

    double __SENSORS_RANGE__;
    tTrack * m_tTrack = nullptr;
    std::shared_ptr<Sensors> m_focusSensor;
    std::shared_ptr<Sensors> m_trackSensor;
    std::shared_ptr<ObstacleSensors> m_oppSensor;
    float m_trackSensAngle[19];
    float m_prevDist = -1.f;
    float m_distRaced = 0.0f;
    uint64_t  m_total_tics = 0;

private:
    long m_seqIdxStatus = -1;
    long m_seqIdxDriveInfo = -1;
public:
    RaceBot(RaceServer & server, int idx);
    virtual ~RaceBot();
    const std::string & getName() const { return m_name; };
    const int getIdx() const { return m_idx; };
    const uint64_t getStep() const { return m_total_tics; };
    virtual void close();
    const Race::BotInfoPtr getBotInfo() const { return m_botInfo; };
    const Race::StatusPtr getRaceStatus() const { return m_raceStatus;};
    const Race::DriveInfoPtr getDriveInfo() const { return m_driveInfo; };
    void setRaceInitParam(const Race::BotRaceInitParamPtr & initParam) { *m_raceInitParams = *initParam; }
//    void setDriveInfo(Race::DriveInfoPtr driveInfo);
    virtual void newTrack(tTrack * track, void * car, void ** carParam, tSituation * s);
    virtual void newRace(tCarElt * car, tSituation * s);
    virtual void endRace(tCarElt * car, tSituation * s);
    virtual void drive(tCarElt * car, tSituation * s) = 0;

//    RecycleQueue<_BufStatus> m_queueStatus;
    virtual void fillStatus(const tCarElt * car, const tSituation * s);
};

class RaceBotHook : public RaceBot {
public:
    RaceBotHook(RaceServer & server, int idx, struct RobotItf * robot, tTrack * track);
public:
    // for hook driver
    static void cbRbNewTrack(int index, tTrack *track, void *carHandle, void **myCarSettings, tSituation *s);
/** Callback prototype */
    static void cbRbNewRace (int index, tCarElt *car, tSituation *s);
/** Callback prototype */
    static void cbRbEndRace (int index, tCarElt *car, tSituation *s);
/** Callback prototype */
    static void cbRbDrive (int index, tCarElt *car, tSituation *s);
/** Callback prototype */
    static void cbRbShutdown(int index);
/** Callback prototype */
    static int cbRbPitCmd (int index, tCarElt* car, tSituation *s);
private:
    virtual void fillStatus(const tCarElt * car, const tSituation * s);

    virtual void newTrack(tTrack * track, void * car, void ** carParam, tSituation * s);
    virtual void newRace(tCarElt * car, tSituation * s);
    virtual void endRace(tCarElt * car, tSituation * s);
    virtual void drive(tCarElt * car, tSituation * s);
    virtual void shutdown();
    virtual int pitCmd(tCarElt * car, tSituation * s);

    tfRbNewTrack m_cbNewTrack = nullptr;
    tfRbNewRace m_cbNewRace = nullptr;
    tfRbEndRace m_cbEndRace = nullptr;
    tfRbDrive m_cbDrive = nullptr;
    tfRbShutdown m_cbShutdown = nullptr;
    tfRbPitCmd m_cbPitCmd = nullptr;

//    Race::DriveInfoPtr m_lastDriveInfo;
};

class RaceBotCtrl : public RaceBot {
public:
    RaceBotCtrl(RaceServer & server, int idx) : RaceBot(server, idx), m_queueDriveInfo(8) {

    }
    virtual void close() {
        RaceBot::close();
        m_queueDriveInfo.close();
    }
    RecycleQueue<_BufDriveInfo> m_queueDriveInfo;
};

class RaceServer : public Race::Server
{
public:
    RaceServer(uint32_t maxBotCount);
    virtual void init(::Ice::Identity, ::std::shared_ptr<::Race::ServerInitParam>, const ::Ice::Current&);
    virtual ::Race::BotInfoList getBots(const ::Ice::Current&);
//    virtual void init(::Ice::Int botIdx, const ::Race::BotInitParamPtr&, const ::Ice::Current& = ::Ice::Current());
//    virtual void keyPress(const std::string & , ::Ice::Int, const ::Ice::Current& = ::Ice::Current());
    virtual ::std::shared_ptr<::Race::StepResult> reset(::std::shared_ptr<::Race::ResetParam>, const ::Ice::Current&);
//    virtual ::Race::StatusPtr getStatus(::Ice::Int, const ::Ice::Current& = ::Ice::Current());
    virtual ::std::shared_ptr<::Race::StepResult> step(::std::shared_ptr<::Race::StepParam>, const ::Ice::Current&);
    virtual void shutdown(const ::Ice::Current &);
    std::shared_ptr<RaceBot> getBot(uint32_t idx);
    void setBot(uint32_t botIdx, std::shared_ptr<RaceBot> bot);

    void onReInitCars(tRmInfo * ReInfo);
    void onReBeforeOneStep(tRmInfo * ReInfo);
    void onReAfterOneStep(tRmInfo * ReInfo);
    void onReBeforeDrive(tRmInfo * ReInfo);
    void onReAfterDrive(tRmInfo * ReInfo);
    Race::PoolPrx getCurrentPool();
    void close();
    bool isFlagEnd() const { return m_flagEnd;};
protected:
//    QSemaphore m_semDriveInfoAvaiable;
//    QSemaphore m_semStatusAvaiable;
    QSemaphore m_semResetSignalAvaiable;
    QSemaphore m_semCarsInited;

    RecycleQueue<_BufStatusList> m_queueStatusList;
    ::Race::ServerInitParamPtr m_serverInitParam;
    void _connect2Dataio();
private:
    bool m_flagEnd = false;
    bool m_flagReset = false;
    bool m_flagCarInited = false;
    uint64_t m_resetCount = 0;
    uint64_t m_stepTotal = 0;
    uint64_t m_stepEpisode = 0;
    std::vector<std::shared_ptr<RaceBot>> m_bots;
    std::shared_ptr<Race::PoolPrx> m_currentClient;
//    QSemaphore m_semClientAvaiable;
//    DataFlow::DataServerPrx m_dataserverPrx;
};


#endif //TORCS_1_3_6_RACE_SERVER_H
