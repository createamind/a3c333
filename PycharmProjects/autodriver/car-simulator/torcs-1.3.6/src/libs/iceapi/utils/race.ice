#pragma once

#include <Ice/Identity.ice>
module Race {
sequence<float> FloatSeq;
sequence<int> IntSeq;
sequence<byte> ByteSeq;
class DriveInfo {
    int ident = -1;
    long  seqIdx = -1;
    float steering = 0.0; // <0 向右打方向，>0向左打方向
    float acceleration = 0.0;
    float brake = 0.0;
    float  clutch = 0.0;
    int focus = 0;
    int gear = 0;
};

class Image {
    int width;
    int height;
    ByteSeq data;
};
class Status {
    int ident = -1;
    long seqIdx = -1;
    float angle = 0.;   // > 0 头偏右，<0头偏左
    float curLapTime = 0.;
    float damage = 0.;
    float distFromStart = 0.;
    float distRaced = 0.;
    float fuel = 0.;
    int gear = 0;
    float lastLapTime = 0.;
    FloatSeq opponents;
    int racePos = 0;
    float rpm = 0.0;
    float speedX = 0.0;
    float speedY = 0.0;
    float speedZ = 0.0;
    FloatSeq track;
    float trackPos = 0.0;       // >0 在路左侧，<0在路右侧
    FloatSeq wheelSpinVel;
    float z = 0.0;
    FloatSeq focus;
    int laps = 0;
    DriveInfo driveInfo;

    Image image;

    bool isOver = false;
};



class BotInitParam {

};

class BotRaceInitParam {
    int ident = -1;
    FloatSeq angles;
    bool screenCapture = false;
};

class BotInfo {
    int idx;
    string name;
};

class EventParams {
};

sequence<BotInfo> BotInfoList;
sequence<Status> StatusList;
sequence<DriveInfo> DriveInfoList;
sequence<BotRaceInitParam> BotRaceInitParamList;
class StepParam {
    DriveInfoList driveInfos;
};

class ResetParam {
    BotRaceInitParamList raceInit;
};
class StepResult {
    StatusList statusList;
    bool isOver = false;
    bool isAllFinished = false;
};
class ServerInitParam {
    string dataioHost;
    int dataioPort;
    string name;
    float timeMult = 1.; // >= 1/128, <= 64
};

interface Pool {
    void onEvent(string evt, EventParams param);
    DriveInfoList predict(StatusList status);
};

interface Server {
    void init(Ice::Identity ident, ServerInitParam param);
    BotInfoList getBots();
    StepResult step(StepParam param);
    StepResult reset(ResetParam param);
    // void keyPress(string key, int duration);
    void shutdown();
};

};