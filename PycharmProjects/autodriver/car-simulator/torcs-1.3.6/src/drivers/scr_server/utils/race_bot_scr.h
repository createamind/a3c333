//
// Created by Robin Huang on 8/8/17.
//

#ifndef SCR_SERVER_RACE_BOT_H
#define SCR_SERVER_RACE_BOT_H
#include <memory>
#include <string>
#include <QtCore/QtCore>
#include "race.h"

class RaceServer;
class Sensors;
class ObstacleSensors;
#include "race_server.h"
class RaceBot_SCR : public RaceBotCtrl {
public:
    RaceBot_SCR(RaceServer & server, int idx);
//    void newTrack(tTrack * track, void * car, void ** carParam, tSituation * s);
//    void newRace(tCarElt * car, tSituation * s);
//    void endRace(tCarElt * car, tSituation * s);
    void drive(tCarElt * car, tSituation * s);
    void close();
};

#endif //SCR_SERVER_RACE_BOT_H
