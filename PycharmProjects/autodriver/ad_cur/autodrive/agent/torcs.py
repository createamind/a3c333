#coding: utf-8

import filelock
import numpy as np
from ..utils import logger
from .base import AgentBase
from .simulator.gym_torcs import TorcsEnv


class AgentTorcs(AgentBase):
    @staticmethod
    def startNInstance(count, **kwargs):
        for aidx in range(count):
            if TorcsEnv.checkEnvExist(aidx):
                logger.info("agent {} already run, skip starting".format(aidx))
                continue
            logger.info("start Agent {}".format(aidx))
            env = AgentTorcs.initEnv(aidx, **kwargs)
            del env

    @staticmethod
    def initEnv(agentIdent, **kwargs):
        winpos = (int(640 * (agentIdent % 6)),  480 * int(agentIdent // 6))
        return TorcsEnv(agentIdent, vision=kwargs.get("vision", False),
                               throttle=kwargs.get("throttle", True),
                               gear_change=kwargs.get('gear_change', False),
                               winpos=winpos)

    def _init(self):
        self._torcs = AgentTorcs.initEnv(self._agentIdent, **self._kwargs)
        self._totalSteps = 0
        if self._isTrain:
            self._exploreEpisode = 1.
            self._exploreDecay = 1. / 100000.
            self._speedHist = []
        super(AgentTorcs, self)._init()

    def _reset(self):
        if self._isTrain:
            self._histObs = []
            # if self._exploreEpisode > 0:
            #     self._exploreEpisode -= self._exploreDecay
            self._maxSteps = -1
            # if self._episodeCount <= 1: # 增加agent之间的异步，防止训练样本相关性太大
            #     self._maxSteps = self._rng.randint(50, 500)
            self._maxStepsCheckBlocking = self._rng.randint(50, 70)
        self._speedMax = 0.
        # memory leak of torcs
        # ob = self._torcs.reset(relaunch=True)
        ob = self._torcs.reset(relaunch=(self._episodeCount%300==299))
        self._data_ob = np.zeros((29, 4), dtype=np.float32)
        return self._selectOb(ob)

    def _selectOb(self, ob):
        self._ob_orig = ob
        if self._isTrain:
            self._histObs.append(ob)
        ret = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
        ret = ret.astype(np.float32)
        # self._data_ob[:, :-1] = self._data_ob[:, 1:]
        # self._data_ob[:, -1] = ret
        # return self._data_ob.reshape((-1,))
        return ret

    def _ouProcess(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * self._rng.randn(1)

    def _step(self, predict):
        action, value = predict
        assert (len(action.shape) == 1 and action.shape[0] == 2)
        act_save = np.zeros_like(action)
        act_save[:] = action[:]
        action[1] = 0.5
        if self._isTrain:
            # self._exploreEpisode -= self._exploreDecay
            # action[0] = max(self._exploreEpisode, 0) * self._ouProcess(action[0], 0.0, 0.60, 0.30)
            # if action[1] >= 0:
            #     action[1] = max(self._exploreEpisode, 0) * self._ouProcess(action[0], 0.5 , 1.00, 0.10)
            # else:
            #     action[1] = max(self._exploreEpisode, 0) * self._ouProcess(action[0], -0.1, 1.00, 0.05)
            # 能否在初期得到比较好的reward决定了收敛的快慢，所以此处加入一些先验
            # 新手上路，方向盘保守一点，带点油门，不踩刹车
            if action[1] < 0 and len(self._histObs) >= 10:
                past_max_speed = np.max([o.speedX for o in self._histObs[-10:]])
                if past_max_speed < (50/300.) or self._rng.rand() <= 0.01:
                    action[1] = self._rng.rand() * 0.5 +.5
            # #
            if self._speedMax < (50/300.):
                action[0] = np.clip(action[0], -0.3, 0.3)
                if action[1] < 0 or self._rng.rand() < 0.2:
                    action[1] = self._rng.rand() * 0.2 + 0.3
            if self._rng.rand() < 0.8 \
                    and ((self._ob_orig.trackPos >= 0.5 and action[0] > 0.01) or (self._ob_orig.trackPos <= -0.5 and action[0] < -0.01)):
                if self._ob_orig.trackPos > 0: action[0] = -self._rng.rand() * 0.5
                elif self._ob_orig.trackPos < 0: action[0] = self._rng.rand() * 0.5

            # elif self._speedMax < (100/300.):
            #     action[0] = np.clip(action[0], -0.3, 0.3)
            #     if self._rng.rand() < 0.1:
            #         action[1] = self._rng.rand() * 0.5 + 0.3
            # elif self._speedMax < (200/300.):
            #     action[0] = np.clip(action[0], -0.3, 0.3)

        ob, _reward, is_over, info = self._torcs.step(action)
        reward = 0.
        border = 1.2
        speed = ob.speedX * 300.
        if speed <= 10:
            speed_coff = speed / 10.
        else:
            speed_coff = np.log10(speed)
        # _reward = speed_coff * (border - min(border, abs(ob.trackPos)))
        _reward = 1. / (1 + np.exp(-(-np.abs(ob.trackPos) + 1) * 8)) - 0.1
        reward = _reward + (ob.speedX * 300.) / 300.
        steeringLoss = action[0] * ob.trackPos * (0.1 + 0.1 * speed_coff)
        logger.info("steering loss = {:.04f}, steering={:.4f}, trackPos={:.4f}".format(steeringLoss, action[0], ob.trackPos))
        reward -= steeringLoss

        # if self._episodeSteps <= 30 and ob.damage > 0.:
        #     is_over = True
        if self._speedMax < ob.speedX: self._speedMax = ob.speedX
        # if self._speedMax >= (50 / 300.) and ob.speedX <= (15. / 300.):  #
        #     if reward > 0: reward *= abs(ob.speedX / self._speedMax)

        track = np.array(ob.track)
        trackPos = ob.trackPos
        # if abs(trackPos) > 1.001:
        #     logger.info("out of track, trackPos={}".format(trackPos))
        #     is_over = True
        trackLoss = -(np.abs(track.any()) - 1.)
        trackPosLoss = -(abs(trackPos) - 1.)
        # 车辆应该在路中间
        # if self._episodeSteps >= 100 and ob.speedX > (50/300.):
        #     reward -= abs(trackPos) * 0.01



        if trackLoss < -1e-4 or trackPosLoss < -1e-4:
            # 不奖励离开车道的行为
            # reward  = -0.1
            if self._speedMax < (60./300.):
                reward = -1.
                is_over = True
        #     if reward >= 0.: reward *= 0.5
        #     if trackLoss < 0:
        #         reward += trackLoss
        #     if trackPosLoss < 0:
        #         reward += trackPosLoss
        #     logger.info("trackLoss={:.4f}[{}], trackPosLoss={:.4f}[{:.4f}".format(trackLoss, track, trackPosLoss, trackPos))
        # if self._isTrain and self._maxSteps > 0 and self._episodeSteps > self._maxSteps:
        #     logger.info("episode {} reach max steps {}, finish episode".format(self._episodeCount, self._maxSteps))
        #     is_over = True
        maxspeed = -1.
        if self._isTrain:
            if  abs(trackPos) >= 0.9 and self._episodeSteps > self._maxStepsCheckBlocking:
                speeds = np.array([v.speedX for v in self._histObs[-self._maxStepsCheckBlocking:]])
                maxspeed = np.max(np.abs(speeds))
                if maxspeed <= (15./300.): # or np.where(speeds <= -1e-5)[0].shape[0] >= 3:
                    logger.info("episode {}: max speed {} too small, maybe blocked, restart".format(self._episodeCount, maxspeed))
                    is_over = True
            # if self._episodeSteps >= 100:

        # reward = self._rng.rand() * 2. -1.
        logger.info("[{}]: step {}: act={:+.3f}/{:+.3f}[{:+.3f}/{:+.3f}], value={:.3f}, reward={:+.4f}[{:+.4f}][total={:.4f}], speed.max/pastmax={:.4f}/{:.4f}, ob={}, is_over={}"
                    .format(self._episodeCount, self._episodeSteps,
                            action[0], action[1], act_save[0], act_save[1], value,
                            reward, _reward, self._episodeRewards,
                            self._speedMax*300, maxspeed*300,
                            ob,
                            is_over))
        return self._selectOb(ob), action, float(reward), is_over, info

