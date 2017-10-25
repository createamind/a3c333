#coding: utf-8

import filelock
import numpy as np
from ..utils import logger
from .base import AgentBase
from .simulator.gym_torcs import TorcsEnv
import Ice, os, sys
Ice.loadSlice("-I" + Ice.getSliceDir() + " --all " + os.path.dirname(__file__) + "/race.ice")
sys.path.append(os.path.dirname(__file__) + "/slice")
Ice.updateModules()
import Race

class _TorcsBot(object):
    def __init__(self, name, idx, agent):
        self._name = name
        self._idx = idx
        self._agent = agent     # type: AgentTorcs2
        self._init()

    def _init(self):
        self._cur_status = None
        self._cur_driveInfo = Race.DriveInfo()
        self._cur_driveInfo.ident = self._idx
        self._isHookBot = not self._name.startswith("scr_server")
        self._prev_rpm = None
        self._reward = 0.
        self._isOver = False
        self._hist_status = []
        self._hist_actions = []
        self._cur_laps = 0
        self._speed_max = 0.
        self._cur_value = 0.
        # self._hist_ob = []

    def getRaceInitParam(self):
        angles = [0 for x in range(19)]
        for i in range(5):
            angles[i] = -90. + i * 15
            angles[18 - i] = 90. - i * 15

        for i in range(5, 9):
            angles[i] = -20. + (i - 5) * 5
            angles[18 - i] = 20. - (i - 5) * 5

        initParam = Race.BotRaceInitParam()
        initParam.angles = angles
        initParam.ident = self._idx
        initParam.screenCapture = self._agent._kwargs.get("screen_capture", False)
        return initParam

    def parseStatus(self, status):
        speedXDelta = angleDelta = trackPosDelta = 0.

        focus = np.array(status.focus)
        _focusScaled = focus / 200.
        if _focusScaled.max() < 0: _focusScaled = -1 * np.ones_like(_focusScaled)
        focusDelta = np.zeros_like(focus)
        last_status = None
        if len(self._hist_status) > 0:
            last_status = self._hist_status[-1]
            speedXDelta = (status.speedX - last_status.speedX)
            angleDelta = (status.angle - last_status.angle)
            trackPosDelta = (status.trackPos - last_status.trackPos)
            lastFocus = np.array(last_status.focus)
            if focus.min() > 0 and lastFocus.min() > 0:
                focusDelta = focus - lastFocus

        _track = np.array(status.track) / 200.
        if _track[0] < 0: _track = -1 * np.ones_like(_track)
        self._ob_status = np.hstack((_focusScaled,
                                     status.angle / np.pi,
                                     _track,
                                     status.trackPos,
                                     status.speedX / 200.,
                                     status.speedY / 200.,
                                     status.speedZ / 200.,
                                     np.array(status.wheelSpinVel) / 100.0,
                                     status.rpm / 10000.,
                                     speedXDelta,
                                     angleDelta * 10.,
                                     trackPosDelta * 10.,
                                     ))
        assert (np.isfinite(self._ob_status).all()), 'status has infinite: {}'.format(self._ob_status)
        assert(not np.isnan(self._ob_status).any())

        if status.speedX > self._speed_max:
            self._speed_max = status.speedX
        # self._hist_ob.append(self._ob_status)
        if self._agent._episodeSteps > 0:
            speedCoff = 2 * (1. / (1 + np.exp(-(-np.abs(status.trackPos) + 0.8) * 10)) - (1.0 / (1 + np.exp(0))))
            steerCoff = 1.0 - (1. / (1 + np.exp(-(-np.abs(status.trackPos) + 1) * 8)))
            speedScale = status.speedX / 300.
            reward_speed = speedCoff * speedScale
            reward_steer = 0.
            reward_angle = (angleDelta) * speedScale * (_focusScaled.min())
            reward_focus = ((focus.max() - 30)/200.) * speedScale * speedCoff
            reward_dangerous = 0.
            last_actions = np.array([a[0] for a in self._hist_actions[-10:]])
            if focus.min() < 20 and len(self._hist_status) > 5:
                last_focus_array = np.array([np.array(s.focus).min() for s in self._hist_status[-6:]])
                diff = last_focus_array[1:] - last_focus_array[:-1]
                if (diff < 0).all():
                    reward_dangerous = 100.0 / min(100., (focus.min() / diff[-1]))

            reward_brake = 0.
            if focus.max() < 20:
                # reward_focus = 0.
                reward_brake = -speedXDelta * speedScale * 0.1
                reward_steer = 0.5 * steerCoff * (-self._cur_driveInfo.steering * status.trackPos) #* (-trackPosDelta*status.trackPos*0.1 + -self._cur_driveInfo.steering * status.trackPos) \
                                         #  * (min(30/300., speedScale))
            # reward_steer *= (0.01 + status.speedX/300.)
            reward = np.sum([
                reward_speed,
                reward_focus,
                # reward_brake,
                reward_steer,
                reward_dangerous,
                # reward_angle,
            ])
            # _reward = trackPosCoff - 0.1
            # reward = (_reward * 0.5 + 0.5) * (status.speedX) / 300.
            # reward_pos = (1. - trackPosCoff) * (-trackPosDelta * status.trackPos) * (1 + status.speedX/300.) * 20.
            # reward += reward_pos
            # reward = speedScale * (np.cos(status.angle) - np.sin(status.angle) - np.abs(status.trackPos))
            # steeringLoss = self._ob_ctrl[0] * status.trackPos * (0.1 + 0.1 * speed_coff)
            # logger.info("steering loss = {:.04f}, steering={:.4f}, trackPos={:.4f}".format(steeringLoss, action[0], ob.trackPos))
            # reward -= steeringLoss
            self._reward = reward
            di = self._cur_driveInfo
            if status.damage > last_status.damage:
                self._reward = -1.
            # logger.info("curLapTime = {}".format(status.curLapTime))
            # logger.info("focus = {}, speedXDelta={:.2f}".format(['{:.3f}'.format(f) for f in focus.tolist()], speedXDelta))
            # logger.info("track = {}".format(['{:.3f}'.format(f) for f in _track.tolist()]))
            #logger.info("S={:+06.1f}[{:05.1f}] P={}{:+.3f} R={:+.3f}[{:+.3f}/{:+.3f}/{:+.3f}/{:+.3f}] V={:+4.1f} S/A/B={}{:+.3f}/{:+.3f}/{:+.3f} M/S={:+.3f}\u00B1{:.2f}/{:+.3f}\u00B1{:.2f} A={:.3f}[{}{:+.3f}] D={:.1f}[{:.1f}] F={:.1f}/{:.1f}" #, C={:.2f}, G={}[{:.1f}]"
             #           .format(status.speedX, self._speed_max,
              #                  'L' if status.trackPos > 0 else 'R', status.trackPos,
               #                 self._reward,
                #                reward_speed, reward_steer, reward_focus, reward_dangerous,
                 #               self._cur_value,
                  #              'L' if di.steering > 0 else 'R', di.steering, di.acceleration, di.brake,
                   #             self._cur_mu[0], self._cur_sigma[0], self._cur_mu[1], self._cur_sigma[1],
                    #            np.cos(status.angle), 'HL' if status.angle < 0 else 'HR', status.angle,
                     #           status.damage-last_status.damage, last_status.damage,
                      #          focus.min(), focus.max(),
                                # di.clutch,
                                # di.gear, status.rpm,
                       #         ))

        self._hist_status.append(status)
        if self._cur_laps != status.laps:
            logger.info("[Bot]: {}: laps {}, steps = {}".format(self._name, status.laps, self._agent._episodeSteps))
            self._cur_laps = status.laps

        check_steps = 100 if abs(status.trackPos) >= 1.0 else 5000
        if len(self._hist_status) >= check_steps:
            speeds = np.array([s.speedX for s in self._hist_status[-check_steps:]])
            maxspeed = np.max(speeds)
            if maxspeed <= 5:
                logger.info("[Bot]: {}: episode {}: max speed {} too small, maybe blocked, restart"
                            .format(self._name, self._agent._episodeCount, maxspeed))
                self._reward = -1.
                self._isOver = True
            del self._hist_status[0]
        if status.isOver:
            logger.info("[Bot]: {}: internal over, maybe internal error".format(self._name))
            self._isOver = True
        if np.cos(status.angle) < 0.5:  # Episode is terminated if the agent runs backward
            logger.info("[Bot]: {}: episode {} runs backward, end race"
                        .format(self._name, self._agent._episodeCount))
            self._reward = -1.
            self._isOver = True
        if _focusScaled.max() < 0:
            logger.info("[Bot]: {} episode {} focus no range, end race".format(self._name, self._agent._episodeCount))
            self._reward = -1.
            self._isOver = True

        if status.image:
            assert(len(self._agent._bots) == 1)
            self._cur_screen = self._agent._cur_screen = np.frombuffer(bytearray(status.image.data), dtype=np.uint8).reshape((status.image.height, status.image.width, 3))

        if self._isHookBot:
            if self._agent._episodeSteps > 0:
                assert (status.driveInfo is not None)
            if status.driveInfo:
                self._cur_driveInfo = di = status.driveInfo
                self._ob_ctrl = ob_ctrl = np.zeros(2, dtype=np.float32)
                ob_ctrl[0] = di.steering
                if di.brake > 1e-5:
                    ob_ctrl[1] = -di.brake
                else:
                    ob_ctrl[1] = di.acceleration
                # logger.info('[{:02d}]: lap={}, Speed = {:.4f}, steering={:+.6f}, accel={:.4f}, brake={:.4f}, gear={}, clutch={:.4f}'
                #       .format(self._idx, status.laps, status.speedX, di.steering, di.acceleration, di.brake, di.gear, di.clutch))
        else:
            assert(status.driveInfo is None)
        self._cur_status = status

    def parsePredict(self, predict):
        action, value, mu, sigma = predict
        self._cur_value = value
        self._cur_mu = mu
        self._cur_sigma = sigma
        if self._isHookBot: return
        di = self._cur_driveInfo
        status = self._cur_status
        if self._agent._isTrain:
            # def ema(values, window):
            #     weights = np.exp(np.linspace(-1., 0., window))
            #     weights /= weights.sum()
            #     a = np.convolve(values, weights, mode='full')[:len(values)]
            #     a[:window] = a[window]
            #     return a
            # if len(self._hist_actions) > 5:
            #     action[0] = ema([a[0] for a in self._hist_actions[-6:]], 5)[-1]
            rng = self._agent._rng
            focus = np.array(status.focus)
            # if self._speed_max >= 100:
            #     if rng.rand() < 0.1:
            #         action[1] = -rng.rand() / 10.
            # if abs(self._cur_status.trackPos) > .5:
            #     if rng.rand() < 0.9:
            #         action[0] = -np.sign(self._cur_status.trackPos) * rng.rand()

            # if np.min(focus) > 0 and np.max(focus) < 30:
            #     if rng.rand() < 0.5:
            #         action[1] = rng.rand() * 0.5
            #     if rng.rand() < 0.5:
            #         action[0] = 0.5 * -rng.rand() * np.sign(self._cur_status.trackPos)
            #     if rng.rand() < 0.1:
            #         action[1] = -rng.rand() * 0.5
            # action[0] = self._agent._ouProcess(action[0], 0, 0.6, 0.3)
            angle = np.cos(status.angle)
            prob = 1. - np.minimum(focus.max() / 30., 1.)
            if focus.max() < 20 and (abs(status.angle) > 0.25 ) and rng.rand() < 0.1:
                steer_lock = 0.785398
                steer_lock = 3.
                steer = (status.angle - status.trackPos) / steer_lock
                action[0] = steer
                logger.info("[{:.2f}]: correct steer for angle = {}{:+.3f}".
                            format(prob, 'L' if steer > 0 else 'R', steer))
            if status.speedX <= 15:
                if rng.rand() < 0.9:
                    action[1] = 1.
                    action[0] = (status.angle - status.trackPos) / 0.785398
            else:
                if self._agent._exploreStepMax > 0 and self._agent._exploreStep < self._agent._exploreStepMax:
                    action = self._agent.sample_action()
                if self._agent._maxSpeedLimit > 0 and self._cur_status.speedX >= self._agent._maxSpeedLimit:
                    action[1] = di.acceleration - 0.1
            # if action[1] < 0:
            #     action[1] = np.clip(action[1], -0.1, 0.)
        # logger.info("agent {} action={}, speedMax={}, isTrain={}".format(self._agent._agentIdent, action, self._speed_max, self._agent._isTrain))
        action[0] = np.clip(action[0], -1., 1.)
        action[1] = np.clip(action[1], -1., 1.)
        di.steering = float(action[0])
        di.acceleration = float(action[1] if action[1] > 0. else 0.)
        di.brake = float(-action[1] if action[1] < 0. else 0.)
        self._ob_ctrl = action
        status = self._cur_status
        rpm = status.rpm
        gear = status.gear
        if len(action.shape) <= 2:
            if self._prev_rpm is None:
                up = True
            else:
                if (self._prev_rpm - rpm) < 0:
                    up = True
                else:
                    up = False
            if up and rpm > 7000: gear += 1
            if not up and rpm < 3000: gear -= 1
        else:
            gear = int(action.shape[2])
        di.gear = max(gear, 1)
        di.ident = self._idx
        assert(not np.isnan(di.steering))
        assert(not np.isnan(di.acceleration))
        assert(not np.isnan(di.brake))
        self._hist_actions.append(action)
        return di

    def sample_action(self):
        if self._isHookBot:
            return np.zeros(2, dtype=np.float32)
        driveInfo = self._cur_driveInfo
        driveInfo.restart = False
        status = self._cur_status
        angle = status.angle
        dist = status.trackPos

        steer_lock = 0.785398
        max_speed = 120
        steer = (angle - dist) / steer_lock

        speed = status.speedX
        accel = driveInfo.acceleration
        if speed < max_speed:
            accel += 0.1
            if accel > 1:
                accel = 1.0
        else:
            accel -= 0.1
            if accel < 0:
                accel = 0.0
        return np.array([steer, accel], dtype=np.float32)

from drlutils.agent.single_loop import AgentSingleLoop
class AgentTorcs2(AgentSingleLoop, Race.Pool):
    tracks = {'road': ['aalborg', 'alpine-1', 'alpine-2', 'brondehach', 'corkscrew', 'eroad', 'e-track-1', 'e-track-2', 'e-track-3', 'e-track-4', 'e-track-6', 'forza',
                       'g-track-1', 'g-track-2', 'g-track-3', 'ole-road-1', 'ruudskogen', 'spring', 'street-1', 'wheel-1', 'wheel-2'],
              # 'dirt': ['dirt-%d' % (i + 1) for i in range(6)] + ['mixed-0', 'mixed-1'],
              'oval': ['%s-speedway' % i for i in 'abcdefg'] + ['e-track-5', 'michigan'],
              }
    botNames = ['berniw', 'berniw2', 'berniw3', 'bt', 'damned', 'inferno', 'inferno2', 'lliaw', 'olethros', 'tita']
    @staticmethod
    def startNInstance(count, **kwargs):
        # for aidx in range(count):
        #     if TorcsEnv.checkEnvExist(aidx):
        #         logger.info("agent {} already run, skip starting".format(aidx))
        #         continue
        #     logger.info("start Agent {}".format(aidx))
        #     env = AgentTorcs2.initEnv(aidx, **kwargs)
        #     del env
        pass

    @staticmethod
    def initEnv(agentIdent, **kwargs):
        winpos = (int(640 * (agentIdent % 6)),  480 * int(agentIdent // 6))
        # return TorcsEnv(agentIdent, vision=kwargs.get("vision", False),
        #                        throttle=kwargs.get("throttle", True),
        #                        gear_change=kwargs.get('gear_change', False),
        #                        winpos=winpos)

    def _init(self):
        super(AgentTorcs2, self)._init()
        logger.info("agent {} init: isTrain={}".format(self._agentIdent, self._isTrain))
        import os
        from ..utils import ensure_dir_exists
        self._torcs_dir = os.path.expanduser('~/torcs1.3.6')
        if not os.path.exists(self._torcs_dir + '/lib/torcs/torcs-bin'):
            raise IOError("torcs not correct installed, please install torcs into HOME/torcs1.3.6")

        self._torcs_run_dir = ensure_dir_exists(self._torcs_dir + '/run')
        self._torcs_profile_dir = ensure_dir_exists(os.path.expanduser('~/.torcs-{:04d}'.format(self._agentIdent)))
        if not os.path.exists(self._torcs_profile_dir + '/config'):
            os.system('cp -a ~/.torcs/* {}'.format(self._torcs_profile_dir))
        self._init_Ice()

        self._torcs_proc = None
        self._raceServerPrx = None
        self._cur_track = None

        self._text_mode = self._kwargs.get('text_mode', False)
        self._softResetCountInTextMode = 0
        if self._text_mode:
            logger.info("[{:04d}]: run with text mode".format(self._agentIdent, self._text_mode))
        self._start_torcs()
        # self._torcs = AgentTorcs2.initEnv(self._agentIdent, **self._kwargs)
        self._totalSteps = 0
        if self._isTrain:
            self._exploreEpisode = 1.
            self._exploreDecay = 1. / 100000.
            self._speedHist = []

    def _start_torcs(self):
        self._ensure_torcs_running(bots=self._kwargs.get('bots', None),
                                   track=self._kwargs.get('track', 'road/g-track-1'),
                                   )

    def _ensure_torcs_stop(self):
        if self._text_mode and self._raceServerPrx:
            try:
                self._raceServerPrx.begin_shutdown()
            except Ice.Exception:
                pass
            con = self._raceServerPrx.ice_getCachedConnection()
            if con:
                con.close(Ice.ConnectionClose.Forcefully)
            self._raceServerPrx = None
        pidfile = self._torcs_run_dir + '/.{:04d}.pid'.format(self._agentIdent)
        if self._torcs_proc:
            pid = self._torcs_proc.pid
            import signal
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        else:

            from ..utils import terminate_pid_file
            terminate_pid_file(pidfile)
        if os.path.exists(pidfile):
            os.unlink(pidfile)

    def _init_Ice(self):
        import Ice
        props = Ice.createProperties()
        props.setProperty("Ice.ThreadPool.Server.SizeMax", "16")
        # props.setProperty("Ice.ThreadPool.SizeMax", "16")
        props.setProperty("Ice.MessageSizeMax", "0")
        # props.setProperty("Ice.Trace.ThreadPool", "1")
        # props.setProperty("Ice.Trace.Network", "1")
        # props.setProperty("Ice.Trace.Protocol", "1")
        data = Ice.InitializationData()
        data.properties = props
        self._ice_ic = ic = Ice.initialize(data=data)
        self._ice_adapter = adapter = ic.createObjectAdapter("")
        self._ice_ident = ident = Ice.Identity()
        ident.name = Ice.generateUUID()
        adapter.add(self, ident)
        adapter.activate()

    def _set_race_config_file(self, config_xml_file, track = None, bots = None, laps = None):
        if track is None and bots is None and laps is None: return
        if bots is not None: assert (type(bots) in [list, tuple])
        import xml.etree.ElementTree as ET
        tree = ET.parse(config_xml_file)
        root = tree.getroot()
        hasChange = False
        if track:
            if track == 'random':
                tcat = self._rng.choice(list(self.tracks.keys()))
                tname = self._rng.choice(self.tracks[tcat])
                logger.info("[{:04d}]: select random track {}[{}]".format(self._agentIdent, tname, tcat))
            else:
                tcat, tname = track.split('/')
                assert (tcat in self.tracks.keys())
                assert (tname in self.tracks[tcat])
            self._cur_track = tcat + '/' + tname
            d = root.findall(".//*[@name='Tracks']")[0]
            if d[1][0].attrib['val'] != tname or d[1][1].attrib['val'] != tcat:
                d[1][0].attrib['val'] = tname
                d[1][1].attrib['val'] = tcat
                hasChange = True
        if laps:
            assert(int(laps) > 0)
            d = root.findall(".//*[@name='laps']")[0]
            if d.attrib['val'] != str(laps):
                d.attrib['val'] = str(laps)
                hasChange = True
        if bots:
            d = root.findall(".//*[@name='Drivers']")[0]
            bot_section = d.findall('section')
            needChange = False
            if len(bot_section) != len(bots):
                needChange = True
            else:
                for bidx, bot in enumerate(bots):
                    if bot == 'random':
                        needChange = True
                        break
                    if bot != bot_section[bidx].findall('attstr')[0].attrib['val']:
                        needChange = True
                        break
            if needChange:
                for sd in bot_section:
                    d.remove(sd)
                for bidx, bot in enumerate(bots):
                    if bot != 'scr_server' and bot != 'random':
                        assert (bot in self.botNames)
                    if bot == 'random': bot = self._rng.choice(self.botNames)
                    es = ET.SubElement(d, 'section', attrib={'name': str(bidx + 1)})
                    ET.SubElement(es, 'attnum', attrib={'name': 'idx', 'val': str(bidx + 1)})
                    ET.SubElement(es, 'attstr', attrib={'name': 'module', 'val': bot})
                hasChange = True
            pass
        if hasChange:
            logger.info("Rewrite config file {}".format(config_xml_file))
            tree.write(config_xml_file)

    def _ensure_torcs_running(self, raceman = 'quickrace',
                              bots = None,
                              track = None,
                              nofuel = True, nodamage = False, nolaptime=False,
                              screenCapture = False,
                              ):
        if self._text_mode and screenCapture:
            raise Exception("screen capture mode can not run in text_mode")
        if screenCapture and bots and len(bots) > 1:
            raise Exception("only one bot per instance allowd in screen capture mode")
        from ..utils import check_proc_exist
        import filelock, os
        torcsIdent = self._agentIdent + int(self._kwargs['torcsIdxOffset'])
        lockfile = self._torcs_run_dir + '/.{:04d}.lock'.format(torcsIdent)
        lock = filelock.FileLock(lockfile)
        try:
            with lock.acquire(30):
                pidfile = self._torcs_run_dir + '/.{:04d}.pid'.format(torcsIdent)
                port = self._kwargs.get('port_base', 30000) + torcsIdent
                cmd = 'export LD_LIBRARY_PATH={}/lib/torcs/lib;ulimit -c unlimited;cd {}/share/games/torcs;{}/lib/torcs/torcs-bin -l {} -L {}/lib/torcs -D {}/share/games/torcs -pidfile {} -port {} -title {}' \
                    .format(self._torcs_dir, self._torcs_dir, self._torcs_dir,
                            self._torcs_profile_dir,
                            self._torcs_dir, self._torcs_dir, pidfile,
                            port, str(port),
                            )
                config_xml_file = self._torcs_profile_dir + '/config/raceman/{}.xml'.format(raceman)
                self._set_race_config_file(config_xml_file, track=track, bots=bots,
                                           laps=self._kwargs.get('laps', 20),
                                           )
                if self._text_mode:
                    cmd += ' -r {}'.format(config_xml_file)

                if nofuel: cmd += ' -nofuel'
                if nodamage: cmd += ' -nodamage'
                if nolaptime: cmd += ' -nolaptime'

                cmd_save_file = self._torcs_run_dir + '/.cmdline-{:04d}'.format(torcsIdent)
                from ..utils import readlinesFromFile, writelinesToFile, read_pid_file
                oldcmdline = readlinesFromFile(cmd_save_file)[0]
                from time import sleep
                for runretry in range(3):
                    need_run_torcs = (not check_proc_exist(pidfile)) or oldcmdline != cmd
                    if need_run_torcs:
                        self._ensure_torcs_stop()
                        import subprocess, os
                        self._torcs_proc = subprocess.Popen([cmd + ' 2>&1 > {}/run-{:04d}.out &'.format(self._torcs_run_dir, torcsIdent)],
                                                            shell=True, preexec_fn=os.setsid)
                        if self._episodeCount == 0:
                            logger.info("[{:04d}]: run torcs cmd: {}".format(torcsIdent, cmd))
                        else:
                            logger.debug("[{:04d}]: run torcs cmd: {}".format(torcsIdent, cmd))
                        writelinesToFile(cmd_save_file, cmd)
                    else:
                        logger.info("[{:04d}]: torcs already run, pid={}".format(torcsIdent, read_pid_file(pidfile)))

                    host = self._kwargs.get('ice_host', '127.0.0.1')
                    endpoint = 'RaceServer:tcp -h {} -p {} -t {}'.format(host, port, self._kwargs.get('ice_timeout', 3000))
                    logger.debug("[{:04d}]: connecting to {}".format(torcsIdent, endpoint))
                    bots = []
                    for retry in range(10):
                        try:
                            proxy = self._ice_ic.stringToProxy(endpoint)
                            self._raceServerPrx = proxy = Race.ServerPrx.checkedCast(proxy)
                            logger.debug("[{:04d}]: connected to {}".format(torcsIdent, endpoint))
                            proxy.ice_getConnection().setAdapter(self._ice_adapter)
                            param = Race.ServerInitParam()
                            param.name = self._kwargs.get("poolName")
                            dio_address = self._kwargs.get("dio_addr", None)
                            if dio_address:
                                param.dataioHost = dio_address[0]
                                param.dataioPort = dio_address[1]
                            param.timeMult = self._kwargs.get("timeScale", 1.)
                            proxy.init(self._ice_ident, param)
                            if need_run_torcs and (not self._text_mode):
                                os.system('bash {}/startrace.sh {} {}'.format(os.path.dirname(__file__), str(port), raceman))
                                logger.debug("[{:04d}]: run raceman {}".format(torcsIdent, raceman))

                            bots =  proxy.getBots()
                            logger.info("[{:04d}]: bots running = {}".format(torcsIdent, [b.name for b in bots]))
                            break
                        except Ice.ConnectionRefusedException:
                            sleep(0.1)
                            continue
                    if len(bots) > 0: break
                    if track and track == 'random':
                        self._set_race_config_file(config_xml_file, track, bots)
                    if os.path.exists(pidfile):
                        os.unlink(pidfile)

        except filelock.Timeout:
            logger.error("[{:04d}: can not lock file {}".format(self._agentIdent, lockfile))

    def onEvent(self, evt, param, current = None):
        if evt == 'close':
            logger.info("remote request close")
            self._raceServerPrx = None

    def _reset(self):
        logger.info("[{:04d}]: start resetting".format(self._agentIdent))
        if self._text_mode:
            self._ensure_torcs_stop()
            logger.info("[{:04d}]: enable hard resetting".format(self._agentIdent))
        maxEpisodeSteps = self._kwargs.get('maxEpisodeSteps', None)
        self._maxEpisodeSteps = -1
        if maxEpisodeSteps is not None:
            if isinstance(maxEpisodeSteps, tuple):
                self._maxEpisodeSteps = self._rng.randint(maxEpisodeSteps[0], maxEpisodeSteps[1])
                logger.info("set maxEpisodeSteps = {}".format(self._maxEpisodeSteps))
            elif isinstance(maxEpisodeSteps, int):
                self._maxEpisodeSteps = maxEpisodeSteps
                logger.info("set maxEpisodeSteps = {}".format(self._maxEpisodeSteps))
            else:
                assert(0)
        if self._isTrain:
            pass
            self._maxSpeedLimit = -1
            self._explore = self._rng.rand() * 0.1
            self._exploreStepMax = 0
            self._exploreStep = 0

            # self._maxSpeedLimit = 100 + (self._episodeCount * 5)

        for retry in range(10):
            if self._raceServerPrx is None:
                self._start_torcs()

            try:
                bots = self._raceServerPrx.getBots()

                break
            except Ice.ConnectionRefusedException:
                self._raceServerPrx = None
            from time import sleep

        if len(bots) == 0:
            raise Exception("logical error, bots is empty")

        scr_count = 0
        self._bots = []     # type: list[_TorcsBot]
        for b in bots:
            logger.debug("create TrocsBot {}".format(b.name))
            bot = _TorcsBot(b.name, b.idx, self)
            self._bots.append(bot)
            if b.name.startswith('scr_server'):
                scr_count += 1

        if scr_count > 1:
            raise Exception("now only support 1 scr_server bot")

        logger.debug("got bots = {}".format(bots))
        resetParams = Race.ResetParam()
        resetParams.raceInit = [bot.getRaceInitParam() for bot in self._bots]
        result = self._raceServerPrx.reset(resetParams)
        logger.info("[{:04d}]: episode {} start, track={}, bots={}".format(self._agentIdent, self._episodeCount, self._cur_track, [b._name for b in self._bots]))
        return self._parseStatus(result.statusList)

    def _parseStatus(self, status):
        assert (len(status) == len(self._bots)), len(status)
        for s in status:
            assert(s.ident > 0 and s.ident <= len(self._bots)), s.ident
            bot = self._bots[s.ident-1]
            bot.parseStatus(s)

        if len(self._bots) > 1:
            return np.concatenate([np.expand_dims(bot._ob_status, 0) for bot in self._bots], axis=0)
        return self._bots[0]._ob_status

    def sample_action(self):
        for bot in self._bots:
            if not bot._isHookBot:
                return bot.sample_action()
        return self._bots[0].sample_action()

    def _ouProcess(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * self._rng.randn(1)

    def _step(self, predict):
        action, value, _, _ = predict
        assert (len(action.shape) == 1 and action.shape[0] == 2), action.shape

        if self._isTrain and self._explore > 0:
            if self._exploreStep == self._exploreStepMax:
                if self._rng.rand() < self._explore:
                    self._exploreStepMax = self._rng.randint(1, 5)
                    self._exploreStep = 0
            else:
                self._exploreStep += 1

        for bot in self._bots:
            bot.parsePredict(predict)
        stepParam = Race.StepParam()
        stepParam.driveInfos = [bot._cur_driveInfo for bot in self._bots if not bot._isHookBot]

        def return_end_state():
            return np.zeros_like(self._last_result[0]), np.zeros_like(self._last_result[1]), np.zeros_like(self._last_result[2]), True
        try:
            result = self._raceServerPrx.step(stepParam)
            if len(result.statusList) == 0:
                return return_end_state()

            obs = self._parseStatus(result.statusList)
            if len(self._bots) > 1:
                action = np.concatenate([np.expand_dims(bot._ob_ctrl, 0) for bot in self._bots], axis=0)
                reward = np.array([bot._reward for bot in self._bots], dtype=np.float32)
                isOver = np.array([bot._isOver for bot in self._bots], dtype=np.float32)
            else:
                bot = self._bots[0]
                action = bot._ob_ctrl
                reward = bot._reward
                isOver = bot._isOver

            if self._maxEpisodeSteps > 0 and self._episodeSteps >= self._maxEpisodeSteps:
                logger.info("reach max episode step {}".format(self._maxEpisodeSteps))
                isOver = True

            if result.isAllFinished:
                logger.info("torcs will exit")
                isOver = True
                self._raceServerPrx = None
            self._last_result = (obs, action, reward)
            return obs, action, reward, isOver
        except (Ice.NoValueFactoryException, Ice.ConnectionLostException, Ice.ConnectionRefusedException):
            self._blockResetCount = 0
            return_end_state()
        except Ice.Exception as e:
            logger.exception(e)
            return return_end_state()
        return return_end_state()


    def finishEpisode(self):
        import datetime as dt
        timeEnd = dt.datetime.now()
        use_time = (timeEnd - self._timeStart).total_seconds()
        ret = self._finishEpisode()
        logger.info("[{:04d}]: finish episode: track={}, rewards={:.2f}, steps={}, time={:.2f}s[{:.2f}ms/perstep], episode={}"
                    .format(self._agentIdent, self._cur_track, self._episodeRewards, self._episodeSteps,
                            use_time, use_time * 1000.0 / self._episodeSteps,
                            self._episodeCount,
                            ))
        return ret
