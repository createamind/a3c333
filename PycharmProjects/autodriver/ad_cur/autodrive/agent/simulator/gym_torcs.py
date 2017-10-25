import gym
from gym import spaces
# from os import path
from .snakeoil3_gym import Client
import numpy as np
import copy
import collections as col
import os
import subprocess
import time
import signal

from ...utils import logger

class TorcsEnv:
    terminal_judge_start = 500  # Speed limit is applied after this step
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = True
    @staticmethod
    def checkEnvExist(agentIdent):
        import os
        port = TorcsEnv._getPort(agentIdent)
        return os.system('xdotool search --name "{}" 2>&1 >/dev/null'.format(port)) == 0

    @staticmethod
    def _getPort(agentIdent):
        import os
        port = os.environ.get("TORCS_CLIENT_PORT_BASE", 31000) + agentIdent
        return port

    def _set_track(self):
        if self.track_name is '':
            import random
            if True: #self.training:
                t_name = random.choice(
                    ['g-track-1', 'g-track-2', 'ruudskogen', 'forza',
                     'ole-road-1', 'street-1'])
            else:
                t_name = random.choice(
                    ['g-track-3', 'e-track-6', 'alpine-2'])
        else:
            t_name = self.track_name

        import os, sys
        os.system('{} {}/set_track.py -t {}'.format(sys.executable, os.path.dirname(__file__), t_name))

    def start_torcs_process(self, force = False):
        if (not force) and TorcsEnv.checkEnvExist(self._ident): return
        import os
        logdir = os.path.expanduser('~/tmp/torcs_run')
        if not os.path.exists(logdir): os.makedirs(logdir)
        pidfile = logdir + '/{:03d}.pid'.format(self._ident)
        pid = -1
        if self.torcs_proc is not None:
            pid = self.torcs_proc.pid
        elif os.path.exists(pidfile):
            with open(pidfile, 'r') as f:
                pid = int(f.readline()[:-1])
        if pid >= 0:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                pass
            time.sleep(0.5)
            logger.info("torcs {} is killed".format(pid))
            self.torcs_proc = None
        window_title = str(self.port)
        torcs_dir = os.path.expanduser('~/torcs')
        if not os.path.exists(torcs_dir):
            raise IOError("can not found torcs dir, please install torcs into ${HOME}/torcs")
        # self._set_track()

        import filelock
        lockfile = logdir + '/.torcs_autorestart'
        try:
            lock = filelock.FileLock(lockfile)
            with lock.acquire(30):
                torcs_config_dir = os.path.expanduser('~/.torcs_{:03d}'.format(self._ident))
                command = 'export LOCAL_CONF_DIR={};{}/bin/torcs -nofuel -nodamage -nolaptime -title {} -p {}'\
                    .format(torcs_config_dir, torcs_dir, window_title, self.port)
                if self.vision is True:
                    command += ' -vision'

                if not os.path.exists(logdir): os.makedirs(logdir)
                command += ' 2>&1 > {}/{:03d}.log'.format(logdir, self._ident)
                self.torcs_proc = subprocess.Popen([command], shell=True, preexec_fn=os.setsid)
                for retry in range(10):
                    if self.torcs_proc.pid >= 0:
                        pidfile = logdir + '/{:03d}.pid'.format(self._ident)
                        if os.path.exists(pidfile): os.unlink(pidfile)
                        with open(pidfile, 'w') as f:
                            f.writelines([str(self.torcs_proc.pid)])
                            f.flush()

                time.sleep(1.)
                cmd = 'sh {}/autostart.sh {}'.format(os.path.dirname(__file__), window_title)
                if self.winpos is not None:
                    assert(len(self.winpos) == 2)
                    cmd += ' {} {}'.format(self.winpos[0], self.winpos[1])
                os.system(cmd)
                time.sleep(1.)
        except filelock.Timeout:
            logger.error("can not lock file {}".format(lockfile))

    def __init__(self, ident, vision=False, throttle=False, gear_change=False, winpos=None, track_name = 'g-track-1'):
       #print("Init")
        self._ident = ident
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.port = TorcsEnv._getPort(ident)
        self.torcs_proc = None
        self.track_name = track_name
        self.winpos = winpos
        logger.info("init gym torcs: ident={}, vision={}, throttle={}, gear_change={}".format(ident, vision, throttle, gear_change))

        self.initial_run = True

        ##print("launch torcs")

            # time.sleep(0.5)
        self.start_torcs_process()
            # time.sleep(0.5)

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        """
        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        if vision is False:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
            self.observation_space = spaces.Box(low=low, high=high)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
            self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u):
       #print("Step")
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']
            action_torcs['brake'] = this_action['brake']

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            if self.throttle:
                if client.S.d['speedX'] > 50:
                    action_torcs['gear'] = 2
                if client.S.d['speedX'] > 80:
                    action_torcs['gear'] = 3
                if client.S.d['speedX'] > 110:
                    action_torcs['gear'] = 4
                if client.S.d['speedX'] > 140:
                    action_torcs['gear'] = 5
                if client.S.d['speedX'] > 170:
                    action_torcs['gear'] = 6
        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        trackPos = np.array(obs['trackPos'])
        sp = np.array(obs['speedX'])
        damage = np.array(obs['damage'])
        rpm = np.array(obs['rpm'])

        progress = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle']))  - sp * np.abs(obs['trackPos'])

       # # Reward setting Here
       # # direction-dependent positive reward
       #  progress = (
       #      np.array(obs['speedX']) *
       #      (np.cos(obs['angle']) - np.sin(obs['angle'])))

        reward = progress

        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -1

        # Termination judgement #########################
        episode_terminate = False
        # if (abs(track.any()) > 1 or abs(trackPos) > 1):  # Episode is terminated if the car is out of track
        #    reward = -200
        #    episode_terminate = True
        #    client.R.d['meta'] = True

        #if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
        #    if progress < self.termination_limit_progress:
        #        print("No progress")
        #        episode_terminate = True
        #        client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            episode_terminate = True
            client.R.d['meta'] = True


        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta'], {}

    def reset(self, relaunch=False):
        #print("Reset")

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = Client(self.start_torcs_process, p=self.port)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()

    def end(self):
        os.killpg(os.getpgid(self.torcs_proc.pid), signal.SIGKILL)

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
        print("relaunch torcs")
        time.sleep(0.5)
        self.start_torcs_process(force=True)
        time.sleep(0.5)

    def agent_to_torcs(self, u):
        accel = brake = 0
        if u[1] > 0:
            accel = u[1]
        else:
            brake = -u[1]
        torcs_action = {'steer': u[0], 'accel': accel, 'brake': brake}
        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        rgb = []
        temp = []
        # convert size 64x64x3 = 12288 to 64x64=4096 2-D list 
        # with rgb values grouped together.
        # Format similar to the observation in openai gym
        for i in range(0,12286,3):
            temp.append(image_vec[i])
            temp.append(image_vec[i+1])
            temp.append(image_vec[i+2])
            rgb.append(temp)
            temp = []
        return np.array(rgb, dtype=np.uint8)

    def make_observaton(self, raw_obs):
        if self.vision is False:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle', 'damage',
                     'opponents',
                     'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel']
            Observation = col.namedtuple('Observaion', names)
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/300.0,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/300.0,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/300.0,
                               angle=np.array(raw_obs['angle'], dtype=np.float32)/3.1416,
                               damage=np.array(raw_obs['damage'], dtype=np.float32),
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32)/10000,
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32))
        else:
            assert(0)
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle',
                     'opponents',
                     'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel',
                     'img']
            Observation = col.namedtuple('Observaion', names)

            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]])

            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               trackPos=np.array(raw_obs['trackPos'], dtype=np.float32)/1.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb)
