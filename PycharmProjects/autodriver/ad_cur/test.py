#coding: utf-8
import numpy as np
from ad_cur.autodrive.utils import logger
from ad_cur.autodrive.agent.torcs2 import AgentTorcs2
import distutils.spawn
import distutils.version
import os
import os.path as osp
import subprocess
import h5py
import cv2
from datetime import datetime
import time


logger.info("start running")


def write_h5(filename, x, rewards, actions):
    h5py_writer = h5py.File(filename, 'w')
    h5py_writer.create_dataset("X", data = x)
    h5py_writer.create_dataset("speed", data = rewards)
    h5py_writer.create_dataset("steering_angle", data = actions[:, 0])
    h5py_writer.close()

from drlutils.utils.imageview import SimpleImageViewer
def check_path(p):
    if not osp.exists(p):
        os.mkdir(p)

def test_torcs():
    import os
    os.system('killall -9 torcs-bin; sleep 0.5')
    # agentCount = 3
    agents = []
    tracks = ['aalborg', 'alpine-1', 'alpine-2', 'brondehach', 'corkscrew', 'eroad', 'e-track-1', 'e-track-2', 'e-track-3','e-track-4', 'e-track-6', 'forza','g-track-1', 'g-track-2', 'g-track-3', 'ole-road-1', 'ruudskogen', 'spring', 'street-1', 'wheel-1', 'wheel-2']
    #tracks = ['aalborg', 'alpine-1', 'alpine-2', 'brondehach', 'corkscrew', 'eroad', 'e-track-1', 'e-track-2', 'e-track-3', 'e-track-4', 'e-track-6', 'forza','g-track-1', 'g-track-2', 'g-track-3', 'ole-road-1', 'ruudskogen', 'spring', 'street-1', 'wheel-1', 'wheel-2']

    #tracks = ['eroad'] #'alpine-2'
    agentCount = len(tracks)
    dir = './tmp/data/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    #os.makedirs(dir)
    size = (256,256)

    for aidx in range(agentCount):
        dumppath = dir+tracks[aidx]
        check_path(dumppath)
        os.system('killall -9 torcs-bin; sleep 0.5')
        try :
            agent = AgentTorcs2(aidx, bots=['berniw'], track='road/{}'.format(tracks[aidx]), text_mode=False, laps=30, torcsIdxOffset=0, screen_capture = True, timeScale = 1.45)
        except :
            print("load  {}  error!!!".format(tracks[aidx]))
            continue
        # agent = AgentTorcs2(aidx, bots=['berniw'], track='road/g-track-1', text_mode=False, laps=3, torcsIdxOffset=0, screen_capture = False, timeScale = 0.25)
        # agent = AgentTorcs2(aidx, bots=['scr_server', 'olethros', 'berniw', 'bt', 'damned'], track='road/g-track-1', text_mode=True)

        agents.append(agent)
        logger.info("start running")






        stepCount = 0
        # frame = agent._cur_screen
        shape = [160,320,3]
        # encoder = ImageEncoder('video1.mp4', shape, 24)
        actions= []
        rewards = []
        frames= []
        viewer = SimpleImageViewer()
        agent.reset()



        encoder = ImageEncoder('{}.mp4'.format(tracks[aidx]), shape, 24)
        for i in range(3300):
            rng = agent._rng
            act = agent.sample_action()
            print("-----------------")
            print(act)
            if rng.rand() < 0.8:
                act[0] = rng.rand() - 0.5
            ob, action, reward, isOver = agent.step((act, 0., [0., 0.], [0., 0.]))
            angle = ob[5]
            speedx = ob[26]
            speedy = ob[27]
            print("angle : {} \n speedx : {} \n speedy : {} ".format(angle , speedx ,speedy))
            print([i.shape for i in [ob, action, reward] ])
            ret = np.hstack([ob, action, reward])
            print(act)
            print(action)
            #print("ret ",ret.shape,ret)
            print("sleeep Start")
            print(datetime.now().time())

            #time.sleep(0.02)




            frame = agent._cur_screen
            #print(frame.shape) # 576*720
            frame = frame[:, int(360-576/2):int(360+576/2), :]
            #frame = frame[:,40:280,:] #for 160*160
            #print(frame.shape)
            # frame = frame[40:200,:,:]
            #1024 * 1024
            #frmae = frame[:1024,:,:]

            #reverse
            frame = frame[::-1,:,:]
            frame = cv2.resize(frame,size)
            frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

            #print(frame.shape)
            if i >800 : #and i mod 5 == 0 :
                actions.append(action)

                rewards.append(reward)
                name=i

                cv2.imwrite('{}/{}.jpg'.format(dumppath,name),frame)
                ret.dump('{}/{}.npy'.format(dumppath,name))

                #encoder.capture_frame(frame)
                frame_t = np.transpose(frame ,(2,0,1) )
                frames.append(frame_t)
            if stepCount % 100 == 0:
                 viewer.imshow(frame)
            from time import sleep
            stepCount += 1
            # logger.info('action : {} ,state {} '.format(action, ob))
            if isinstance(isOver, np.ndarray): isOver = np.any(isOver)
            if isOver:
                # logger.info("[{}]: reset".format(aidx))
                agent.finishEpisode()
                agent.reset()
        encoder.close()
        frames = np.asarray(frames)
        actions = np.asarray((actions))
        rewards = np.asarray((rewards))

        #write_h5('{}.{}.h5'.format(tracks[aidx],datetime.now().time()), frames, rewards, actions)

        print('action shape {} ; rewards shape {} '.format(actions.shape, rewards.shape))






class ImageEncoder(object):
    def __init__(self, output_path, frame_shape, frames_per_sec):
        self.proc = None
        self.output_path = output_path
        # Frame shape should be lines-first, so w and h are swapped
        h, w, pixfmt = frame_shape
        if pixfmt != 3 and pixfmt != 4:
            raise Exception(
                "Your frame has shape {}, but we require (w,h,3) or (w,h,4), i.e. RGB values for a w-by-h image, with an optional alpha channl.".format(
                    frame_shape))
        self.wh = (w, h)
        self.includes_alpha = (pixfmt == 4)
        self.frame_shape = frame_shape
        self.frames_per_sec = frames_per_sec

        if distutils.spawn.find_executable('avconv') is not None:
            self.backend = 'avconv'
        elif distutils.spawn.find_executable('ffmpeg') is not None:
            self.backend = 'ffmpeg'
        else:
            raise Exception(
                """Found neither the ffmpeg nor avconv executables. On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`.""")

        self.start()

    @property
    def version_info(self):
        return {
            'backend': self.backend,
            'version': str(subprocess.check_output(
                [self.backend, '-version'],
                stderr=subprocess.STDOUT)),
            'cmdline': self.cmdline
        }

    def start(self):
        self.cmdline = (
            self.backend,
            '-nostats',
            '-loglevel', 'error',  # suppress warnings
            '-y',
            '-r', '%d' % self.frames_per_sec,
            '-f', 'rawvideo',  # input
            '-s:v', '{}x{}'.format(*self.wh),
            '-pix_fmt', ('rgb32' if self.includes_alpha else 'rgb24'),
            '-i', '-',  # this used to be /dev/stdin, which is not Windows-friendly
            '-vf', 'vflip',
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            # '-threads',6,
            self.output_path
        )

        if hasattr(os, 'setsid'):  # setsid not present on Windows
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid)
        else:
            self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)

    def capture_frame(self, frame):
        if distutils.version.LooseVersion(np.__version__) >= distutils.version.LooseVersion('1.9.0'):
            self.proc.stdin.write(frame.tobytes())
        else:
            self.proc.stdin.write(frame.tostring())

    def close(self):
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            raise Exception("VideoRecorder encoder exited with status {}".format(ret))


test_torcs()