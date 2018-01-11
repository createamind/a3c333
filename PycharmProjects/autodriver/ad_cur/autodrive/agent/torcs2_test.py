#coding: utf-8
import numpy as np

from ad_cur.autodrive.utils import logger
from .torcs2 import AgentTorcs2

def test_torcs():
    import os
    os.system('killall -9 torcs-bin; sleep 0.5 > /dev/null 2>&1')
    agentCount = 1      # 设置agentCount的数量，此处只开一个，如果需要多个平行，请设置相应的数量
    agents = []
    for aidx in range(agentCount):
        # 创建Agent
        agent = AgentTorcs2(aidx,
                            bots=['random'], # 自动驾驶选取的bot，目前bot支持【'berniw', 'berniw2', 'berniw3', 'bt', 'damned', 'inferno', 'inferno2', 'lliaw', 'olethros', 'tita'】，如果设置为random，会随机从列表中取一个
                            track='road/g-track-1', # 选取赛道，格式为"type/name", 目前支持以下类型：
                            # 'road': ['aalborg', 'alpine-1', 'alpine-2', 'brondehach', 'corkscrew', 'eroad', 'e-track-1', 'e-track-2', 'e-track-3', 'e-track-4', 'e-track-6', 'forza', 'g-track-1', 'g-track-2', 'g-track-3', 'ole-road-1', 'ruudskogen', 'spring', 'street-1', 'wheel-1', 'wheel-2'],
                            #  'oval': ['%s-speedway' % i for i in 'abcdefg'] + ['e-track-5', 'michigan'],
                            text_mode=False, # 是否需要显示画面，如果需要抓取画面，必须要设置text_mode=True
                            laps=1, # 运行多少个laps以后结束
                            torcsIdxOffset=0,  # 如需多个torcs实例运行，需要设置其torcsIdxOffset不同，不能重复
                            screen_capture = False,  # 是否需要捕捉视频画面，如果需要抓取画面，必须要设置text_mode=True
                            timeScale = 1., # torcs的运行速度，越小越快
                            )
        observation = agent.reset()   # reset Torcs，创建以后需要调用一次reset, 返回第一个观察到的state
        agents.append(agent)
    logger.info("start running")
    from drlutils.utils.imageview import SimpleImageViewer
    viewer = SimpleImageViewer()
    stepCount = 0
    while True:
        for aidx, agent in enumerate(agents):
            act = agent.sample_action()
            rng = agent._rng
            if rng.rand() < 0.8:
                act[0] = rng.rand() - 0.5
            # agent的接口模仿gym，此处传入的参数为：
            # policy：(np.array([steer, accel]) 传入方向盘角度(-1. <-> 1.)和油门（0. 1.)
            result = agent.step({'policy': act, 'value': 0., 'mus': [0., 0.], 'sigmas': [0., 0.]})
            # 返回一个StepResult（请看race.ice定义）
            # 包含每辆车（现在支持一辆车/实例）的当前状态Status（race.ice)，DriveInfo（bot的驾驶信息，如果存在）
            # 如需显示图像，请取消以下注释
            # if stepCount % 100 == 0:
            #     viewer.imshow(agent._cur_screen)
            from time import sleep
            stepCount += 1
            if isinstance(result.isOver, np.ndarray): isOver = np.any(result.isOver)
            if result.isOver: # 已经跑完所有laps
                # logger.info("[{}]: reset".format(aidx))
                agent.finishEpisode()  #这个函数里面做信息统计
                observation = agent.reset() # 重置agent
