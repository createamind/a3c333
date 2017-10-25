#coding: utf-8
import numpy as np


from ad_cur.autodrive.agent.torcs2 import AgentTorcs2

from ad_cur.autodrive.utils import logger
def test_torcs():
    import os
    os.system('killall -9 torcs-bin; sleep 0.5')
    agentCount = 1
    agents = []
    for aidx in range(agentCount):
        agent = AgentTorcs2(aidx, bots=['scr_server'], track='road/g-track-1', text_mode=False, laps=3, torcsIdxOffset=0, screen_capture = False, timeScale = 0.25)
        # agent = AgentTorcs2(aidx, bots=['scr_server', 'olethros', 'berniw', 'bt', 'damned'], track='road/g-track-1', text_mode=True)
        agent.reset()
        agents.append(agent)
    logger.info("start running")
    from drlutils.utils.imageview import SimpleImageViewer
    viewer = SimpleImageViewer()
    stepCount = 0
    while True:
        for aidx, agent in enumerate(agents):
            rng = agent._rng
            act = agent.sample_action()
            if rng.rand() < 0.8:
                act[0] = rng.rand() - 0.5
            ob, action, reward, isOver = agent.step((act, 0., [0., 0.], [0., 0.]))
            # if stepCount % 100 == 0:
            #     viewer.imshow(agent._cur_screen)
            from time import sleep
            stepCount += 1
            if isinstance(isOver, np.ndarray): isOver = np.any(isOver)
            if isOver:
                # logger.info("[{}]: reset".format(aidx))
                agent.finishEpisode()
                agent.reset()
