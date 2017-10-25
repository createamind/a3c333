#coding: utf-8

import numpy as np
from ..utils import logger
from .base import AgentBase
import Ice

class AgentSingleLoop(AgentBase):
    def _init(self):
        super(AgentSingleLoop, self)._init()
        self._dataioClient = None
        dio_addr = self._kwargs.get('dio_addr', None)
        self._dio_pred = self._dio_train = None
        if dio_addr:
            from ..rpcio.pyclient import DataIOClient
            dio_name = self._kwargs.get('dio_name')
            self._dio_pred = DataIOClient(dio_name + '/pred', self._agentIdent, dio_addr[0], dio_addr[1])
            if self._isTrain:
                self._dio_train = DataIOClient(dio_name + '/train', self._agentIdent, dio_addr[0], dio_addr[1])

    def run(self):
        from ..dataflow.load_slice import DataFlow
        dio_pred = self._dio_pred
        if dio_pred:
            dio_pred.waitForConnected()
        if self._isTrain:
            dio_train = self._dio_train
            dio_train.waitForConnected()

            logger.info("[{}]: dio connected, start to run".format(self._agentIdent))
        gamma = self._kwargs.get("gamma", 0.99)
        local_t_max = self._kwargs.get("local_t_max", 5)
        memory = []
        state = self.reset()

        pred_seqlen = np.array([1], dtype=np.int32)
        pred_resetRNN = np.zeros([1], dtype=np.int32)
        _agentIdent = np.array([self._agentIdent], dtype=np.int32)
        is_over = True
        while True:
            try:
                if len(state.shape) < 3:
                    state = state[np.newaxis, np.newaxis]
                # logger.info("[{}]: state = {}".format(self._agentIdent, state.shape))
                if is_over: pred_resetRNN[0] = 1
                dio_pred.putData(_agentIdent, state, pred_seqlen, pred_resetRNN)
                pred_resetRNN[0] = 0
                pred = dio_pred.getData()
                # logger.info("[{}]: got data: action={}, value={}".format(self._agentIdent, action, value))
                action = pred[0][0]
                value = pred[1][0]
                state, action, reward, is_over = self.step([p[0] for p in pred])
                # logger.info("state = {}, actoin={}, reward={}, is_over={}".format(state.shape, action.shape, reward.shape, is_over))

                if self._isTrain:
                    memory.append((state, action, reward, is_over, value))
                    if len(memory) > local_t_max or is_over:
                        if not is_over:
                            last = memory[-1]
                            mem = memory[:-1]
                            init_r = last[-1]
                        else:
                            init_r = 0.
                            mem = memory

                        def discount(x, gamma):
                            from scipy.signal import lfilter
                            return lfilter(
                                [1], [1, -gamma], x[::-1], axis=0)[::-1]

                        rewards_plus = np.asarray([m[2] for m in mem] + [float(init_r)])
                        discounted_rewards = discount(rewards_plus, gamma)[:-1]
                        values_plus = np.asarray([m[4] for m in mem] + [float(init_r)])
                        rewards = np.asarray([m[2] for m in mem]).astype(np.float32)
                        advantages = (rewards + gamma * values_plus[1:] - values_plus[:-1]).astype(np.float32)

                        states = np.concatenate([m[0][np.newaxis, np.newaxis] for m in mem], axis=1).astype(np.float32)
                        actions = np.concatenate([m[1][np.newaxis, np.newaxis] for m in mem], axis=1).astype(np.float32)
                        seqLength = np.array([states.shape[1]], dtype=np.int32)
                        isOver = np.array([is_over], dtype=np.int32)
                        # logger.info("rewards = {}, advantage = {}".format(rewards.shape, advantages.shape))
                        dio_train.putData(_agentIdent, states, actions,
                                          discounted_rewards[np.newaxis],
                                          advantages[np.newaxis],
                                          seqLength, isOver)
                        if not is_over:
                            memory = [last]
                        else:
                            memory = []
                if is_over:
                    self.finishEpisode()
                    state = self.reset()
            except DataFlow.ExceptionClosed as e:
                break
            except Ice.ConnectionRefusedException as e:
                break
            except Ice.ConnectionLostException as e:
                break
            except Ice.Exception as e:
                raise e
