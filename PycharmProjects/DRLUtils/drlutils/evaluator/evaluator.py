#coding: utf-8
import numpy as np
import pandas as pd
import abc
from tensorpack.callbacks.base import Callback
from tensorpack.utils.concurrency import LoopThread
from ..utils.logger import logger
import tensorflow as tf
class EvaluatorBase(Callback):
    def __init__(self, data_io, name, is_training = False, idx = 0, batch_size = -1,
                 sync_step = -1, # sync only in every epoch
                 **kwargs):
        if not name.startswith('evaluate/'): name = 'evaluate/' + name
        self._name = name
        self._idx = idx
        from ..dataflow.tensor_io import TensorIO_AgentPools
        assert(isinstance(data_io, TensorIO_AgentPools))
        self._data_io = data_io         #type: TensorIO_AgentPools
        self._is_training = is_training

        assert(batch_size > 0)
        logger.info("Eval: {} create, batchSize={}, is_train={}, sync_step={}".format(name, batch_size, is_training, sync_step))
        assert(batch_size > 0)
        self._batch_size = batch_size
        self._kwargs = kwargs
        self._tensor_io = None
        self._pool_name = name.replace('/', '_')
        self._sync_step = sync_step

    @property
    def name(self): return self._name

    @property
    def batch_size(self): return self._batch_size

    def _setup_graph(self):
        model = self.trainer.model

        weights_train = model.getWeightsTrain()
        self._op_sync_weights = tf.group(*[d.assign(s) for d, s in zip(self._weights, weights_train)])
        self._callbacks = []
        from tensorpack.callbacks.summary import MergeAllSummaries_RunWithOp, MovingAverageSummary
        c_vars = tf.get_collection(self._name + '-ema_op')
        if len(c_vars) > 0:
            self._callbacks.append(MovingAverageSummary(self._name + '-ema_op'))
        self._callbacks.append(MergeAllSummaries_RunWithOp(0, self._name))
        for c in self._callbacks:
            c.setup_graph(self.trainer)

        self._tensor_io._setup_graph()
        pass

    def getTensorIO(self, input_desc, **kwargs):
        if not self._tensor_io:
            self.get_input_tensor(input_desc, **kwargs)
        return self._tensor_io

    def get_input_tensor(self, input_desc, **kwargs):
        if self._tensor_io: return self._tensor_io.getInputTensors()

        self._input_desc = input_desc
        kwargs = kwargs.copy()
        kwargs.update(self._kwargs)
        self._tensor_io = self._data_io.getTensorIO(self._pool_name + '/pred', input_desc, queue_size = 0, is_training = self._is_training, allow_no_full_batch = True, **kwargs)
        # self._tensor_io = TensorIO_AgentPool(self._name, self._datasets, input_desc, self._batch_size, queue_size = 0, is_training = self._is_training, **kwargs)
        return self._tensor_io.getInputTensors()

    def set_output_tensor(self, *outputs):
        assert(self._tensor_io is not None)
        self._tensor_io.setOutputTensors(*outputs)

    def set_weights(self, weights):
        self._weights = weights

    def _before_train(self):
        model = self.trainer.model
        from tensorpack.callbacks.hooks import CallbackToHook
        from tensorpack.train.base import ReuseSessionCreator
        if len(self._callbacks) > 0:
            self._sess = tf.train.MonitoredSession(session_creator=ReuseSessionCreator(self.trainer.sess),
                                                   hooks=[CallbackToHook(cb) for cb in self._callbacks])
        else:
            self._sess = self.trainer.sess
        self.trainer.sess.run(self._op_sync_weights)
        self._thread = LoopThread(self._run_loop)
        self._thread.start()
        # self._tensor_io._before_train()

    def _trigger(self):
        self.trainer.sess.run(self._op_sync_weights)
        pass

    def _before_run(self, ctx):
        if self._sync_step > 0 and self.local_step % self._sync_step == 0:
            return [self._op_sync_weights]
        return None

    def _after_train(self):
        if self._tensor_io:
            self._tensor_io.close()

    def _run_loop(self):
        if self._sess is None: return
        logger.info("Evaluator {} thread start, fetch tensors = {}, batch = {}".format(self._name, len(self._tensor_io._output_tensors), self._batch_size))
        hooked_sess = self.trainer.hooked_sess
        sess = self._sess
        try:
            tensor_io = self._tensor_io
            while not hooked_sess.should_stop():
                tensor_io._loopStep(sess)
                # logger.info("evaluator loop")
        except (tf.errors.CancelledError, tf.errors.OutOfRangeError):
            pass
        except Exception as e:
            logger.exception("Exception in Evaluator Thread: {}".format(e))
        finally:
            try:
                self._tensor_io.close()
            except Exception:
                pass
            logger.info("Evaluator {} Thread Exited.".format(self._name))
            self._sess = None
        pass

class Evaluator(EvaluatorBase):
    pass

