#coding: utf-8
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorpack.train.multigpu import MultiGPUTrainerBase, SyncMultiGPUTrainerParameterServer, apply_prefetch_policy, LeastLoadedDeviceSetter
from tensorpack.train.feedfree import FeedfreeTrainerBase
from tensorpack.tfutils.tower import get_current_tower_context, TowerContext, _CurrentTowerContext
from tensorpack.train.input_source import StagingInputWrapper, QueueInput
from tensorpack.train.multigpu import _check_tf_version
from tensorpack.utils.concurrency import LoopThread
from ..utils.logger import logger
import itertools

class _AuxTrainOp(object):
    def __init__(self, name):
        self._name = name
        self._train_ops = []
        self._callbacks = []


class MultiGPUTrainer(SyncMultiGPUTrainerParameterServer):
    def __init__(self, config, ps_device='gpu'):
        """
                Args:
                    config: same as in :class:`QueueInputTrainer`.
                    ps_device: either 'gpu' or 'cpu', where variables are stored.
                """
        apply_prefetch_policy(config, False)
        self._input_source = config.data

        assert ps_device in ['gpu', 'cpu'], ps_device
        self._ps_device = ps_device
        super(SyncMultiGPUTrainerParameterServer, self).__init__(config)

    def _setup(self):
        from tensorpack.tfutils import symbolic_functions
        self._v_epoch_num = symbolic_functions.get_scalar_var('epoch_num', 0, summary=True)

        import multiprocessing as mp
        self._epoch_shared = mp.Value('i', 0)

        super(SyncMultiGPUTrainerParameterServer, self)._setup()

        raw_devices = ['/device:GPU:{}'.format(k) for k in self.config.tower]
        # raw_devices = ['/gpu:{}'.format(k) for k in self.config.tower]
        if self._ps_device == 'gpu':
            devices = [LeastLoadedDeviceSetter(d, raw_devices) for d in raw_devices]
        else:
            devices = [tf.train.replica_device_setter(
                worker_device=d, ps_device='/cpu:0', ps_tasks=1) for d in raw_devices]

        from ..model.base import ModelBase
        model = self.model      # type: ModelBase
        assert(isinstance(model, ModelBase))

        logger.info("Building graph ...")
        model.build_graph(None)

        from tensorpack.callbacks.summary import MergeAllSummaries_RunWithOp, MovingAverageSummary
        train_ops_main = []
        train_ops_aux = {}
        for lname, loss in model._losses.items():
            logger.info("Building opt for {} loss {} ...".format('main' if loss._isMainLoss else 'aux ', lname))
            opt = model.get_optimizer() if loss._opt is None else loss._opt
            grads_array = []
            for l in loss._losses:
                grads = opt.compute_gradients(
                    l,
                    gate_gradients=tf.train.Optimizer.GATE_NONE,
                    colocate_gradients_with_ops=True)
                grads = [(g, v) for g, v in grads if g is not None]
                grads_array.append(grads)
            grads = self._average_grads(grads_array)
            train_op = opt.apply_gradients(grads)
            summary_callbacks = []
            if isinstance(loss._summary_collection, str):
                c_vars = tf.get_collection(loss._summary_collection + '-ema_op')
                if len(c_vars) > 0:
                    summary_callbacks.append(MovingAverageSummary(loss._summary_collection + '-ema_op'))
                summary_callbacks.append(MergeAllSummaries_RunWithOp(0, loss._summary_collection))
            if loss._isMainLoss:
                train_ops_main.append(train_op)
                for c in summary_callbacks:
                    self.register_callback(c)
                if loss._tensor_io:
                    loss._tensor_io._is_main = True
                    self.register_callback(loss._tensor_io)
            elif loss._trainOpGroup is not None:
                if loss._trainOpGroup not in train_ops_aux:
                    train_ops_aux[loss._trainOpGroup] = _AuxTrainOp(loss._trainOpGroup)
                auxTrainOp = train_ops_aux[loss._trainOpGroup]
                auxTrainOp._train_ops.append(train_op)
                auxTrainOp._callbacks += summary_callbacks
                if loss._tensor_io: auxTrainOp._callbacks.append(loss._tensor_io)
            else:
                auxTrainOp = _AuxTrainOp(lname)
                auxTrainOp._train_ops = [train_op]
                auxTrainOp._callbacks += summary_callbacks
                if loss._tensor_io: auxTrainOp._callbacks.append(loss._tensor_io)
                train_ops_aux[lname] = (auxTrainOp)

        for n, auxTrainOp in train_ops_aux.items():
            assert(len(auxTrainOp._train_ops) > 0)
            auxTrainOp._train_op = tf.group(*auxTrainOp._train_ops, name = n + '/train_op')
            for c in auxTrainOp._callbacks:
                c.setup_graph(self)
        # for rname, rop in model._run_ops.items():
        #     train_ops_aux.append(tf.group(*rop._run_ops, name=rname + '/run-op'))

        var_lists = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        var_lists[:] = [v for v in var_lists if not v.name.startswith('evaluate/')]

        self.train_op = tf.group(*train_ops_main, name='train_op')
        self._train_ops_aux = train_ops_aux


    def _trigger_epoch(self):
        self._epoch_shared.value = self.epoch_num
        self._v_epoch_num.load(self.epoch_num, self.sess)

    def setup(self):
        super(MultiGPUTrainer, self).setup()

        self._epoch_shared.value = self.epoch_num = self.config.starting_epoch - 1
        self._v_epoch_num.load(self.epoch_num, self.sess)

        self._training_aux_threads = []
        self._training_aux_running = False
        self._training_aux_step_counter = itertools.count()
        from tensorpack.callbacks.hooks import CallbackToHook
        from tensorflow.python.training.monitored_session \
            import _HookedSession as HookedSession

        for tidx, n in enumerate(self._train_ops_aux):
            auxTrainOp = self._train_ops_aux[n]
            logger.info("Create aux train ops {}".format(auxTrainOp._name))
            if len(auxTrainOp._callbacks) > 0:
                auxTrainOp._sess = HookedSession(self.sess, hooks=[CallbackToHook(cb) for cb in auxTrainOp._callbacks])
            else:
                auxTrainOp._sess = self.sess

            def f(op=auxTrainOp):  # avoid late-binding
                try:
                    op._sess.run([op._train_op])  # TODO this won't work with StageInput
                except RuntimeError: # exited
                    pass
                except tf.errors.CancelledError:
                    pass
                # next(self._training_aux_step_counter)   # atomic due to GIL

            th = LoopThread(f)
            th.name = "AsyncLoopThread-{}".format(tidx)
            th.pause()
            th.start()
            self._training_aux_threads.append(th)



    def main_loop(self):
        """
        Run the main training loop.
        """
        from tensorpack.tfutils.common import get_global_step_value
        from tensorpack.train.base import StopTraining
        import time
        with self.sess.as_default():
            self._starting_step = get_global_step_value()
            try:
                self._callbacks.before_train()
                for self.epoch_num in range(
                        self.config.starting_epoch, self.config.max_epoch + 1):
                    logger.info("Start Epoch {} ...".format(self.epoch_num))
                    start_time = time.time()
                    for self.local_step in range(self.config.steps_per_epoch):
                        if self._monitored_sess.should_stop():
                            return
                        self.run_step()  # implemented by subclass
                        self._callbacks.trigger_step()
                    logger.info("Epoch {} (global_step {}) finished, time:{:.2f} sec.".format(
                        self.epoch_num, self.global_step, time.time() - start_time))

                    # trigger epoch outside the timing region.
                    self._trigger_epoch()
                    self._callbacks.trigger_epoch()
            except (StopTraining, tf.errors.OutOfRangeError):
                logger.info("Training was stopped.")
            except KeyboardInterrupt:
                logger.info("Detected Ctrl-C and exiting main loop.")
            except:
                raise
            finally:
                self._callbacks.after_train()
                self._monitored_sess.close()

    def run_step(self):
        if not self._training_aux_running:
            self._training_aux_running = True
            for th in self._training_aux_threads:  # resume all threads
                th.resume()
        next(self._training_aux_step_counter)
        return self.hooked_sess.run(self.train_op)
        # return super(MultiGPUTrainer, self).run_step()

