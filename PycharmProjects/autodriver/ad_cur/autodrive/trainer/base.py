#coding: utf-8
import numpy as np
import pandas as pd

from tensorpack.train.multigpu import AsyncMultiGPUTrainer

from tensorpack.train.multigpu import AsyncMultiGPUTrainer, LeastLoadedDeviceSetter, MultiGPUTrainerBase, ScaleGradient
import tensorflow as tf

class MyMultiGPUTrainer(AsyncMultiGPUTrainer):
    def _setup(self):
        super(AsyncMultiGPUTrainer, self)._setup()
        raw_devices = ['/gpu:{}'.format(k) for k in self.config.tower]
        devices = [LeastLoadedDeviceSetter(d, raw_devices) for d in raw_devices]
        grad_list = MultiGPUTrainerBase.build_on_multi_tower(
            self.config.tower,
            lambda: MultiGPUTrainerBase._build_graph_get_grads(
                self.model, self._input_source), devices)
        MultiGPUTrainerBase._check_grad_list(grad_list)

        if self._scale_gradient and self.config.nr_tower > 1:
            # pretend to average the grads, in order to make async and
            # sync have consistent effective learning rate
            gradproc = ScaleGradient(('.*', 1.0 / self.config.nr_tower), verbose=False)
            grad_list = [gradproc.process(gv) for gv in grad_list]
        # Ngpu x Nvar x 2

        train_ops = []
        opts = self.model.get_optimizer()
        if type(opts) not in [list, tuple]:
            opts = [opts] * len(grad_list[0])
        for i, grad_and_vars_group in enumerate(grad_list):
            # Ngpu x 2
            assert(len(grad_and_vars_group) == len(opts))
            v = grad_and_vars_group[0][0][1]
            with tf.device(v.device):
                # will call apply_gradients (therefore gradproc) multiple times
                for opt, grad_and_vars in zip(opts, grad_and_vars_group):
                    train_ops.append(opt.apply_gradients(
                        grad_and_vars, name='apply_grad_{}'.format(i)))
        self.train_op = tf.group(*train_ops, name='train_op')
