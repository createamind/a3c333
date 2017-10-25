#coding: utf-8
import numpy as np
from tensorpack.graph_builder.model_desc import ModelDesc, InputDesc
from ..utils.logger import logger
import tensorflow as tf
import abc
_current_nn_context = None      # type: NNContext

def get_current_nn_context():
    ''':rtype:NNContext'''
    return _current_nn_context

class NNContext(object):
    def __init__(self, name, device = None, add_variable_scope = False, summary_collection = None, is_training = False, is_evaluating = False, reuse=None):
        self._name = name
        self._add_variable_scope = add_variable_scope
        if summary_collection is True:
            self._summary_collection = name
        elif summary_collection is False:
            self._summary_collection = False
        else:
            self._summary_collection = summary_collection
        self._device = device
        self._is_training = is_training
        self._is_evaluating = is_evaluating
        self._reuse = reuse

    def __enter__(self):
        global _current_nn_context
        assert(_current_nn_context is None), 'nested nn context'
        _current_nn_context = self
        self._ctxs = []
        if self._device is not None: self._ctxs.append(tf.device(self._device))
        if self._add_variable_scope: self._ctxs.append(tf.variable_scope(self._name))
        for c in self._ctxs:
            c.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _current_nn_context
        _current_nn_context = None
        for c in self._ctxs[::-1]:
            c.__exit__(exc_type, exc_val, exc_tb)

    @property
    def is_training(self): return self._is_training
    @property
    def is_evaluating(self): return self._is_evaluating
    @property
    def reuse(self): return self._reuse
    @property
    def name(self): return self._name

    @property
    def device(self): return self._device

    @property
    def summary_collection(self): return self._summary_collection

class ModelBase(ModelDesc):
    class LossInfo(object):
        def __init__(self, name, opt = None, summaryGrad = True, isMainLoss = True, trainOpGroup = None, tensor_io = None, var_list = None):
            self._name = name
            self._losses = []       # type: list[tf.Tensor]
            self._opt = opt
            self._summaryGrad = summaryGrad
            self._isMainLoss = isMainLoss
            self._summary_collection = None
            self._trainOpGroup = trainOpGroup
            self._var_list = var_list
            # from ..dataflow.tensor_io import TensorIO
            self._tensor_io = tensor_io     # type: TensorIO
            nnc = get_current_nn_context()
            if nnc is not None and nnc._summary_collection is not False:
                self._summary_collection = nnc._summary_collection

    class ThreadOp(object):
        def __init__(self, name):
            self._name = name
            self._ops = []

    class RunOp(object):
        def __init__(self, name):
            self._name = name
            self._run_ops = []

    def __init__(self):
        self._init()
        pass

    def _init(self):
        from collections import OrderedDict
        self._losses = OrderedDict()        # type: dict[str, ModelBase.LossInfo]
        self._run_ops = OrderedDict()       # type: dict[str, ModelBase.RunOp]
        self._thread_ops = OrderedDict()       # type: dict[str, ModelBase.ThreadOp]
        self._weights_train = []
        self._weights_evaluate = []
        self._sync_evaluate_weights_op = None
        pass

    def _addLoss(self, name, loss, opt = None, summaryGrad = True, isMainLoss = True, trainOpGroup = None, tensor_io = None, var_list = None):
        nnc = get_current_nn_context()
        if nnc and not nnc.is_training:
            logger.info("{} is not in trainning, skip loss".format(nnc.name))
            return
        assert(tensor_io is not None)
        if name not in self._losses:
            self._losses[name] = ModelBase.LossInfo(name, opt,
                                                    summaryGrad = summaryGrad, isMainLoss = isMainLoss,
                                                    trainOpGroup = trainOpGroup, tensor_io = tensor_io,
                                                    var_list = var_list,
                                                    )
        self._losses[name]._losses.append(loss)

    def _addRunOp(self, name, op):
        if name not in self._run_ops:
            self._run_ops[name] = ModelBase.RunOp(name)
        self._run_ops[name]._run_ops.append(op)

    def _addThreadOp(self, name, op):
        if name not in self._thread_ops:
            self._thread_ops[name] = ModelBase.ThreadOp(name)
        assert(isinstance(op, tf.Operation))
        self._thread_ops[name]._ops.append(op)

    def _addMovingSummary(self, v, *args, **kwargs):
        """
        Args:
            v (tf.Tensor or list): tensor or list of tensors to summary. Must have
                scalar type.
            args: tensors to summary (support positional arguments)
            decay (float): the decay rate. Defaults to 0.95.
            collection (str): the name of the collection to add EMA-maintaining ops.
                The default will work together with the default
                :class:`MovingAverageSummary` callback.
        """
        from tensorpack.tfutils.summary import add_moving_summary, MOVING_SUMMARY_OPS_KEY
        from tensorpack.tfutils.tower import get_current_tower_context
        from tensorpack.tfutils.common import get_global_step_var
        import re
        import tensorflow as tf
        decay = kwargs.pop('decay', 0.95)
        collection = MOVING_SUMMARY_OPS_KEY
        summary_collection = None
        global _current_nn_context
        if _current_nn_context and _current_nn_context.summary_collection is False:
            return

        if _current_nn_context and _current_nn_context.summary_collection:
            summary_collection = [_current_nn_context.summary_collection]
            collection = _current_nn_context._summary_collection + '-ema_op'
        elif 'collection' in kwargs:
            collection = kwargs.pop('collection')

            assert len(kwargs) == 0, "Unknown arguments: " + str(kwargs)

        if not isinstance(v, list):
            v = [v]
        v.extend(args)
        for x in v:
            assert (isinstance(x, tf.Tensor) or isinstance(x, tf.Variable)), x
            assert x.get_shape().ndims == 0, x.get_shape()
        # TODO will produce tower0/xxx?
        # TODO use zero_debias
        with tf.name_scope(None):
            averager = tf.train.ExponentialMovingAverage(
                decay, num_updates=get_global_step_var(), name='EMA')
            avg_maintain_op = averager.apply(v)

            for c in v:
                # TODO do this in the EMA callback?
                name = re.sub('tower[pe0-9]+/', '', c.op.name)
                tf.summary.scalar(name + '-summary', averager.average(c), collections=summary_collection)

        tf.add_to_collection(collection, avg_maintain_op)
        return averager, avg_maintain_op
    def _addSummary(self, v, *args, **kwargs):
        """
        Args:
            v (tf.Tensor or list): tensor or list of tensors to summary. Must have
                scalar type.
            args: tensors to summary (support positional arguments)
            decay (float): the decay rate. Defaults to 0.95.
            collection (str): the name of the collection to add EMA-maintaining ops.
                The default will work together with the default
                :class:`MovingAverageSummary` callback.
        """
        from tensorpack.tfutils.summary import add_moving_summary, MOVING_SUMMARY_OPS_KEY
        from tensorpack.tfutils.tower import get_current_tower_context
        from tensorpack.tfutils.common import get_global_step_var
        import re
        import tensorflow as tf
        assert len(kwargs) == 0, "Unknown arguments: " + str(kwargs)

        collections = None
        global _current_nn_context
        if _current_nn_context and _current_nn_context.summary_collection is False:
            return

        if _current_nn_context and _current_nn_context.summary_collection:
            collections = [_current_nn_context._summary_collection]
        if not isinstance(v, list):
            v = [v]
        v.extend(args)
        for x in v:
            assert isinstance(x, tf.Tensor), x
            assert x.get_shape().ndims == 0, x.get_shape()
        # TODO will produce tower0/xxx?
        # TODO use zero_debias
        with tf.name_scope(None):
            for c in v:
                # TODO do this in the EMA callback?
                name = re.sub('tower[pe0-9]+/', '', c.op.name)
                tf.summary.scalar(name + '-summary', c, collections=collections)

    def _addParamSummary(self, summary_lists, params = None, regex = None):
        """
            Add summary Ops for all trainable variables matching the regex.

            Args:
                summary_lists (list): each is (regex, [list of summary type to perform]).
                Summary type can be 'mean', 'scalar', 'histogram', 'sparsity', 'rms'
            """
        nnc = get_current_nn_context()

        collections = None
        if nnc and nnc.summary_collection is False:
            return
        elif nnc and len(nnc.summary_collection) > 0:
            collections = [nnc.summary_collection]


        def perform(var, action):
            ndim = var.get_shape().ndims
            name = var.name.replace(':0', '')
            if action == 'scalar':
                assert ndim == 0, "Scalar summary on high-dimension data. Maybe you want 'mean'?"
                tf.summary.scalar(name, var, collections=collections)
                return
            assert ndim > 0, "Cannot perform {} summary on scalar data".format(action)
            if action == 'histogram':
                tf.summary.histogram(name, var, collections=collections)
                return
            if action == 'sparsity':
                tf.summary.scalar(name + '-sparsity', tf.nn.zero_fraction(var), collections=collections)
                return
            if action == 'mean':
                tf.summary.scalar(name + '-mean', tf.reduce_mean(var), collections=collections)
                return
            if action == 'rms':
                from tensorpack.tfutils.symbolic_functions import rms
                tf.summary.scalar(name + '-rms', rms(var), collections=collections)
                return
            if action == 'absmax':
                tf.summary.scalar(name + '-absmax', tf.reduce_max(tf.abs(var)), collections=collections)
                return
            raise RuntimeError("Unknown summary type: {}".format(action))

        if params is None:
            params = tf.trainable_variables()
            if regex:
                import re
                _params = []
                for p in params:
                    if re.match(regex, p.name):
                        _params.append(p)
                params = _params

        # params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        import re
        with tf.name_scope('00/SummaryParam'):
            for p in params:
                name = p.name
                for rgx, actions in summary_lists:
                    if not rgx.endswith('$'):
                        rgx = rgx + '(:0)?$'
                    if re.match(rgx, name):
                        for act in actions:
                            perform(p, act)

    def get_cost(self):
        raise NotImplementedError

    def build_graph(self, model_inputs):
        assert(len(self._weights_train) == 0)
        from tensorpack.tfutils.tower import get_current_tower_context, TowerContext
        self._build_graph()
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        self._weights_train = [w for w in weights if not w.name.startswith('evaluate/')]

    @abc.abstractmethod
    def _build_graph(self):
        raise NotImplementedError

    def getWeightsTrain(self):
        return self._weights_train

    def _buildRNN(self, inputs, cell, batchSize,
                  i_sequenceLength, i_resetRNN,
                  i_agentIdent = None,
                  initial_states = None, name = 'rnn'):
        import tensorflow as tf
        from tensorflow.python.util import nest
        from tensorpack.tfutils.tower import get_current_tower_context, TowerContext
        assert(batchSize > 0)
        nn_context = get_current_nn_context()
        if initial_states is None:
            assert(nn_context)
            with tf.variable_scope('rnnstate/'+name, reuse=False):
                flatten_sizes = nest.flatten(cell.state_size)
                initial_states_flatten = []
                for size in flatten_sizes:
                    if isinstance(size, int): size = [size]
                    elif isinstance(size, tf.TensorShape): size = size.as_list()
                    else: raise TypeError("unknown type size {}".format(size))
                    v = tf.Variable(np.zeros([batchSize] + size, inputs.dtype.as_numpy_dtype), dtype=inputs.dtype, trainable=False)
                    initial_states_flatten.append(v)
                # for sidx, si in enumerate(flatten_sizes):

                initial_states = nest.pack_sequence_as(cell.state_size, initial_states_flatten)
        initial_states_flatten = nest.flatten(initial_states)
        notKnowBatchSize = inputs.get_shape().as_list()[0] is None
        if notKnowBatchSize: # not know batch
            assert(i_agentIdent is not None)
            initial_states = nest.pack_sequence_as(cell.state_size, [tf.gather(s, i_agentIdent) for s in initial_states_flatten])
        if not nn_context.is_training:
            reset_ops = []
            for s in initial_states_flatten:
                if notKnowBatchSize:
                    resetIndexs = tf.gather(i_agentIdent, tf.where(tf.not_equal(i_resetRNN, 0)))
                else:
                    resetIndexs = tf.where(tf.not_equal(i_resetRNN, 0))
                reset_ops.append(tf.scatter_sub(s, resetIndexs, tf.gather(s, resetIndexs)))
            with tf.control_dependencies(reset_ops):
                inputs = tf.identity(inputs)

        outputs, output_states = tf.nn.dynamic_rnn(cell,
                                                   inputs,
                                                   initial_state=initial_states,
                                                   sequence_length=i_sequenceLength,
                                                   time_major=False,
                                                   scope=name,
                                                   )

        update_ops = []
        for d, s in zip(initial_states_flatten, nest.flatten(output_states)):
            update_data = tf.where(tf.equal(i_resetRNN, 0), s, tf.zeros_like(s))
            if notKnowBatchSize:
                update_ops.append(tf.scatter_update(d, i_agentIdent, update_data))
            else:
                update_ops.append(tf.assign(d, update_data))

        with tf.control_dependencies([outputs]):
            update_op = tf.group(*update_ops, name=name+'/udpateRNN')
        with tf.control_dependencies([update_op]):
            # outputs = tf.cond(tf.count_nonzero(i_resetRNN) > 0,
            #                   lambda: tf.Print(outputs, [tf.count_nonzero(i_resetRNN), tf.gather(initial_states_flatten[0], reset_indices), initial_states_flatten[0]], 'resetRNN='),
            #                   # lambda: tf.Print(outputs, initial_states_flatten, 'init states.{} = '.format(name), summarize=4),
            #                   lambda: outputs,
            #                   )
            return tf.identity(outputs, name=name+'/output')