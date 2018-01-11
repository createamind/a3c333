#coding: utf-8


import numpy as np
# import six
import tensorflow as tf
from drlutils.utils import logger

GAMMA = 0.99
LOCAL_TIME_MAX = 5
STEPS_PER_EPOCH = 1000
EVAL_EPISODE = 0
POOL_SIZE = 64
TRAIN_BATCH_SIZE = 32
PREDICT_MAX_BATCH_SIZE = 16     # batch for efficient forward
PREDICTOR_THREAD_PER_GPU = 3
PREDICTOR_THREAD = None
INIT_LEARNING_RATE_A = 1e-4
INIT_LEARNING_RATE_C = 1e-4

evaluators = []

import docopt, os
args = docopt.docopt(
'''
Usage:
    main.py train [--gpu GPU] [options]
    main.py dataserver [options]
    main.py infer  [--gpu GPU] [--load MODULEWEIGHTS] [options]
    main.py test  [options]

Options:
    -h --help                   Show the help
    --version                   Show the version
    --gpu GPU                   comma separated list of GPU(s)
    --load MODELWEIGHTS         load weights from file
    --simulators SIM            simulator count             [default: 16]
    --debug_mode                set debug mode
    --a3c_instance_idx IDX      set a3c_instance_idx            [default: 0]
    --continue                  continue mode, load saved weights
    --tfdbg
    --log LEVEL                 log level                       [default: info]
    --target TARGET             test target
    --fake_agent                use fake agent to debug          
''', version='0.1')

os.environ['ICE_RPCIO_LISTEN_PORT_BASE'] = "51000"
if args['train']:
    from drlutils.dataflow.tensor_io import TensorIO_AgentPools
    data_io = TensorIO_AgentPools(train_targets=['AD'])

# from tensorpack.tfutils import summary
from drlutils.model.base import ModelBase
class Model(ModelBase):
    def _get_inputs(self, select = None):
        from drlutils.model.base import ModelBase, InputDesc
        inputs = [
            InputDesc(tf.float32, (None, 37), 'state'),
            InputDesc(tf.float32, (None, 2), 'action'),
            InputDesc(tf.float32, (None,), 'reward'),
            InputDesc(tf.float32, (None,), 'advantage'),
            InputDesc(tf.int32, (), 'sequenceLength'),
            InputDesc(tf.int32, (), 'resetRNN'),
            # InputDesc(tf.float32, (None,), 'action_prob'),
        ]
        if select is None:
            return inputs

        assert(type(select) in [list, tuple])
        return [i for i in inputs if i.name in select]

    def _build_ad_nn(self, tensor_io):
        from drlutils.dataflow.tensor_io import TensorIO
        assert(isinstance(tensor_io, TensorIO))
        from drlutils.model.base import get_current_nn_context
        from tensorpack.tfutils.common import get_global_step_var
        global_step = get_global_step_var()
        nnc = get_current_nn_context()
        is_training = nnc.is_training
        i_state = tensor_io.getInputTensor('state')
        i_agentIdent = tensor_io.getInputTensor('agentIdent')
        i_sequenceLength = tensor_io.getInputTensor('sequenceLength')
        i_resetRNN = tensor_io.getInputTensor('resetRNN')
        l = i_state
        # l = tf.Print(l, [i_state, tf.shape(i_state)], 'State = ')
        # l = tf.Print(l, [i_agentIdent, tf.shape(i_agentIdent)], 'agentIdent = ')
        # l = tf.Print(l, [i_sequenceLength, tf.shape(i_sequenceLength)], 'SeqLen = ')
        # l = tf.Print(l, [i_resetRNN, tf.shape(i_resetRNN)], 'resetRNN = ')
        with tf.variable_scope('critic', reuse=nnc.reuse) as vs:
            def _get_cell():
                cell = tf.nn.rnn_cell.BasicLSTMCell(256)
                # if is_training:
                #     cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.9)
                return cell

            cell = tf.nn.rnn_cell.MultiRNNCell([_get_cell() for _ in range(1)])
            rnn_outputs = self._buildRNN(l, cell, tensor_io.batchSizeMax,
                                         i_agentIdent=i_agentIdent,
                                         i_sequenceLength=i_sequenceLength,
                                         i_resetRNN=i_resetRNN,
                                         )
            rnn_outputs = tf.reshape(rnn_outputs, [-1, rnn_outputs.get_shape().as_list()[-1]])
            l = rnn_outputs
            from ad_cur.autodrive.model.selu import fc_selu
            for lidx in range(2):
                l = fc_selu(l, 200,
                            keep_prob=1.,  # 由于我们只使用传感器训练，关键信息不能丢
                            is_training=is_training, name='fc-{}'.format(lidx))
            value = tf.layers.dense(l, 1, name='fc-value')
            value = tf.squeeze(value, [1], name="value")
            if not hasattr(self, '_weights_critic'):
                self._weights_critic = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        with tf.variable_scope('actor', reuse=nnc.reuse) as vs:
            l = tf.stop_gradient(l)
            l = tf.layers.dense(l, 128, activation=tf.nn.relu6, name='fc-actor')
            mu_steering = 0.5 * tf.layers.dense(l, 1, activation=tf.nn.tanh, name='fc-mu-steering')
            mu_accel = tf.layers.dense(l, 1, activation=tf.nn.tanh, name='fc-mu-accel')
            mus = tf.concat([mu_steering, mu_accel], axis=-1)
            # mus = tf.layers.dense(l, 2, activation=tf.nn.tanh, name='fc-mus')
            # sigmas = tf.layers.dense(l, 2, activation=tf.nn.softplus, name='fc-sigmas')
            # sigmas = tf.clip_by_value(sigmas, -0.001, 0.5)
            def saturating_sigmoid(x):
                """Saturating sigmoid: 1.2 * sigmoid(x) - 0.1 cut to [0, 1]."""
                with tf.name_scope("saturating_sigmoid", [x]):
                    y = tf.sigmoid(x)
                    return tf.minimum(1.0, tf.maximum(0.0, 1.2 * y - 0.1))

            sigma_steering_ = 0.1 * tf.layers.dense(l, 1, activation=tf.nn.sigmoid, name='fc-sigma-steering')
            sigma_accel_ = 0.25 * tf.layers.dense(l, 1, activation=tf.nn.sigmoid, name='fc-sigma-accel')

            if not nnc.is_evaluating:
                sigma_beta_steering = tf.get_default_graph().get_tensor_by_name('actor/sigma_beta_steering:0')
                sigma_beta_accel = tf.get_default_graph().get_tensor_by_name('actor/sigma_beta_accel:0')
                sigma_beta_steering = tf.constant(1e-4)
                # sigma_beta_steering_exp = tf.train.exponential_decay(0.3, global_step, 1000, 0.5, name='sigma/beta/steering/exp')
                # sigma_beta_accel_exp = tf.train.exponential_decay(0.5, global_step, 5000, 0.5, name='sigma/beta/accel/exp')
            else:
                sigma_beta_steering = tf.constant(1e-4)
                sigma_beta_accel = tf.constant(1e-4)
            sigma_steering = (sigma_steering_ + sigma_beta_steering)
            sigma_accel = (sigma_accel_ + sigma_beta_accel)

            sigmas = tf.concat([sigma_steering, sigma_accel], axis=-1)
            # if is_training:
            #     pass
            #     # 如果不加sigma_beta，收敛会很慢，并且不稳定，猜测可能是以下原因：
            #     #   1、训练前期尽量大的探索可以避免网络陷入局部最优
            #     #   2、前期过小的sigma会使normal_dist的log_prob过大，导致梯度更新过大，网络一开始就畸形了，很难恢复回来
            #
            # if is_training:
            #     sigmas += sigma_beta_steering
            # sigma_steering = tf.clip_by_value(sigma_steering, sigma_beta_steering, 0.5)
            # sigma_accel = tf.clip_by_value(sigma_accel, sigma_beta_accel, 0.5)
            # sigmas = tf.clip_by_value(sigmas, 0.1, 0.5)
            # sigmas_orig = sigmas
            # sigmas = sigmas + sigma_beta_steering
            # sigmas = tf.minimum(sigmas + 0.1, 100)
            # sigmas = tf.clip_by_value(sigmas, sigma_beta_steering, 1)
            # sigma_steering += sigma_beta_steering
            # sigma_accel += sigma_beta_accel

            # mus = tf.concat([mu_steering, mu_accel], axis=-1)

            from tensorflow.contrib.distributions import Normal
            dists = Normal(mus, sigmas + 0.01)
            policy = tf.squeeze(dists.sample([1]), [0])
            # 裁剪到两倍方差之内
            policy = tf.clip_by_value(policy, mus - 2*sigmas, mus + 2*sigmas)
            if is_training:
                self._addMovingSummary(tf.reduce_mean(mu_steering, name='mu/steering/mean'),
                                       tf.reduce_mean(mu_accel, name='mu/accel/mean'),
                                       tf.reduce_mean(sigma_steering, name='sigma/steering/mean'),
                                       tf.reduce_max(sigma_steering, name='sigma/steering/max'),
                                       tf.reduce_mean(sigma_accel, name='sigma/accel/mean'),
                                       tf.reduce_max(sigma_accel, name='sigma/accel/max'),
                                       # sigma_beta_accel,
                                       # sigma_beta_steering,
                                       )
            # actions = tf.Print(actions, [mus, sigmas, tf.concat([sigma_steering_, sigma_accel_], -1), actions],
            #                    'mu/sigma/sigma.orig/act=', summarize=4)
            if not hasattr(self, '_weights_actor'):
                self._weights_actor = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
        policy = tf.identity(policy, 'policy')
        value = tf.identity(value, 'value')
        mus = tf.identity(mus, 'mus')
        sigmas = tf.identity(sigmas, 'sigmas')

        if not is_training:
            tensor_io.setOutputTensors(policy, value, mus, sigmas)
            return

        i_actions = tensor_io.getInputTensor("action")
        # i_actions = tf.Print(i_actions, [i_actions], 'actions = ')
        i_actions = tf.reshape(i_actions, [-1] + i_actions.get_shape().as_list()[2:])
        log_probs = dists.log_prob(i_actions)
        # exp_v = tf.transpose(
        #     tf.multiply(tf.transpose(log_probs), advantage))
        # exp_v = tf.multiply(log_probs, advantage)
        i_advantage = tensor_io.getInputTensor("advantage")
        i_advantage = tf.reshape(i_advantage, [-1] + i_advantage.get_shape().as_list()[2:])
        exp_v = log_probs * tf.expand_dims(i_advantage, -1)
        entropy = dists.entropy()
        entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                       initializer=tf.constant_initializer(0.01), trainable=False)
        exp_v = entropy_beta * entropy + exp_v
        loss_policy = tf.reduce_mean(-tf.reduce_sum(exp_v, axis=-1), name='loss/policy')

        i_futurereward = tensor_io.getInputTensor("reward")
        i_futurereward = tf.reshape(i_futurereward, [-1] + i_futurereward.get_shape().as_list()[2:])
        loss_value = tf.reduce_mean(0.5 * tf.square(value - i_futurereward))

        loss_entropy = tf.reduce_mean(tf.reduce_sum(entropy, axis=-1), name='xentropy_loss')

        from tensorflow.contrib.layers.python.layers.regularizers import apply_regularization, l2_regularizer
        loss_l2_regularizer = apply_regularization(l2_regularizer(1e-4), self._weights_critic)
        loss_l2_regularizer = tf.identity(loss_l2_regularizer, 'loss/l2reg')
        loss_value += loss_l2_regularizer
        loss_value = tf.identity(loss_value, name='loss/value')

        # self.cost = tf.add_n([loss_policy, loss_value * 0.1, loss_l2_regularizer])

        self._addParamSummary([('.*', ['rms', 'absmax'])])
        pred_reward = tf.reduce_mean(value, name='predict_reward')
        import tensorpack.tfutils.symbolic_functions as symbf
        advantage = symbf.rms(i_advantage, name='rms_advantage')
        self._addMovingSummary(loss_policy, loss_value,
                                   loss_entropy,
                                   pred_reward, advantage,
                                   loss_l2_regularizer,
                                   tf.reduce_mean(policy[:, 0], name='actor/steering/mean'),
                                   tf.reduce_mean(policy[:, 1], name='actor/accel/mean'),
                                   )
        return loss_policy, loss_value

    def _build_graph(self):
        from drlutils.model.base import NNContext
        gpu_towers = [0]
        for towerIdx, tower in enumerate(gpu_towers):
            with NNContext("ADTrain", device='/device:GPU:%d' % tower, summary_collection=towerIdx==0, is_training=True):
                data_io.createPool('AD-%d'%towerIdx, POOL_SIZE, sub_batch_size = 1, is_training = True,
                                   train_batch_size=TRAIN_BATCH_SIZE,
                                   predict_max_batch_size=PREDICT_MAX_BATCH_SIZE,
                                   torcsIdxOffset = data_io.getAgentCountTotal(),
                                   )
                tensor_io = data_io.getTensorIO("AD-%d/train"%towerIdx, self._get_inputs(), queue_size=50, is_training=True)
                loss_policy, loss_value = self._build_ad_nn(tensor_io)
                self._addLoss('policy%d'%towerIdx, loss_policy, opt=self._get_optimizer('actor'),
                              trainOpGroup='AD', tensor_io=tensor_io, var_list=self._weights_actor)
                self._addLoss('value%d'%towerIdx, loss_value, opt=self._get_optimizer('critic'),
                              trainOpGroup='AD', tensor_io=tensor_io, var_list=self._weights_critic)
                if not hasattr(self, '_weights_ad_train'):
                    self._weights_ad_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            with NNContext("ADPred", device='/device:GPU:%d'%tower, summary_collection=False, is_training=False):
                with tf.variable_scope('pred', reuse=towerIdx>0) as vs:
                    tensor_io = data_io.getTensorIO("AD-%d/pred"%towerIdx,
                                                    self._get_inputs(['agentIdent', 'state', 'sequenceLength', 'resetRNN']),
                                                    is_training=False, allow_no_full_batch = True,
                                                    )
                    self._build_ad_nn(tensor_io)
                    self._addThreadOp('AD-%d/pred'%towerIdx, tensor_io.getOutputOp())
                    if not hasattr(self, '_weights_ad_pred'):
                        self._weights_ad_pred = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
                        assert(len(self._weights_ad_pred) == len(self._weights_ad_train))
                        self._sync_op_pred = tf.group(*[d.assign(s) for d, s in zip(self._weights_ad_pred, self._weights_ad_train)])
                        # self._sync_op_pred = tf.group(*[d.assign(s + tf.random_normal(tf.shape(s), stddev=2e-4)) for d, s in zip(self._weights_ad_pred, self._weights_ad_train)])

        for eidx, evaluator in enumerate(evaluators):
            with NNContext(evaluator.name, device='/device:GPU:%d' % gpu_towers[eidx%len(gpu_towers)], add_variable_scope=True, summary_collection=True, is_evaluating=True):
                data_io.createPool(evaluator.name.replace('/', '_'), evaluator._batch_size, sub_batch_size=1, is_training=False, torcsIdxOffset = data_io.getAgentCountTotal())
                tensor_io = evaluator.getTensorIO(self._get_inputs(['agentIdent', 'state', 'sequenceLength', 'resetRNN']))
                self._build_ad_nn(tensor_io)
                evaluator.set_weights(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=evaluator.name))

    def _calc_learning_rate(self, name, epoch, lr):
        def _calc():
            lr_init = INIT_LEARNING_RATE_A if name == 'actor' else INIT_LEARNING_RATE_C
            lrs = [(0, lr_init * 0.25),
                   (1, lr_init * 0.5),
                   (2, lr_init * 1.0),
                   (3, lr_init * 0.5),
                   (4, lr_init * 0.25),
                   (5, lr_init * 0.128),
                   # (100, lr_init/16),
                   ]
            for idx in range(len(lrs) - 1):
                if epoch >= lrs[idx][0] and epoch < lrs[idx+1][0]:
                    return lrs[idx][1]
            return lrs[-1][1]
        # return INIT_LEARNING_RATE_A
        # ret = INIT_LEARNING_RATE_A if name == 'actor' else INIT_LEARNING_RATE_C
        ret = _calc()
        return ret

    def _get_optimizer(self, name):
        from tensorpack.tfutils import optimizer
        from tensorpack.tfutils.gradproc import SummaryGradient, GlobalNormClip, MapGradient
        init_lr = INIT_LEARNING_RATE_A if name == 'actor' else INIT_LEARNING_RATE_C
        import tensorpack.tfutils.symbolic_functions as symbf
        lr = symbf.get_scalar_var('learning_rate/' + name, init_lr, summary=True)
        opt = tf.train.AdamOptimizer(lr)
        logger.info("create opt {}".format(name))
        if name == 'critic':
            gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.05), regex='^critic/.*')]
        elif name == 'actor':
            gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1), regex='^actor/.*')]
        else: assert(0)
        gradprocs.append(SummaryGradient())
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt

def get_config():
    M = Model()

    dataflow = data_io
    from tensorpack.callbacks.base import Callback
    class CBSyncWeight(Callback):
        def _before_run(self, ctx):
            if self.local_step % 10 == 0:
                return [M._sync_op_pred]
    import functools
    from tensorpack.train.config import TrainConfig
    from tensorpack.callbacks.saver import ModelSaver
    from tensorpack.callbacks.graph import RunOp
    from tensorpack.callbacks.param import ScheduledHyperParamSetter, HumanHyperParamSetter, HyperParamSetterWithFunc
    from tensorpack.tfutils import sesscreate
    from tensorpack.tfutils.common import get_default_sess_config
    import tensorpack.tfutils.symbolic_functions as symbf

    sigma_beta_steering = symbf.get_scalar_var('actor/sigma_beta_steering', 0.3, summary=True, trainable=False)
    sigma_beta_accel = symbf.get_scalar_var('actor/sigma_beta_accel', 0.3, summary=True, trainable=False)

    return TrainConfig(
        model=M,
        data=dataflow,
        callbacks=[
            ModelSaver(),
            HyperParamSetterWithFunc(
                'learning_rate/actor',
                functools.partial(M._calc_learning_rate, 'actor')),
            HyperParamSetterWithFunc(
                'learning_rate/critic',
                functools.partial(M._calc_learning_rate, 'critic')),

            # ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            # HumanHyperParamSetter('learning_rate'),
            # HumanHyperParamSetter('entropy_beta'),
            ScheduledHyperParamSetter('actor/sigma_beta_accel', [(1, 0.2), (2, 0.01)]),
            ScheduledHyperParamSetter('actor/sigma_beta_steering', [(1, 0.1), (2, 0.01)]),
            CBSyncWeight(),
            data_io,
            # PeriodicTrigger(Evaluator(
            #     EVAL_EPISODE, ['state'], ['policy'], get_player),
            #     every_k_epochs=3),
        ] + evaluators,
        session_creator=sesscreate.NewSessionCreator(
            config=get_default_sess_config(0.5)),
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
    )


if __name__ == '__main__':
    if args['train']:
        from drlutils.utils import logger
        logger.info("Begin train task")
        from ad_cur.autodrive.agent.pool import AgentPool, AgentPoolFake
        clsPool = AgentPoolFake if args['--fake_agent'] else AgentPool
        from drlutils.evaluator.evaluator import EvaluatorBase
        class Evaluator(EvaluatorBase):
            def _init(self):
                super(Evaluator, self)._init()


        if args['--gpu']:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(sorted(list(set(args['--gpu'].split(',')))))
        # os.system('killall -9 torcs-bin')

        dirname = '/tmp/torcs/trainlog'
        from tensorpack.utils import logger
        logger.set_logger_dir(dirname, action='k' if args['--continue'] else 'b')
        logger.info("Backup source to {}/source/".format(logger.LOG_DIR))
        source_dir = os.path.dirname(__file__)
        os.system('rm -f {}/sim-*; mkdir -p {}/source; rsync -a --exclude="core*" --exclude="cmake*" --exclude="build" {} {}/source/'
                  .format(source_dir, logger.LOG_DIR, source_dir, logger.LOG_DIR))
        if not args['--continue']:
            os.system('rm -rf /tmp/torcs/memory')
        os.system('rm -f /tmp/torcs_run/*.pid')

        # if not args['--fake_agent']:
        #     logger.info("Create simulators, please wait...")
            # clsPool.startNInstance(BATCH_SIZE)
        evaluators = [
            Evaluator(data_io, 'evaluate/valid', batch_size=4, is_training=False, cls_pool=clsPool, sync_step = 10),
        ]
        from tensorpack.utils.gpu import get_nr_gpu
        from tensorpack.train.feedfree import QueueInputTrainer
        from tensorpack.tfutils.sessinit import get_model_loader
        nr_gpu = get_nr_gpu()
        # trainer = QueueInputTrainer
        assert(nr_gpu > 0)
        if nr_gpu > 1:
            predict_tower = list(range(nr_gpu))[-nr_gpu // 2:]
        else:
            predict_tower = [0]
        PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU
        train_tower = list(range(nr_gpu))[:-nr_gpu // 2] or [0]
        logger.info("[Batch-A3C] Train on gpu {} and infer on gpu {}".format(
            ','.join(map(str, train_tower)), ','.join(map(str, predict_tower))))
        from drlutils.train.multigpu import MultiGPUTrainer
        trainer = MultiGPUTrainer
        config = get_config()
        if os.path.exists(logger.LOG_DIR + '/checkpoint'):
            from tensorpack.tfutils.sessinit import SaverRestore
            config.session_init = SaverRestore(logger.LOG_DIR + '/checkpoint')
        elif args['--load']:
            config.session_init = get_model_loader(args['--load'])
        config.tower = train_tower
        config.predict_tower = predict_tower
        trainer(config).train()
        import sys
        sys.exit(0)
    elif args['dataserver']:
        import os
        os.system('killall -9 torcs-bin > /dev/null 2>&1')
        from drlutils.dataflow.server import DataFlowServer
        from ad_cur.autodrive.agent.pool import AgentPool, AgentPoolFake
        clsPool = AgentPoolFake if args['--fake_agent'] else AgentPool

        try:
            ds = DataFlowServer(AgentPoolFake if args['--fake_agent'] else AgentPool, local_t_max=LOCAL_TIME_MAX, gamma=GAMMA)
            ds.run()
        except KeyboardInterrupt:
            pass
        import sys
        sys.exit(0)
    elif args['infer']:
        assert args['--load'] is not None
        from tensorpack.predict.config import PredictConfig
        from tensorpack.tfutils.sessinit import get_model_loader
        from tensorpack.predict.base import OfflinePredictor
        cfg = PredictConfig(
            model=Model(),
            session_init=get_model_loader(args['--load']),
            input_names=['state'],
            output_names=['policy', 'value'])
        if args['--target'] == 'play':
            def play_one_episode(player, func, verbose=False):
                player.restart_episode()
                def f(s):
                    # spc = player.get_action_space()
                    act = func([[s]])
                    print('act = {}'.format(act))
                    # if random.random() < 0.001:
                    #     act = spc.sample()
                    if verbose:
                        print(act)
                    return act[0][0], act[1][0]
                return np.mean(player.play_one_episode(f))


            def play_model(cfg, player):
                predfunc = OfflinePredictor(cfg)
                while True:
                    score = play_one_episode(player, predfunc)
                    print("Total:", score)
            pass
            play_model(cfg, get_player(0, train=False))
        # elif args.task == 'eval':
        #     eval_model_multithread(cfg, args.episode, get_player)
        # elif args.task == 'gen_submit':
        #     play_n_episodes(
        #         get_player(train=False, dumpdir=args.output),
        #         OfflinePredictor(cfg), args.episode)
            # gym.upload(output, api_key='xxx')
    elif args['test']:
        if args['--target'] == 'pool':
            from drlutils.dataflow.tensor_io import TensorIO_AgentPools
            from drlutils.model.base import InputDesc
            data_io = TensorIO_AgentPools(train_targets=['AD'])
            data_io.createPool('AD-0', batch_size=2, sub_batch_size=1, is_training=True)
            input_descs = [InputDesc(tf.float32, [None, 32], 'state'),
                           InputDesc(tf.int32, (), 'sequenceLength'),
                           InputDesc(tf.int32, (), 'resetRNN'),
                           ]
            tensor_io = data_io.getTensorIO('AD-0/pred', input_descs, is_training=False)
            state = tensor_io.getInputTensor('state')
            output_op = tensor_io.setOutputTensors(state)
            tensor_io_train = data_io.getTensorIO('AD-0/train', input_descs, is_training = True)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                data_io._before_train()
                sess.run(output_op)


        pass
