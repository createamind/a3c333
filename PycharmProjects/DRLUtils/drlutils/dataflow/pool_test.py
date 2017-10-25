#coding: utf-8
import numpy as np

def test_pool():
    from ..utils import logger
    from ..model.base import InputDesc
    import tensorflow as tf

    from .tensor_io import TensorIO_AgentPools

    pool = TensorIO_AgentPools(train_targets=['AD'])
    pool.createPool('main', 16, 1, True)

    pred_inputs = [InputDesc(tf.float32, (None, 29), 'state'),
                   ]
    train_inputs = pred_inputs + [InputDesc(tf.float32, (None, 2), 'action'),
                                  InputDesc(tf.float32, (None,), 'futurereward'),
                                  InputDesc(tf.float32, (None,), 'advantage'),
                                  InputDesc(tf.int32, (), 'sequenceLength'),
                                  InputDesc(tf.int32, (), 'isOver')
                                  ]
    tensorio_pred = pool.getTensorIO('main/pred', pred_inputs, is_training=False)
    output_op = tensorio_pred.setOutputTensors(
        tensorio_pred.getInputTensor('state')[:, :, :2],
        tensorio_pred.getInputTensor('state')[:, :, 0],
    )
    tensorio_train = pool.getTensorIO('main/train', train_inputs, is_training=True, queue_size=50)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(100):
            sess.run(output_op)

        pool.close()



