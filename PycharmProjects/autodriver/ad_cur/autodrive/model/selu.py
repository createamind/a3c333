#coding: utf-8
import numpy as np
import pandas as pd

def selu(x):
    from tensorflow.python.framework import ops
    import tensorflow as tf

    with ops.name_scope('selu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

    # import tensorflow as tf
    # alpha = 1.6732632423543772848170429916717
    # scale = 1.0507009873554804934193349852946
    # return scale*tf.where(x>0.0, x, alpha*tf.exp(x)-alpha)


def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                 noise_shape=None, seed=None, name=None, training=False):
    from tensorflow.python.framework import ops, tensor_shape, tensor_util
    from tensorflow.python.ops import array_ops, random_ops, math_ops
    from tensorflow.python.layers import utils
    import numbers
    import tensorflow as tf
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0. < keep_prob <= 1.:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))


def fc_selu(input, out_dim, name = None, keep_prob = 1., is_training = False):
    import tensorflow as tf
    shape_input = input.get_shape().as_list()
    assert(len(shape_input) == 2)
    dim_input = shape_input[-1]
    l = tf.layers.dense(input, out_dim, activation=selu,
                        kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1.0/dim_input)),
                        bias_initializer=tf.random_normal_initializer(stddev=0.),
                        # bias_initializer=tf.random_normal_initializer(stddev=0.),
                        name=name)
    import numbers
    if isinstance(keep_prob, numbers.Real):
        assert(keep_prob >= 0. and keep_prob <= 1.)
        if not is_training: keep_prob = 1.
    if not is_training: return l
    l = dropout_selu(l, rate=1.-keep_prob, training = is_training, name=name)
    return l