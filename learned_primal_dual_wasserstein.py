"""Learned Primal-Dual Reconstruction with wasserstein loss."""

import os
import adler
from adler.tensorflow import prelu, cosine_decay

import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow

from wasserstein_util import wasserstein_distance
from phantom import random_phantom_w_translation

np.random.seed(5)
name = os.path.splitext(os.path.basename(__file__))[0]

sess = tf.InteractiveSession()

# Create ODL data structures
size = 512
space = odl.uniform_discr([-256, -256], [256, 256], [size, size],
                          dtype='float32')

geometry = odl.tomo.parallel_beam_geometry(space, num_angles=30)
operator = odl.tomo.RayTransform(space, geometry)

# Ensure operator has fixed operator norm for scale invariance
opnorm = odl.power_method_opnorm(operator)
operator = (1 / opnorm) * operator

# Create tensorflow layer from odl operator
odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(operator,
                                                          'RayTransform')
odl_op_layer_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(operator.adjoint,
                                                                  'RayTransformAdjoint')

# User selected paramters
n_data = 1
n_iter = 10
n_primal = 5
n_dual = 5

def generate_data(validation=False):
    """Generate a set of random data."""
    n_generate = 1 if validation else n_data

    y_arr = np.empty((n_generate, operator.range.shape[0], operator.range.shape[1], 1), dtype='float32')
    x_true_arr = np.empty((n_generate, space.shape[0], space.shape[1], 1), dtype='float32')

    for i in range(n_generate):
        p1, p2 = random_phantom_w_translation(space, offset_pixels=40)

        data = operator(p2)
        noisy_data = data + odl.phantom.white_noise(operator.range) * np.mean(np.abs(data)) * 0.05

        x_true_arr[i, ..., 0] = p1
        y_arr[i, ..., 0] = noisy_data

    return y_arr, x_true_arr


with tf.name_scope('placeholders'):
    x_true = tf.placeholder(tf.float32, shape=[None, size, size, 1], name="x_true")
    y_rt = tf.placeholder(tf.float32, shape=[None, operator.range.shape[0], operator.range.shape[1], 1], name="y_rt")
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')


def apply_conv(x, filters=32):
    return tf.layers.conv2d(x, filters=filters, kernel_size=3, padding='SAME',
                            kernel_initializer=tf.contrib.layers.xavier_initializer())


with tf.name_scope('tomography'):
    with tf.name_scope('initial_values'):
        primal = tf.concat([tf.zeros_like(x_true)] * n_primal, axis=-1)
        dual = tf.concat([tf.zeros_like(y_rt)] * n_dual, axis=-1)

    for i in range(n_iter):
        with tf.variable_scope('dual_iterate_{}'.format(i)):
            evalop = odl_op_layer(primal[..., 1:2])
            update = tf.concat([dual, evalop, y_rt], axis=-1)

            update = prelu(apply_conv(update), name='prelu_1')
            update = prelu(apply_conv(update), name='prelu_2')
            update = apply_conv(update, filters=n_dual)
            dual = dual + update

        with tf.variable_scope('primal_iterate_{}'.format(i)):
            evalop = odl_op_layer_adjoint(dual[..., 0:1])
            update = tf.concat([primal, evalop], axis=-1)

            update = prelu(apply_conv(update), name='prelu_1')
            update = prelu(apply_conv(update), name='prelu_2')
            update = apply_conv(update, filters=n_primal)
            primal = primal + update

    x_result = primal[..., 0:1]


with tf.name_scope('loss'):
    residual = x_result - x_true
    squared_error = residual ** 2
    loss_l2 = tf.reduce_mean(squared_error)

    x_result_pos = tf.nn.relu(x_result)
    x_corr = x_result_pos * tf.reduce_mean(x_true) / (tf.reduce_mean(x_result_pos) + 1e-5)
    wd = wasserstein_distance(x_true[..., 0] + 1e-3, x_corr[..., 0] + 1e-3,
                              epsilon=1e-3, niter=10, cutoff=0.3, p=4)

    loss_sum = (tf.reduce_mean(x_true) - tf.reduce_mean(x_result)) ** 2
    loss_wasserstein = tf.reduce_mean(wd)

    loss = 1e2 * loss_sum + loss_wasserstein


with tf.name_scope('optimizer'):
    # Learning rate
    global_step = tf.Variable(0, trainable=False)
    maximum_steps = 20001
    starter_learning_rate = 1e-3
    learning_rate = cosine_decay(starter_learning_rate,
                                 global_step,
                                 maximum_steps,
                                 name='learning_rate')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt_func = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                          beta2=0.99)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1)
        optimizer = opt_func.apply_gradients(zip(grads, tvars),
                                             global_step=global_step)


# Summaries
# tensorboard --logdir=...

with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('loss_sum', loss_sum)
    tf.summary.scalar('loss_l2', loss_l2)
    tf.summary.scalar('loss_wasserstein', loss_wasserstein)
    tf.summary.scalar('psnr', adler.tensorflow.psnr(x_result, x_true))

    tf.summary.image('x_result', x_result_pos)
    tf.summary.image('x_true', x_true)
    tf.summary.image('squared_error', squared_error)
    tf.summary.image('residual', residual)

    merged_summary = tf.summary.merge_all()
    test_summary_writer, train_summary_writer = adler.tensorflow.util.summary_writers(name, cleanup=True)

# Initialize all TF variables
sess.run(tf.global_variables_initializer())

# Add op to save and restore
saver = tf.train.Saver()

# Generate validation data
if 0:
    saver.restore(sess,
                  adler.tensorflow.util.default_checkpoint_path(name))

# Train the network
y_arr_validate, x_true_arr_validate = generate_data(validation=True)
for i in range(0, maximum_steps):
    y_arr, x_true_arr = generate_data()

    _, merged_summary_result_train, global_step_result = sess.run([optimizer, merged_summary, global_step],
                              feed_dict={x_true: x_true_arr,
                                         y_rt: y_arr,
                                         is_training: True})

    if i>0 and i%10 == 0:
        loss_result, merged_summary_result, global_step_result = sess.run([loss, merged_summary, global_step],
                              feed_dict={x_true: x_true_arr_validate,
                                         y_rt: y_arr_validate,
                                         is_training: False})

        train_summary_writer.add_summary(merged_summary_result_train, global_step_result)
        test_summary_writer.add_summary(merged_summary_result, global_step_result)

        print('iter={}, loss={}'.format(global_step_result, loss_result))

    if i>0 and i%1000 == 0:
        saver.save(sess,
                   adler.tensorflow.util.default_checkpoint_path(name))
