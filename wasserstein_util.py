"""Utilities for computing the wasserstein distance."""

import tensorflow as tf
import numpy as np


def inner(u, v):
    """Compute the inner product of u, v batch wise."""
    return tf.reduce_sum(u * v, axis=[1, 2])


def divide(u, v):
    """Compute u/v with zero division guard."""
    return tf.where(tf.less(tf.abs(v), 1e-7), tf.zeros_like(u), u / v)


class KMatrixFFT2(object):

    """Computes Ku quickly using the fourier transform."""

    def __init__(self, dist_matrix):
        self.dist_matrix_fft = self.fft(dist_matrix)

    def fft(self, x):
        return tf.fft2d(tf.cast(x, 'complex64'))

    def ifft(self, x):
        return tf.real(tf.ifft2d(x))

    def __call__(self, x):
        with tf.name_scope('K'):
            return self.ifft(self.dist_matrix_fft * self.fft(x))

    def adjoint(self, y):
        with tf.name_scope('Kt'):
            # Get the exact gradient using tf.gradients
            return tf.gradients(self(y), y, y)[0]


def wasserstein_distance_helper(matrix_param, epsilon, mu0, mu1, niter):
    """Helper function for wasserstein distance."""
    with tf.name_scope('wasserstein_distance_helper'):
        matrix_param = tf.cast(matrix_param, 'float32')

        with tf.name_scope('initialize_operators'):
            K_op = KMatrixFFT2(tf.exp(-matrix_param / epsilon))
            K_op_2 = KMatrixFFT2(matrix_param * tf.exp(-matrix_param / epsilon))

        with tf.name_scope('return_diagonal_scalings'):
            v = tf.ones_like(mu1) / tf.cast(tf.size(mu1), tf.float32)

            for j in range(niter):
                with tf.name_scope('iterate_{}'.format(j)):
                    u = divide(mu0, K_op(v))
                    v = divide(mu1, K_op.adjoint(u))

        return inner(u, K_op_2(v)) / tf.cast(tf.size(mu1), tf.float32)


def wasserstein_distance(mu0, mu1, epsilon, niter, cutoff, p=2):
    """Compute the wasserstein distance betwenn mu0 and mu1.

    This computes the entropy regularized entropy distance, where the cost is
    given by::

        c(x1, x2) = (1 - exp((||x1-x2||/cutoff)^p)

    This behaves like ||x1-x2||^p close to 0, but goes to 1 for
    ||x1-x2||>cutoff.

    Parameters
    ----------
    mu0, mu1 : `tensorflow.Tensor` with shape (batchn, nx, ny)
        The images we want to compute the difference between.
    epsilon : positive `float`
        Entropy regularization term.
    niter : positive `int`
        Number of sinkhorn iterations.
    cutoff : positive `float`
        Cutoff parameter for the p-norm, used for numerical stability.
    p : positive `float`
        Power of the p-norm
    """
    with tf.name_scope('wasserstein_distance'):
        # Perform all computations using float32
        mu0 = tf.cast(mu0, 'float32')
        mu1 = tf.cast(mu1, 'float32')

        # Compute the costs relative to the midpoint
        shape = mu0.shape
        xp, yp = np.meshgrid(np.linspace(-1, 1, shape[1]),
                             np.linspace(-1, 1, shape[2]))

        matrix_param = xp ** (p) + yp ** (p)

        # Multiply by cutoff ** p here for numerical stability, divide later
        matrix_param = cutoff ** p * (1 - np.exp(-matrix_param / cutoff ** p))

        # Translate cost relative to upper left corner
        matrix_param = (np.fft.ifftshift(matrix_param))[None, ...]

        # Compute the distance
        result = wasserstein_distance_helper(matrix_param,
                                             epsilon=epsilon,
                                             mu0=mu0,
                                             mu1=mu1,
                                             niter=niter)

        return result / cutoff ** (p)
