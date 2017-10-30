"""Test that the wasserstein distance works as intended.

We validate this by creating a small square and moving it across the image,
we then expect to get the cost function back.
"""

import tensorflow as tf
import numpy as np
import odl
import matplotlib.pyplot as plt

from wasserstein_util import wasserstein_distance

sess = tf.InteractiveSession()

# Select parameters
size = 512
epsilon = 1e-3
cutoff = 0.3
p = 4

# Create ODL space (for phantom generation)
space = odl.uniform_discr([-1, -1], [1, 1], [size, size],
                          dtype='float32', interp='linear')


# Compute wasserstein distance
x0 = tf.placeholder('float32', [size, size])
x1 = tf.placeholder('float32', [size, size])
wd = wasserstein_distance(x0[None, ...] + 1e-3, x1[None, ...] + 1e-3,
                          epsilon=epsilon, niter=10, cutoff=cutoff, p=p)

x0_arr = odl.phantom.cuboid(space, [-0.02, -0.1], [0.02, 0.1]).asarray() * 1000

offsets = np.linspace(-1.0, 1.0, 101)
y = []
for dx in offsets:
    x1_arr = odl.phantom.cuboid(space, [-0.02 + dx, -0.1], [0.02 + dx, 0.1]).asarray()
    x1_arr *= np.sum(x0_arr) / np.sum(x1_arr)

    result_dx = wd.eval(feed_dict={x0: x0_arr, x1: x1_arr})
    print('{} {}'.format(dx, result_dx))

    y.append(result_dx)

plt.plot(offsets, y,
         label='computed')
plt.plot(offsets, np.mean(x0_arr) * (1 - np.exp(- offsets ** p / cutoff ** p)),
         label='expected')
plt.legend()
