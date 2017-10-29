"""Utilities for generating random phantoms."""

import numpy as np
import odl


def random_ellipse(interior=False):
    if interior:
        x_0 = 0.5 * (np.random.rand() - 0.5)
        y_0 = 0.5 * (np.random.rand() - 0.5)
    else:
        x_0 = 2 * np.random.rand() - 1.0
        y_0 = 2 * np.random.rand() - 1.0

    return (np.random.rand() * np.random.exponential(0.2),
            np.random.exponential() * 0.2, np.random.exponential() * 0.2,
            x_0, y_0,
            np.random.rand() * 2 * np.pi)


def thorus_phantom(spc, ellipses):
    shrunken_ellipses = [(a,
                          0.95 * rx, 0.95 * ry,
                          x, y,
                          theta)
                         for (a, rx, ry, x, y, theta) in ellipses]
    return (odl.phantom.ellipsoid_phantom(spc, ellipses) -
            odl.phantom.ellipsoid_phantom(spc, shrunken_ellipses))


def random_phantom_w_translation(spc, n_ellipse=2, interior=True, offset_pixels=10):
    ellipses = [random_ellipse(interior=interior) for _ in range(n_ellipse)]
    ellipses = [(1.0,
                 0.5, 0.5,
                 x,
                 y,
                 0)
                for (a, rx, ry, x, y, theta) in ellipses]

    cell_sides = 2 / np.array(spc.shape)
    translated_ellipses = [(a,
                            rx, ry,
                            x + cell_sides[0] * np.random.randint(-offset_pixels, offset_pixels),
                            y + cell_sides[1] * np.random.randint(-offset_pixels, offset_pixels),
                            theta)
                           for (a, rx, ry, x, y, theta) in ellipses]
    p1 = thorus_phantom(spc, ellipses)
    p2 = thorus_phantom(spc, translated_ellipses)

    p1 = np.abs(p1)
    p2 = np.abs(p2)

    #p1 = p1 / np.mean(p1)
    #p2 = p2 / np.mean(p2)

    return p1, p2


if __name__ == '__main__':
    spc = odl.uniform_discr([-1, -1], [1, 1], [128, 128])

    p1, p2 = random_phantom_w_translation(spc)

    p1.show('p1')
    p2.show('p2')
