import os

import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def func(input_x):
    return tf.math.sin(input_x[..., 0]) + tf.math.sin(input_x[..., 1])


x = tf.linspace(0., 2 * 3.1415926, 500)
y = tf.linspace(0., 2 * 3.1415926, 500)
point_x, point_y = tf.meshgrid(x, y)
points = tf.stack([point_x, point_y], axis=2)
z = func(points)

plt.figure("plot 2d func value")
plt.imshow(z, origin="lower", interpolation="none")
plt.colorbar()

plt.figure("plot 2d func contour")
plt.contour(point_x, point_y, z)
plt.colorbar()

plt.show()
