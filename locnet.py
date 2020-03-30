import numpy as np
from ops import *


class ConvLocNet(tf.compat.v1.layers.Layer):
    def __init__(self, name='ConvLocNet',
                 filters=None,
                 kernel_size=(5, 5),
                 pool_size=(2, 2),
                 strides=(1, 1),
                 theta_dim=6,
                 **kwargs):

        super(ConvLocNet, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.strides = strides
        self.theta_dim = theta_dim
        self.output_size = (2, 3)
        self.w1 = None
        self.wb1 = None
        self.w2 = None
        self.wb2 = None
        self.l1 = None
        self.lb1 = None
        self.l2 = None
        self.lb2 = None

    def build(self, input_shape):
        k_h = self.kernel_size[0]
        k_w = self.kernel_size[1]
        stddev = 0.02

        initial = np.array([[1., 0., 0.], [0., 1., 0.]])
        initial = initial.astype('float32')
        initial = initial.flatten()

        self.w1 = self.add_variable('w1', [k_h, k_w, input_shape[-1], self.filters[0]], tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))

        self.wb1 = self.add_variable('b1', [self.filters[0]], tf.float32, initializer=tf.constant_initializer(0.0))

        self.w2 = self.add_variable('w2', [k_h, k_w, self.filters[0], self.filters[1]], tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))

        self.wb2 = self.add_variable('b2', [self.filters[1]], tf.float32, initializer=tf.constant_initializer(0.0))

        self.l1 = self.add_variable("fc1", [self.filters[2], self.filters[3]], tf.float32,
                                    initializer=tf.random_normal_initializer(stddev=0.02))

        self.lb1 = self.add_variable("fcb1", [self.filters[3]], tf.float32,
                                     initializer=tf.constant_initializer(0.0))

        self.l2 = self.add_variable("fc2", [self.filters[3], self.theta_dim], dtype=tf.float32, initializer=tf.zeros_initializer())

        self.lb2 = self.add_variable("fcb2", initial.shape, dtype=tf.float32, initializer=tf.constant_initializer(initial))

        self.built = True

    def _compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None,
                int(output_size[0]),
                int(output_size[1]),
                int(input_shape[-1]))

    def call(self, inputs, mask=None):
        if not self.built:
            self.build(inputs.get_shape().as_list())

        d_h, d_w = self.pool_size[0:2]
        x = tf.nn.max_pool2d(inputs, ksize=[1, d_h, d_w, 1], strides=[1, 1, 1, 1], padding='VALID')
        x = tf.nn.conv2d(x, self.w1, strides=[1, 1, 1, 1], padding='VALID')
        x = tf.reshape(tf.nn.bias_add(x, self.wb1), x.get_shape())
        x = tf.nn.relu(x)

        x = tf.nn.max_pool2d(x, ksize=[1, d_h, d_w, 1], strides=[1, 1, 1, 1], padding='VALID')
        x = tf.nn.conv2d(x, self.w2, strides=[1, 1, 1, 1], padding='VALID')
        x = tf.reshape(tf.nn.bias_add(x, self.wb2), x.get_shape())
        x = tf.nn.relu(x)

        x = tf.contrib.layers.flatten(x)

        x = tf.matmul(x, self.l1) + self.lb1

        theta = tf.matmul(x, self.l2) + self.lb2

        return theta


