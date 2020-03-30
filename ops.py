import tensorflow as tf

try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except:
    image_summary = tf.compat.v1.summary.image
    scalar_summary = tf.compat.v1.summary.scalar
    histogram_summary = tf.compat.v1.summary.histogram
    merge_summary = tf.compat.v1.summary.merge
    SummaryWriter = tf.compat.v1.summary.FileWriter

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(input_, n_filters,
           k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
           name="conv2d"):
    """2D Convolution with options for kernel size, stride, and init deviation.
        Parameters
        ----------
        x : Tensor
            Input tensor to convolve.
        n_filters : int
            Number of filters to apply.
        k_h : int, optional
            Kernel height.
        k_w : int, optional
            Kernel width.
        d_h : int, optional
            Stride in rows.
        d_w : int, optional
            Stride in cols.
        stddev : float, optional
            Initialization's standard deviation.
        name : str, optional
            Variable scope to use.
        Returns
        -------
        x : Tensor
            Convolved input.
        """
    with tf.variable_scope(name):
        w = tf.compat.v1.get_variable('w', [k_h, k_w, input_.get_shape()[-1], n_filters],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.compat.v1.get_variable('biases', [n_filters], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def deconv2d(input_, n_filters,
             k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.compat.v1.get_variable('w', [k_h, k_w, n_filters[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=n_filters,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=n_filters,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.compat.v1.get_variable('biases', [n_filters[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x, name=name)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    """Fully-connected network.
        Parameters
        ----------
        x : Tensor
            Input tensor to the network.
        n_units : int
            Number of units to connect to.
        scope : str, optional
            Variable scope to use.
        stddev : float, optional
            Initialization's standard deviation.
        activation : arguments, optional
            Function which applies a nonlinearity
        Returns
        -------
        x : Tensor
            Fully-connected output.
        """
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.compat.v1.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.compat.v1.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
