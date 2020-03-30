import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from stn import SpatialTransformer
from locnet import ConvLocNet

np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_classes = 10
nb_epoch = 12

DIM = 60
mnist_cluttered = "mnist_cluttered_60x60_6distortions.npz"

with tf.Session() as sess:
    data = np.load(mnist_cluttered)
    X_train, y_train = data['x_train'], np.argmax(data['y_train'], axis=-1)
    X_valid, y_valid = data['x_valid'], np.argmax(data['y_valid'], axis=-1)
    X_test, y_test = data['x_test'], np.argmax(data['y_test'], axis=-1)
    # reshape for convolutions
    X_train = X_train.reshape((X_train.shape[0], DIM, DIM, 1))
    X_valid = X_valid.reshape((X_valid.shape[0], DIM, DIM, 1))
    X_test = X_test.reshape((X_test.shape[0], DIM, DIM, 1))

    y_train = sess.run(tf.one_hot(y_train, nb_classes))
    y_valid = sess.run(tf.one_hot(y_valid, nb_classes))
    y_test = sess.run(tf.one_hot(y_test, nb_classes))
    print("Train samples: {}".format(X_train.shape))
    print("Validation samples: {}".format(X_valid.shape))
    print("Test samples: {}".format(X_test.shape))

    input_shape = np.squeeze(X_train.shape[1:])

    print("Input shape:", input_shape)

    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((50, 6), dtype='float32')
    weights = [W, b.flatten()]

    batch_input = tf.placeholder(tf.float32, np.concatenate([[batch_size], input_shape]), name='X')
    y = tf.placeholder(tf.int32, [batch_size, nb_classes])

    stn = SpatialTransformer(localization_net=ConvLocNet(name='locnet',  filters=(20, 20, 720, 50)),
                             output_size=(30, 30),
                             input_shape=input_shape,
                             batch_size=batch_size,
                             num_channels=1)

    stn_output = stn.apply(batch_input)
    out = tf.layers.conv2d(stn_output, 32, (3, 3), activation=tf.nn.relu, padding='VALID')
    out = tf.layers.max_pooling2d(out, (2, 2), strides=(1, 1))
    out = tf.layers.conv2d(out, 32, (3, 3), activation=tf.nn.relu, padding='VALID')
    out = tf.layers.max_pooling2d(out, (2, 2), strides=(1, 1))
    out = tf.contrib.layers.flatten(out)
    out = tf.layers.dense(out, 256, tf.nn.relu)
    logits = tf.layers.dense(out, nb_classes)
    
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    casted_corr_pred = tf.cast(correct_prediction, tf.float32)
    accuracy_operation = tf.reduce_mean(casted_corr_pred)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss_ = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer()
    update_gradient = optimizer.minimize(loss_)

    tf.global_variables_initializer().run()
    idx = np.arange(0, X_train.shape[0])
    for n in range(10):
        np.random.shuffle(idx)
        X_train = X_train[idx]
        y_train = y_train[idx]

        for i in range(150):
            print (i)
            x_batch = X_train[i * batch_size:(i + 1) * batch_size].astype('float32')
            y_batch = y_train[i * batch_size:(i + 1) * batch_size].astype('float32')
            sess.run([update_gradient, loss_],
                     feed_dict={batch_input: x_batch,
                     y: y_batch})
        vloss = []
        vacc = []
        for j in range(X_valid.shape[0]//batch_size):
            x_valid_batch = X_valid[j * batch_size:(j + 1) * batch_size].astype('float32')
            y_valid_batch = y_valid[j * batch_size:(j + 1) * batch_size].astype('float32')
            l, a = sess.run([loss_, accuracy_operation],
                           feed_dict={batch_input: x_valid_batch,
                           y: y_valid_batch})
            vloss.append(l)
            vacc.append(a)

        print ('Mean loss in valid set', np.mean(vloss))
        print ('Mean acc in valid set', np.mean(vacc))

    x_test_batch = X_test[0:batch_size]
    y_test_batch = y_test[0:batch_size]

    stn, loss, acc = sess.run([stn_output, loss_, accuracy_operation],
                              feed_dict={batch_input: x_test_batch,
                                         y: y_test_batch})

    print ('Test results: ')
    print (loss, acc)

    for j in range(1, 9):
        fig = plt.figure(figsize=(7, 7))
        fig.add_subplot(1, 2, 1)
        plt.imshow(np.squeeze(x_test_batch[i]), cmap='gray', interpolation='none')
        fig.add_subplot(1, 2, 2)
        plt.imshow(np.squeeze(stn[i]), cmap='gray', interpolation='none')
        plt.title('Cluttered MNIST', fontsize=20)
        plt.axis('off')
        plt.show()
