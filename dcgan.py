from __future__ import division
from stn import SpatialTransformer
from locnet import ConvLocNet
from scipy.io import loadmat
from glob import glob
from utils import *
from ops import *
import time
import math
import os


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):
    def __init__(self, sess, input_height=16, input_width=16, crop=True,
                 batch_size=20, sample_num=64, output_height=128, output_width=128,
                 df_dim=16, test_num=500, trainin_num=100000,
                 gfc_dim=1024, dfc_dim=1024, dataset_name='default',
                 dcv_fc=512, dcv_sc=256, dcv_th=128, dcv_foc=64,
                 input_fname_pattern='*.jpg', checkpoint_dir=None, data_dir='./data'):
        """

    #Arguments:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      input_height: Height of the input images to the generator
      input_height: Width of the input images to the generator
      output_width: Height of the output images to the generator
      output_width: Width of the output images to the generator
      df_dim: The base dimension for all Discriminator/Generator convolutional filters
    """
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.dcv_frst_channel = dcv_fc
        self.dcv_sec_channel = dcv_sc
        self.dcv_thr_channel = dcv_th
        self.dcv_four_channel = dcv_foc
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir

        self.data = glob(os.path.join(self.data_dir, self.dataset_name, self.input_fname_pattern))
        imreadImg = imread(self.data[0])

        if len(imreadImg.shape) >= 3:  # check if image is a non-grayscale image by checking channel number
            self.c_dim = imreadImg.shape[-1]
        else:
            self.c_dim = 1

        self.test_files = self.data[:test_num]
        self.cv_test_files = self.data[test_num + trainin_num:2 * test_num + trainin_num]
        self.data = self.data[test_num:test_num + trainin_num]
        self.grayscale = (self.c_dim == 1)

        self.build_model()

    def build_vgg_model(self, param_path):
        """Not change this if you want the last relu layer, change otherwise"""
        self.data_mat = loadmat(param_path)
        meta = self.data_mat['meta']
        classes = meta['classes']
        self.class_names = classes[0][0]['description'][0][0]
        normalization = meta['normalization']
        self.average_image = np.squeeze(normalization[0][0]['averageImage'][0][0][0][0])
        self.image_size = np.squeeze(normalization[0][0]['imageSize'][0][0])

    def build_model(self):
        """Here the model is build and the placeholder and Variables created"""
        self.build_vgg_model('vgg-face.mat')

        lr_image_dims = [self.input_height, self.input_width, self.c_dim]
        hd_image_dims = [self.output_height, self.output_width, self.c_dim]

        # Placeholder for G and D input
        self.hd_outputs = tf.compat.v1.placeholder(tf.float32, [self.batch_size] + hd_image_dims, name='hd_images')
        self.lr_inputs = tf.compat.v1.placeholder(tf.float32, [self.batch_size] + lr_image_dims, name='lr_images')

        # Placeholder to tricks for training
        self.is_normal_label = tf.compat.v1.placeholder(tf.float32, name='is_normal_label')
        self.is_training = tf.compat.v1.placeholder(tf.bool, name='is_training')
        self.alpha = tf.compat.v1.placeholder(tf.float32, None, name='alpha')
        self.lmbd = tf.compat.v1.placeholder(tf.float32, None, name='lambda')
        self.gamma = tf.constant(0.01, name='gamma', dtype='float32')

        # Simbolic calculation of G and D, also sampler for testing propose
        self.G = self.generator(self.lr_inputs)
        self.D, self.D_logits = self.discriminator(self.hd_outputs)
        self.sampler = self.generator(self.lr_inputs, reuse=True)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        # Calculating VGG filters on CPU, remove if a new GPU shows up
        with tf.device("/device:CPU:0"):
            G_img = self.G - self.average_image
            hd_img = self.hd_outputs - self.average_image
            self.G_vgg = self.run_vgg_model(G_img)['relu7']
            self.HD_vgg = self.run_vgg_model(hd_img)['relu7']

        # Calculating the loss function
        self.loss_function()

        # Salving histogram and loss summay
        self.save_histogram_summary()
        self.save_loss_summary()

        # Salving weights
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def run_vgg_model(self, input_maps):
        """Few lines of code running the vgg"""
        input_maps = tf.image.resize(input_maps, [self.image_size[0], self.image_size[1]])

        # read layer info
        layers = self.data_mat['layers']
        current = input_maps
        network = {}
        for layer in layers[0]:
            name = layer[0]['name'][0][0]
            layer_type = layer[0]['type'][0][0]
            if layer_type == 'conv':
                if name[:2] == 'fc':
                    padding = 'VALID'
                else:
                    padding = 'SAME'
                stride = layer[0]['stride'][0][0]
                kernel, bias = layer[0]['weights'][0][0]
                # kernel = np.transpose(kernel, (1, 0, 2, 3))
                bias = np.squeeze(bias).reshape(-1)
                conv = tf.nn.conv2d(current, tf.constant(kernel),
                                    strides=(1, stride[0], stride[0], 1), padding=padding)
                current = tf.nn.bias_add(conv, bias)
            elif layer_type == 'relu':
                current = tf.nn.relu(current)
            elif layer_type == 'pool':
                stride = layer[0]['stride'][0][0]
                pool = layer[0]['pool'][0][0]
                current = tf.nn.max_pool2d(current, ksize=(1, pool[0], pool[1], 1),
                                         strides=(1, stride[0], stride[0], 1), padding='SAME')

            elif layer_type == 'softmax':
                current = tf.nn.softmax(tf.reshape(current, [-1, len(self.class_names)]))

            network[name] = current

        return network

    def predict(self, address_lr_set):
        """This Function is to get the prediction of the Generator

            #Argument:
                address_lr_set: The input images paths
        """
        lr_imgs_path = [path.strip() for path in open(address_lr_set)]
        lr_image_dims = [self.input_height, self.input_width, self.c_dim]

        lr_inputs = tf.placeholder(tf.float32, [self.batch_size] + lr_image_dims, name='lr_images')

        G = self.generator(lr_inputs, reuse=True)
        test_batch_idxs = len(lr_imgs_path) // self.batch_size

        for i in range(test_batch_idxs):
            batch_files = lr_imgs_path[i * self.batch_size:self.batch_size * (i + 1)]
            sample_lr_batch = get_lr_image(batch_files, grayscale=self.grayscale)

            if self.grayscale:
                sample_lr_batch_images = np.array(sample_lr_batch).astype(np.float32)[:, :, :, None]
            else:
                sample_lr_batch_images = np.array(sample_lr_batch)

            samples = self.sess.run(
                G,
                feed_dict={
                    lr_inputs: sample_lr_batch_images,
                    self.is_training: False
                }
            )

            for j, img in enumerate(samples):
                image = (img + 1) * 127.5
                scipy.misc.imsave('./{}_gan.jpg'.format(batch_files[j]), image[:, :, 0])

    def test(self):
        """This function is just to test the model with the same training images"""
        lr_image_dims = [self.input_height, self.input_width, self.c_dim]
        lr_inputs = tf.placeholder(tf.float32, [self.batch_size] + lr_image_dims, name='lr_images')

        G = self.generator(lr_inputs, reuse=True)
        _psnr = []
        _bpsnr = []
        test_batch_idxs = len(self.cv_test_files) // self.batch_size
        start_time = time.time()
        for i in range(test_batch_idxs):
            batch_files = self.cv_test_files[i * self.batch_size:self.batch_size * (i + 1)]

            sample_lr_batch, sample_hd_batch = get_lr_hd_image(batch_files,
                                                               input_height=self.input_height,
                                                               input_width=self.input_width,
                                                               output_height=self.output_height,
                                                               output_width=self.output_width,
                                                               crop=self.crop,
                                                               grayscale=self.grayscale)
            if self.grayscale:
                sample_lr_batch_images = np.array(sample_lr_batch).astype(np.float32)[:, :, :, None]
                sample_hd_batch_images = np.array(sample_hd_batch).astype(np.float32)[:, :, :, None]
            else:
                sample_lr_batch_images = np.array(sample_lr_batch)
                sample_hd_batch_images = np.array(sample_hd_batch)

            samples = self.sess.run(
                G,
                feed_dict={
                    lr_inputs: sample_lr_batch_images,
                    self.is_training: False
                }
            )

            for j, lr_image in enumerate(sample_lr_batch_images):
                r_image = scipy.misc.imresize(lr_image,
                                              [sample_hd_batch_images.shape[1], sample_hd_batch_images.shape[2]])
                _bpsnr.append(psnr(sample_hd_batch_images[j], r_image))

            _psnr.append(psnr(samples, sample_hd_batch_images))

        end_time = time.time() - start_time
        print ('bicubic', np.mean(_bpsnr))
        print ('GAN', np.mean(_psnr))
        print ('Time', end_time)

    def train(self, config):
        """Training function"""

        # Seting Adam optimizer for G and D
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            adam = tf.train.AdamOptimizer()
            g_grads = adam.compute_gradients(self.g_loss, var_list=self.g_vars)
            d_grads = adam.compute_gradients(self.d_loss, var_list=self.d_vars)
            d_optim = adam.apply_gradients(d_grads)
            g_optim = adam.apply_gradients(g_grads)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # This is for tensorboard
        self.d_grads_sum = merge_summary([histogram_summary("%s-grad" % g[1].name, g[0]) for g in d_grads])
        self.g_grads_sum = merge_summary([histogram_summary("%s-grad" % g[1].name, g[0]) for g in g_grads])

        self.g_sum = merge_summary([self.d__sum, self.G_sum,
                                    self.d_loss_fake_sum,
                                    self.g_loss_sum, self.g_grads_sum])

        self.d_sum = merge_summary([self.d_sum, self.d_loss_real_sum,
                                    self.d_loss_sum, self.lr_inputs_sum,
                                    self.hd_outputs_sum, self.d_grads_sum])

        self.writer = SummaryWriter("./logs", self.sess.graph)

        # -----------------------
        # Restauring the last saved model
        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # Save HD and LR images to compare
        sample_lr_batch, sample_hd_batch = get_lr_hd_image(self.test_files[:self.sample_num],
                                                           input_height=self.input_height,
                                                           input_width=self.input_width,
                                                           output_height=self.output_height,
                                                           output_width=self.output_width,
                                                           crop=self.crop,
                                                           grayscale=self.grayscale)
        if self.grayscale:
            sample_lr_batch_images = np.array(sample_lr_batch).astype(np.float32)[:, :, :, None]
            sample_hd_batch_images = np.array(sample_hd_batch).astype(np.float32)[:, :, :, None]
        else:
            sample_lr_batch_images = np.array(sample_lr_batch)
            sample_hd_batch_images = np.array(sample_hd_batch)

        save_lr_hd_img(sample_hd_batch_images, sample_lr_batch_images, config)

        # Start the training
        # lmdb is a value multiplied in the loss function, check the article
        lmbd = 0.01
        with open("test_loss.txt", 'wb') as f:  # a file to the loss values
            for epoch in range(config.epoch):
                np.random.shuffle(self.data)
                batch_idxs = min(len(self.data), config.train_size) // config.batch_size

                for idx in range(0, batch_idxs):
                    batch_files = self.data[idx * config.batch_size:(idx + 1) * config.batch_size]

                    lr_batch, hd_batch = get_lr_hd_image(batch_files,
                                                         input_height=self.input_height,
                                                         input_width=self.input_width,
                                                         output_height=self.output_height,
                                                         output_width=self.output_width,
                                                         crop=self.crop,
                                                         grayscale=self.grayscale)

                    if self.grayscale:
                        lr_batch_images = np.array(lr_batch).astype(np.float32)[:, :, :, None]
                        hd_batch_images = np.array(hd_batch).astype(np.float32)[:, :, :, None]
                    else:
                        lr_batch_images = np.array(lr_batch).astype(np.float32)
                        hd_batch_images = np.array(hd_batch).astype(np.float32)

                    # Tricks to swap the label and to get soft labels, check the loss function
                    alpha = np.random.random_integers(70, 120, [self.batch_size]) / 100.0
                    shufle_label = np.random.binomial(1, 0.9)

                    # Update D network
                    _, errD_fake, errD_real, summary_str = self.sess.run([d_optim, self.d_loss_fake,
                                                                          self.d_loss_real, self.d_sum],
                                                                         feed_dict={self.lr_inputs: lr_batch_images,
                                                                                    self.hd_outputs: hd_batch_images,
                                                                                    self.is_normal_label: shufle_label,
                                                                                    self.is_training: True,
                                                                                    self.alpha: alpha,
                                                                                    self.lmbd: lmbd
                                                                                    })

                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, errG, summary_str = self.sess.run([g_optim, self.g_loss, self.g_sum],
                                                         feed_dict={self.lr_inputs: lr_batch_images,
                                                                    self.hd_outputs: hd_batch_images,
                                                                    self.is_normal_label: shufle_label,
                                                                    self.is_training: True,
                                                                    self.alpha: alpha,
                                                                    self.lmbd: lmbd
                                                                    })

                    self.writer.add_summary(summary_str, counter)

                    counter += 1
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (
                    epoch, config.epoch, idx, batch_idxs,
                    time.time() - start_time, errD_fake + errD_real, errG))

                    # After 100 steps we check the results in the valid set
                    if np.mod(counter, 100) == 1:
                        all_test_d_loss = []
                        all_test_g_loss = []
                        all_psnr = []

                        test_batch_idxs = len(self.test_files) // config.batch_size
                        for i in range(test_batch_idxs):
                            batch_files = self.test_files[i * config.batch_size:config.batch_size * (i + 1)]
                            sample_lr_batch, sample_hd_batch = get_lr_hd_image(batch_files,
                                                                               input_height=self.input_height,
                                                                               input_width=self.input_width,
                                                                               output_height=self.output_height,
                                                                               output_width=self.output_width,
                                                                               crop=self.crop,
                                                                               grayscale=self.grayscale)
                            if self.grayscale:
                                sample_lr_batch_images = np.array(sample_lr_batch).astype(np.float32)[:, :, :, None]
                                sample_hd_batch_images = np.array(sample_hd_batch).astype(np.float32)[:, :, :, None]
                            else:
                                sample_lr_batch_images = np.array(sample_lr_batch)
                                sample_hd_batch_images = np.array(sample_hd_batch)

                            samples, d_loss, g_loss = self.sess.run(
                                [self.sampler, self.d_loss, self.g_loss],
                                feed_dict={
                                    self.lr_inputs: sample_lr_batch_images,
                                    self.hd_outputs: sample_hd_batch_images,
                                    self.is_normal_label: 1.0,
                                    self.is_training: False,
                                    self.alpha: np.ones(self.batch_size),
                                    self.lmbd: lmbd
                                }
                            )

                            if i == 0:
                                save_images(samples[:self.sample_num], image_manifold_size(samples.shape[0]),
                                            './{}/test_gen_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))

                                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                            all_psnr.append(psnr(samples, sample_hd_batch_images))
                            all_test_d_loss.append(d_loss)
                            all_test_g_loss.append(g_loss)
                            l = float(test_batch_idxs)

                        f.write("d_loss: %.8f, g_loss: %.8f, psnr: %.8f\n"
                                % (math.fsum(all_test_d_loss) / l,
                                   math.fsum(all_test_g_loss) / l,
                                   math.fsum(all_psnr) / l)
                                )

                    # After 500 steps we save the model
                    if np.mod(counter, 500) == 1:
                        f.flush()
                        os.fsync(f)
                        self.save(config.checkpoint_dir, counter)

                # Decreasing lambda every epoch
                if lmbd > 0.005:
                    lmbd *= 0.995
                    print ('Decreased lambda to', lmbd)

            f.flush()
            os.fsync(f)
            self.save(config.checkpoint_dir, counter)

    def discriminator(self, image, reuse=False):
        """Discriminator network"""
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = conv2d(image, self.df_dim * 4, name='d_h0_conv')
            h0 = tf.nn.leaky_relu(tf.layers.max_pooling2d(h0, pool_size=[2, 2], strides=2))
            h0 = tf.layers.dropout(h0, 0.2)

            h1 = conv2d(h0, self.df_dim * 4, name='d_h1_conv')
            h1 = tf.nn.leaky_relu(tf.layers.max_pooling2d(h1, pool_size=[2, 2], strides=2))
            h1 = tf.layers.dropout(h1, 0.2)

            h2 = conv2d(h1, self.df_dim * 2, name='d_h2_conv')
            h2 = tf.nn.leaky_relu(tf.layers.max_pooling2d(h2, pool_size=[2, 2], strides=2))
            h2 = tf.layers.dropout(h2, 0.2)

            h3 = conv2d(h2, self.df_dim, name='d_h3_conv')
            h3 = tf.nn.leaky_relu(tf.layers.max_pooling2d(h3, pool_size=[2, 2], strides=2))
            h3 = tf.layers.dropout(h3, 0.2)

            h4 = tf.contrib.layers.flatten(h3)
            h4 = tf.layers.dense(h4, units=1024,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                 activation=tf.nn.leaky_relu)
            h4 = tf.layers.dropout(h4, 0.2)

            h5 = tf.layers.dense(h4, units=1, kernel_initializer=tf.random_normal_initializer(stddev=0.02))

            return tf.nn.sigmoid(h5), h5

    def generator(self, z, reuse=False):
        """Generator Network"""
        with tf.compat.v1.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()

            x = z

            s_h, s_w = self.output_height, self.output_width
            # Calculating the output dimension after deconv (x2, x4, x8)
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)  # 64
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)  # 32

            h0 = conv2d(x, self.dcv_frst_channel, k_h=3, k_w=3, name='g_h0')
            h0 = tf.contrib.layers.batch_norm(h0, is_training=self.is_training, scope='g_bn0')
            h0 = tf.nn.leaky_relu(h0)

            stn1 = SpatialTransformer(localization_net=ConvLocNet(name='stn_locnet1',  filters=(20, 20, 4500, 50)),
                                     output_size=(25, 25),
                                     input_shape=(25, 25, 512),
                                     batch_size=self.batch_size,
                                     num_channels=512,
                                     name='stn1')

            _h0 = stn1.apply(h0)

            h1, self.h1_w, self.h1_b = deconv2d(
                _h0, [self.batch_size, s_h4, s_w4, self.dcv_sec_channel], k_h=3, k_w=3, name='g_h1', with_w=True)
            h1 = tf.contrib.layers.batch_norm(h1, is_training=self.is_training, scope='g_bn1')
            h1 = tf.nn.leaky_relu(h1)

            stn2 = SpatialTransformer(localization_net=ConvLocNet(name='stn_locnet2', filters=(20, 20, 32000, 50)),
                                      output_size=(50, 50),
                                      input_shape=(50, 50, 256),
                                      batch_size=self.batch_size,
                                      num_channels=256,
                                      name='stn2')

            _h1 = stn2.apply(h1)

            h2, self.h2_w, self.h2_b = deconv2d(
                _h1, [self.batch_size, s_h2, s_w2, self.dcv_thr_channel], k_h=5, k_w=5, name='g_h2', with_w=True)
            h2 = tf.contrib.layers.batch_norm(h2, is_training=self.is_training, scope='g_bn2')
            h2 = tf.nn.leaky_relu(h2)

            h3, self.h4_w, self.h4_b = deconv2d(
                h2, [self.batch_size, s_h, s_w, self.dcv_four_channel], k_h=5, k_w=5, name='g_h3', with_w=True)
            h3 = tf.contrib.layers.batch_norm(h3, is_training=self.is_training, scope='g_bn3')
            h3 = tf.nn.leaky_relu(h3)

            h4 = conv2d(h3, self.c_dim, k_h=5, k_w=5, name='g_h4')

            return tf.nn.tanh(h4)

    def loss_function(self):
        """Loss function implementation, change if you want another loss function
        preserve the self.g_loos and self.d_loss
        """
        self.d_loss_real = tf.reduce_mean(
            self.sigmoid_cross_entropy_with_logits(self.D_logits, self.is_normal_label * self.alpha))

        self.d_loss_fake = tf.reduce_mean(
            self.sigmoid_cross_entropy_with_logits(self.D_logits_,
                                                   (1.0 - self.is_normal_label) * self.alpha + tf.zeros_like(self.D_)))

        entropy_loss_g = tf.reduce_mean(
            self.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        l2_loss_g = self.mse(self.G, self.hd_outputs, 'pixel_error')

        vgg_loss = self.mse(self.G_vgg, self.HD_vgg, 'vgg_error')

        self.g_loss = tf.add(l2_loss_g,
                             tf.subtract(tf.multiply(self.gamma, vgg_loss),
                                         tf.multiply(self.lmbd, entropy_loss_g)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

    def mse(self, targets, outputs, name):
        """Mean Square Error"""
        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(targets, outputs))), name=name)

    def sigmoid_cross_entropy_with_logits(self, x, y):
        """Sigmoid cross entropy with logits, could be use the tensorflow one directly"""
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

    def save_histogram_summary(self):
        """Saving tensorboard summary"""
        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)
        self.lr_inputs_sum = image_summary("lr_inputs", self.lr_inputs)
        self.hd_outputs_sum = image_summary("hd_outputs", self.hd_outputs)

    def save_loss_summary(self):
        """Saving tensorboard summary"""
        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
