import tensorflow.contrib.slim as slim
import tensorflow as tf
from dcgan import DCGAN
import numpy as np
import pprint
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam [0.001]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.9]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 20, "The size of batch images [64]")
flags.DEFINE_integer("sample_num", 20, "The size of the a test sample [64]")
flags.DEFINE_integer("input_height", 25, "The size of lr image to use (will be center cropped). [16]")
flags.DEFINE_integer("input_width", 25, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 200, "The size of the hd images to produce [128]")
flags.DEFINE_integer("output_width", 200, "The size of the hd images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("predict", False, "True for super resolve a new lr images sets. False otherwise [False]")
flags.DEFINE_string("predict_dataset", "", "The path list of the images to predict [list.txt]")
flags.DEFINE_boolean("crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
FLAGS = flags.FLAGS


def main(_):
    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    #run_config = tf.compat.v1.ConfigProto
    #run_config.gpu_options.allow_growth = True
    #run_config.gpu_options.visible_device_list = ''
    with tf.compat.v1.Session() as sess:
        dcgan = DCGAN(
            sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            sample_num=FLAGS.sample_num,
            dataset_name=FLAGS.dataset,
            input_fname_pattern=FLAGS.input_fname_pattern,
            crop=FLAGS.crop,
            checkpoint_dir=FLAGS.checkpoint_dir,
            data_dir=FLAGS.data_dir)

        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)

        if FLAGS.train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")
            if FLAGS.predict:
                dcgan.predict(FLAGS.predict_dataset)
            else:
                dcgan.test()


if __name__ == '__main__':
    tf.compat.v1.app.run()
