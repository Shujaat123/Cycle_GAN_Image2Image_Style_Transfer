# import tensorflow as tf
import numpy as np
from utils import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def conv_module(x_in, name, filter_size, in_filters, out_filters, strides):
    with tf.variable_scope(name):
        n = filter_size * filter_size * in_filters
        kernel = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters], tf.float32,
                                           initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
        en = tf.nn.conv2d(x_in, spectral_norm(kernel), strides, padding='SAME')
        en_bias = tf.get_variable("bias", shape=[out_filters], dtype=tf.float32,
                                            initializer=tf.constant_initializer(value=0.0))
        en = tf.nn.bias_add(en, en_bias)
        # en = tf.layers.BatchNormalization(axis=-1, momentum=0.9, name='bn')(en)
        en = tf.nn.leaky_relu(en, alpha=0.2, name='lrelu')
        # en = tf.nn.relu(en)
        return en


def conv(x_in, name, filter_size, in_filters, out_filters, strides):
    with tf.variable_scope(name):
        n = filter_size * filter_size * in_filters
        kernel = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters], tf.float32,
                                           initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
        en = tf.nn.conv2d(x_in, spectral_norm(kernel), strides, padding='SAME')
        en_bias = tf.get_variable("bias", shape=[out_filters], dtype=tf.float32,
                                            initializer=tf.constant_initializer(value=0.0))
        en = tf.nn.bias_add(en, en_bias)
        return en


def discriminator(data, scope, training):
    with tf.variable_scope(scope + '_discriminator', reuse=tf.AUTO_REUSE):
        num_filters = 256
        # num_filters = 512
        conv1 = conv_module(data, 'conv1', filter_size=3, in_filters=1, out_filters=num_filters, strides=[1, 2, 2, 1])
        conv1 = tf.layers.batch_normalization(conv1, momentum=0.9, training=training, name='bn1')
        conv1 = conv_module(conv1, 'conv1_1', filter_size=1, in_filters=num_filters, out_filters=num_filters, strides=[1, 2, 2, 1])
        conv1 = tf.layers.batch_normalization(conv1, momentum=0.9, training=training, name='bn2')

        # pool1 = tf.nn.max_pool2d(conv1,  name='pool1', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv2 = conv_module(conv1, 'conv2', filter_size=3, in_filters=num_filters, out_filters=num_filters*2, strides=[1, 2, 2, 1])
        conv2 = tf.layers.batch_normalization(conv2, momentum=0.9, training=training, name='bn3')
        conv2 = conv_module(conv2, 'conv2_1', filter_size=1, in_filters=num_filters*2, out_filters=num_filters*2, strides=[1, 2, 2, 1])
        conv2 = tf.layers.batch_normalization(conv2, momentum=0.9, training=training, name='bn4')
        # pool1 = tf.nn.max_pool(batch1, name='pool2', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv3 = conv_module(conv2, 'conv3', filter_size=3, in_filters=num_filters*2, out_filters=num_filters*4, strides=[1, 2, 2, 1])
        conv3 = tf.layers.batch_normalization(conv3, momentum=0.9, training=training, name='bn5')
        conv3 = conv_module(conv3, 'conv3_1', filter_size=3, in_filters=num_filters*4, out_filters=num_filters*4, strides=[1, 2, 2, 1])
        conv3 = tf.layers.batch_normalization(conv3, momentum=0.9, training=training, name='bn6')

        conv4 = conv_module(conv3, 'conv4', filter_size=3, in_filters=num_filters*4, out_filters=1, strides=[1, 1, 1, 1])

        return conv4
