# import tensorflow as tf
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def conv_module(x_in, name, filter_size, in_filters, out_filters, strides, training):
    with tf.variable_scope(name):
        n = filter_size * filter_size * in_filters
        kernel = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters], tf.float32,
                                           initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
        en = tf.nn.conv2d(x_in, kernel, strides, padding='SAME')
        en_bias = tf.get_variable("bias", shape=[out_filters], dtype=tf.float32,
                                            initializer=tf.constant_initializer(value=0.0))
        en = tf.nn.bias_add(en, en_bias)
        en = tf.layers.batch_normalization(en, momentum=0.9, training=training, name='bn')
        en = tf.nn.relu(en)
        return en


def conv(x_in, name, filter_size, in_filters, out_filters, strides):
    with tf.variable_scope(name):
        n = filter_size * filter_size * in_filters
        kernel = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters], tf.float32,
                                           initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
        en = tf.nn.conv2d(x_in, kernel, strides, padding='SAME')
        en_bias = tf.get_variable("bias", shape=[out_filters], dtype=tf.float32,
                                            initializer=tf.constant_initializer(value=0.0))
        en = tf.nn.bias_add(en, en_bias)
        return en


def generator(data, phase_in, scope):
    with tf.variable_scope(scope + '_generator', reuse=tf.AUTO_REUSE):
        num_filters = 8
        # num_filters = 64
        encoder1_1 = conv_module(data, 'en1_1', filter_size=3, in_filters=1, out_filters=num_filters, strides=[1, 1, 1, 1],
                                     training=phase_in)
        encoder1_2 = conv_module(encoder1_1, 'en1_2', filter_size=3, in_filters=num_filters, out_filters=num_filters,
                                 strides=[1, 1, 1, 1], training=phase_in)

        encoder2_1 = tf.nn.max_pool(encoder1_2, name='en2_1', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                      padding='SAME')
        encoder2_2 = conv_module(encoder2_1, 'en2_2', filter_size=3, in_filters=num_filters, out_filters=num_filters*2,
                                 strides=[1, 1, 1, 1], training=phase_in)
        encoder2_3 = conv_module(encoder2_2, 'en2_3', filter_size=3, in_filters=num_filters*2, out_filters=num_filters*2,
                                 strides=[1, 1, 1, 1], training=phase_in)

        encoder3_1 = tf.nn.max_pool(encoder2_3, name='en3_1', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                      padding='SAME')
        encoder3_2 = conv_module(encoder3_1, 'en3_2', filter_size=3, in_filters=num_filters*2, out_filters=num_filters*4,
                                 strides=[1, 1, 1, 1], training=phase_in)
        encoder3_3 = conv_module(encoder3_2, 'en3_3', filter_size=3, in_filters=num_filters*4, out_filters=num_filters*4,
                                 strides=[1, 1, 1, 1], training=phase_in)

        encoder4_1 = tf.nn.max_pool(encoder3_3, name='en4_1', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                      padding='SAME')
        encoder4_2 = conv_module(encoder4_1, 'en4_2', filter_size=3, in_filters=num_filters*4, out_filters=num_filters*8,
                                 strides=[1, 1, 1, 1], training=phase_in)
        encoder4_3 = conv_module(encoder4_2, 'en4_3', filter_size=3, in_filters=num_filters*8, out_filters=num_filters*8,
                                 strides=[1, 1, 1, 1], training=phase_in)
        # encoder4_4 = tf.nn.dropout(encoder4_3, 0.4, name="drop_4_1")

        encoder5_1 = tf.nn.max_pool(encoder4_3, name='en5_1', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                      padding='SAME')
        encoder5_2 = conv_module(encoder5_1, 'en5_2', filter_size=3, in_filters=num_filters*8, out_filters=num_filters*16,
                                 strides=[1, 1, 1, 1], training=phase_in)
        encoder5_3 = conv_module(encoder5_2, 'en5_3', filter_size=3, in_filters=num_filters*16, out_filters=num_filters*8,
                                 strides=[1, 1, 1, 1], training=phase_in)

        decoder4_1 = tf.image.resize_bilinear(encoder5_3, name='de4_1', size=[32, 32])
        decoder4_1_2 = tf.concat([decoder4_1, encoder4_3], 3, name='de4_1_2')
        decoder4_2 = conv_module(decoder4_1_2, 'de4_2', filter_size=3, in_filters=num_filters*16, out_filters=num_filters*8,
                                 strides=[1, 1, 1, 1], training=phase_in)
        decoder4_3 = conv_module(decoder4_2, 'de4_3', filter_size=3, in_filters=num_filters*8, out_filters=num_filters*4,
                                 strides=[1, 1, 1, 1], training=phase_in)

        decoder3_1 = tf.image.resize_bilinear(decoder4_3, name='de3_1', size=[64, 64])
        decoder3_1_2 = tf.concat([decoder3_1, encoder3_3], 3, name='de3_1_2')
        decoder3_2 = conv_module(decoder3_1_2, 'de3_2', filter_size=3, in_filters=num_filters*8, out_filters=num_filters*4,
                                 strides=[1, 1, 1, 1], training=phase_in)
        decoder3_3 = conv_module(decoder3_2, 'de3_3', filter_size=3, in_filters=num_filters*4, out_filters=num_filters*2,
                                 strides=[1, 1, 1, 1], training=phase_in)

        decoder2_1 = tf.image.resize_bilinear(decoder3_3, name='de2_1', size=[128, 128])
        decoder2_1_2 = tf.concat([decoder2_1, encoder2_3], 3, name='de2_1_2')
        decoder2_2 = conv_module(decoder2_1_2, 'de2_2', filter_size=3, in_filters=num_filters*4, out_filters=num_filters*2,
                                 strides=[1, 1, 1, 1], training=phase_in)
        decoder2_3 = conv_module(decoder2_2, 'de2_3', filter_size=3, in_filters=num_filters*2, out_filters=num_filters,
                                 strides=[1, 1, 1, 1], training=phase_in)

        decoder1_1 = tf.image.resize_bilinear(decoder2_3, name='de1_1', size=[256, 256])
        decoder1_1_2 = tf.concat([decoder1_1, encoder1_2], 3, name='de1_1_2')
        decoder1_2 = conv_module(decoder1_1_2, 'de1_2', filter_size=3, in_filters=num_filters*2, out_filters=num_filters*2,
                                 strides=[1, 1, 1, 1], training=phase_in)
        decoder1_3 = conv_module(decoder1_2, 'de1_3', filter_size=3, in_filters=num_filters, out_filters=num_filters,
                                 strides=[1, 1, 1, 1], training=phase_in)
        decoder1_3_2 = tf.concat([decoder1_3], 3, name='de1_1_2')
        output_layer = conv(decoder1_3_2, 'output_layer', filter_size=1, in_filters=num_filters, out_filters=1,
                            strides=[1, 1, 1, 1])
        output_layer = tf.nn.sigmoid(output_layer, name='out')

        return output_layer


def Adaptive_PSF(data, phase_in, scope):
    with tf.variable_scope(scope + '_generator', reuse=tf.AUTO_REUSE):
        num_filters = 1

        encoder1_1 = conv_module(data, 'en1_1', filter_size=7, in_filters=1, out_filters=num_filters, strides=[1, 1, 1, 1],
                                     training=phase_in)

        output_layer = tf.nn.sigmoid(encoder1_1, name='out')

        return output_layer