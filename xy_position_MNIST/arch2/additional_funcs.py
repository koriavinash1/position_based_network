import tensorflow as tf
import cv2
import numpy as np

from constants import IS_POSITION_BASED, learning_rate, n_classes, n_input, batch_size, epochs, display_steps, test_examples

def pre_processing(images, interpolation_method, size):
    processed_images = []
    for image in images:
        channel_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        intermidiate_resize = cv2.resize(channel_1, size, interpolation= interpolation_method)
        resized = cv2.resize(intermidiate_resize, (28, 28), interpolation = cv2.INTER_AREA)
        unrolled = np.array(resized, dtype="float").flatten()
        normalized = np.divide(unrolled, 255)
        processed_images.append(normalized)
        return processed_images

def batch_norm(x, n_out, phase_train):
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def define_variable(shape, name): 
    initializer = tf.contrib.layers.variance_scaling_initializer()
    return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)

def activation(x, w, b):
    return tf.add(tf.matmul(x, w), b)

def nonlinear(x):
    return tf.nn.relu(x)

def conv2d(x, W, b, name, phase_train):
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name = name)
    x = tf.nn.bias_add(x, b)
    conv = tf.nn.relu(x)
    # norm = batch_norm(conv, W.get_shape().as_list()[3], phase_train)
    return tf.nn.relu(x)

def maxpool2d(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = name)

def all_variables_arch2():
    with tf.device("/cpu:0"):
        # feature map dimension 14*14
        weights = {
            'wc2': define_variable([5, 5, 32, 64], 'wc2'),
            'wfc1': define_variable([7*7*64, 2048], 'wfc1'),
            'wfc2': define_variable([2048, 1024], 'wfc2'), 
            'out': tf.Variable(tf.truncated_normal([1024, n_classes]), 'out')
        }
        tf.summary.histogram("wc2", weights['wc2'])
        tf.summary.histogram("wfc1", weights['wfc1'])
        tf.summary.histogram("wfc2", weights['wfc2'])
        tf.summary.histogram("out", weights['out'])

        if IS_POSITION_BASED:
            weights['wc1'] = define_variable([5, 5, 3, 32], "wc1")
        else:
            weights['wc1'] = define_variable([5, 5, 1, 32], "wc1")
        tf.summary.histogram("wc1", weights['wc1'])

        biases = {
            'bc1': define_variable([32], 'bc1'),
            'bc2': define_variable([64], 'bc2'),
            'bfc1': define_variable([2048], 'bfc1'),
            'bfc2': define_variable([1024], 'bfc2'),
            'out': tf.Variable(tf.truncated_normal([n_classes]), 'out')
        }
        tf.summary.histogram("bc1", biases['bc1'])
        tf.summary.histogram("bc2", biases['bc2'])
        tf.summary.histogram("bfc1", biases['bfc1'])
        tf.summary.histogram("bfc2", biases['bfc2'])
        tf.summary.histogram("out", biases['out'])    
    return (weights, biases)
    
def save(list2save, directory):
	np.save(directory, list2save)
	pass
