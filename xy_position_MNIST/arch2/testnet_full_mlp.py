import tensorflow as tf
from math import sqrt
import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from filter_vis import put_kernels_on_grid
from constants import IS_POSITION_BASED, learning_rate, n_classes, n_input, batch_size, epochs, display_steps, test_examples
from additional_funcs import pre_processing, define_variable, activation, nonlinear, conv2d, maxpool2d, all_variables_arch2, save
# import input_data
# mnist = input_data.read_data_sets(one_hot=True, train_image_number=360000, test_image_number=1000)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./dataset", one_hot=True)

print "IS_POSITION_BASED: ", IS_POSITION_BASED
print "LEARNING RATE = {}".format(learning_rate)+", BATCH SIZE = {}".format(batch_size)+", EPOCHS = {}".format(epochs)
print "##########################################################################"

x = tf.placeholder(tf.float32, shape=(None, n_input))
y = tf.placeholder(tf.float32, shape=(None, n_classes))

nx, ny = (28, 28)
xt = np.linspace(0, 1, nx)
yt = np.linspace(0, 1, ny)
xpos, ypos = np.meshgrid(xt, yt)
xpos = np.array(xpos).flatten()
ypos = np.array(ypos).flatten()

weights={
	'wfc1': tf.Variable(tf.truncated_normal([n_input, 2048]), 'wfc1'),
	'wfc2': tf.Variable(tf.truncated_normal([2048, 1024]), 'wfc2'),
	'wfc3': tf.Variable(tf.truncated_normal([1024, 1024]), 'wfc3'), 
	'wfc4': tf.Variable(tf.truncated_normal([1024, 1024]), 'wfc4'),
	'out': tf.Variable(tf.truncated_normal([1024, n_classes]), 'out')
	}
biases = {
	'bfc1': tf.Variable(tf.truncated_normal([2048]), 'bfc1'),
	'bfc2': tf.Variable(tf.truncated_normal([1024]), 'bfc2'),
	'bfc3': tf.Variable(tf.truncated_normal([1024]), 'bfc3'), 
	'bfc4': tf.Variable(tf.truncated_normal([1024]), 'bfc4'),
	'out': tf.Variable(tf.truncated_normal([n_classes]), 'out')
}

def main_network(x, weights, biases):
    fc1 = tf.add(tf.matmul(x, weights['wfc1']), biases['bfc1'])
    fc1 = tf.nn.relu(fc1, name="relu_fc1")
    
    fc2 = tf.add(tf.matmul(fc1, weights['wfc2']), biases['bfc2'])
    fc2 = tf.nn.relu(fc2, name="relu_fc2")

    fc3 = tf.add(tf.matmul(fc2, weights['wfc3']), biases['bfc3'])
    fc3 = tf.nn.relu(fc2, name="relu_fc3")

    fc4 = tf.add(tf.matmul(fc3, weights['wfc4']), biases['bfc4'])
    fc4 = tf.nn.relu(fc4, name="relu_fc4")

    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out

pred = main_network(x, weights, biases)
    
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step <= epochs * 60000/ batch_size:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # input_batch =  np.swapaxes(np.swapaxes(np.vstack([[batch_x],[[xpos,]*batch_size], [[ypos,]*batch_size]]), 0, 2), 0, 1)
        input_batch =  np.multiply(np.multiply([xpos,]*batch_size, [ypos,]*batch_size), batch_x)
        sess.run(optimizer, feed_dict={x: input_batch, y: batch_y})

        if step % display_steps == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x: input_batch, y: batch_y})
            print "EPOCH= {:.1f}".format(step/batch_size)+", loss= {:.6f}".format(loss) + ", Accuracy= {:.5f}".format(acc)

        step += 1
    print "Optimization Finished!"
    test_data = np.multiply(np.multiply([xpos,]*test_examples, [ypos,]*test_examples), mnist.test.images[:test_examples])
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: mnist.test.labels[:test_examples]})