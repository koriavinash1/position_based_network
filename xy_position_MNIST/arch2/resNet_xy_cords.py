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



if not os.path.exists("./weights"):
    os.mkdir("./weights")
if not os.path.exists("./biases"):
    os.mkdir("./biases")
if not os.path.exists("./logs"):
    os.mkdir("./logs")
if not os.path.exists("./logs/model"):
    os.mkdir("./logs/model")



print "IS_POSITION_BASED: ", IS_POSITION_BASED
print "LEARNING RATE = {}".format(learning_rate)+", BATCH SIZE = {}".format(batch_size)+", EPOCHS = {}".format(epochs)
print "##########################################################################"

if IS_POSITION_BASED:
    x = tf.placeholder(tf.float32, shape=(None, n_input, 3))
else:
    x = tf.placeholder(tf.float32, shape=(None, n_input))

y = tf.placeholder(tf.float32, shape=(None, n_classes))


# some input arrays
nx, ny = (28, 28)
xt = np.linspace(0, 1, nx)
yt = np.linspace(0, 1, ny)
xpos, ypos = np.meshgrid(xt, yt)
xpos = np.array(xpos).flatten()
ypos = np.array(ypos).flatten()
# print len(xpos), len(ypos)

weights, biases = all_variables_arch2()

def main_network(x, weights, biases):

    if IS_POSITION_BASED:
        x = tf.reshape(x, shape=[-1, 28, 28, 3])
    else:
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'], "conv1")

    grid1 = put_kernels_on_grid(weights['wc1'], 8, 4)
    tf.summary.image("merged", grid1, max_outputs=1)

    pool1 = maxpool2d(conv1, "pool1")

    conv2 = conv2d(pool1, weights['wc2'], biases['bc2'], "conv2")
    pool2 = maxpool2d(conv2, "pool2")

    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(pool2, [-1, weights['wfc1'].get_shape().as_list()[0]], "unroll")
    
    # fully connected layer 1
    fc1 = tf.add(tf.matmul(fc1, weights['wfc1']), biases['bfc1'])
    fc1 = tf.nn.relu(fc1, name="relu_fc1")
    
    # fully connected layer 2
    fc2 = tf.add(tf.matmul(fc1, weights['wfc2']), biases['bfc2'])
    fc2 = tf.nn.relu(fc2, name="relu_fc2")

    # Output, class prediction
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out

saver = tf.train.Saver()
pred = main_network(x, weights, biases)
    
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
tf.summary.scalar("cost", cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar("accuracy", accuracy)

merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    if IS_POSITION_BASED:
        train_writer = tf.summary.FileWriter('./logs/position_train', sess.graph)
    else:
        train_writer = tf.summary.FileWriter('./logs/normal_train', sess.graph)
    while step <= epochs * batch_size:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # batch_x = pre_processing(batch_x)
        # print len(batch_x)
        
        if IS_POSITION_BASED:
            input_batch =  np.swapaxes(np.swapaxes(np.vstack([[batch_x],[[xpos,]*batch_size], [[ypos,]*batch_size]]), 0, 2), 0, 1)
        else:
            input_batch = batch_x

        # print input_batch.shape

        sess.run(optimizer, feed_dict={x: input_batch, y: batch_y})
        
        if step % display_steps == 0:
            summary, loss, acc = sess.run([merged, cost, accuracy], feed_dict={x: input_batch, y: batch_y})
            train_writer.add_summary(summary, step)

            print "EPOCH= {:.1f}".format(step/batch_size)+", loss= {:.6f}".format(loss) + ", Accuracy= {:.5f}".format(acc)


        if step % 100 == 0:
            if IS_POSITION_BASED:
                save_path = saver.save(sess, "./logs/model/position/model.ckpt")
            else:
                save_path = saver.save(sess, "./logs/model/normal/model.ckpt")
            print "Model saved in file: %s" % save_path

        step += 1
    print "Optimization Finished!"






    print "saving all variables...."
    if IS_POSITION_BASED:
        save(sess.run(weights['wc1']), "./weights/position/wc1")
        save(sess.run(weights['wc2']), "./weights/position/wc2")
        save(sess.run(weights['wfc1']), "./weights/position/wfc1")
        save(sess.run(weights['wfc2']), "./weights/position/wfc2")
        save(sess.run(weights['out']), "./weights/position/out")
        save(sess.run(biases['bc1']), "./biases/position/bc1")
        save(sess.run(biases['bc2']), "./biases/position/bc2")
        save(sess.run(biases['bfc1']), "./biases/position/bfc1")
        save(sess.run(biases['bfc2']), "./biases/position/bfc2")
        save(sess.run(biases['out']), "./biases/position/out")
        save_path = saver.save(sess, "./logs/model/position/model.ckpt")
    
    else:
        save(sess.run(weights['wc1']), "./weights/normal/wc1")
        save(sess.run(weights['wc2']), "./weights/normal/wc2")
        save(sess.run(weights['wfc1']), "./weights/normal/wfc1")
        save(sess.run(weights['wfc2']), "./weights/normal/wfc2")
        save(sess.run(weights['out']), "./weights/normal/out")
        save(sess.run(biases['bc1']), "./biases/normal/bc1")
        save(sess.run(biases['bc2']), "./biases/normal/bc2")
        save(sess.run(biases['bfc1']), "./biases/normal/bfc1")
        save(sess.run(biases['bfc2']), "./biases/normal/bfc2")
        save(sess.run(biases['out']), "./biases/normal/out")
        save_path = saver.save(sess, "./logs/model/position/model.ckpt")

    print "Model saved in file: %s" % save_path
    

    # test_images = pre_processing(mnist.test.images[:test_examples])
    if IS_POSITION_BASED:
        test_images = np.swapaxes(np.swapaxes(np.vstack([[mnist.test.images[:test_examples]],[[xpos,]*test_examples], [[ypos,]*test_examples]]), 0, 2), 0, 1)
    else:
        test_images = mnist.test.images[:test_examples]
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_images, y: mnist.test.labels[:test_examples]})