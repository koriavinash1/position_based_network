import tensorflow as tf
from math import sqrt
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from filter_vis import put_kernels_on_grid
from constants import IS_POSITION_BASED, learning_rate, n_classes, n_input, batch_size, epochs, display_steps, test_examples, dropout, validation_examples, image_height, image_width
from additional_funcs import pre_processing, define_variable, activation, nonlinear, conv2d, maxpool2d, all_variables_arch2, save
import input_data
mnist = input_data.read_data_sets(one_hot=True, train_image_number=360000, test_image_number=1000)
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("./dataset", one_hot=True)



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
print "#############################################################################"


if IS_POSITION_BASED:
    x = tf.placeholder(tf.float32, shape=(None, n_input, 3))
else:
    x = tf.placeholder(tf.float32, shape=(None, n_input))

y = tf.placeholder(tf.float32, shape=(None, n_classes))

keep_prob = tf.placeholder(tf.float32)
phase_train = tf.placeholder(tf.bool, name='phase_train')

# some input arrays
nx, ny = (image_width, image_height)
xt = np.linspace(0, 1, nx) # use 1/56 to 1-1/56 for line spacing to take intersecting point to the center of image pixel
yt = np.linspace(0, 1, ny)
xpos, ypos = np.meshgrid(xt, yt)
xpos = np.array(xpos).flatten()
ypos = np.array(ypos).flatten()
# print len(xpos), len(ypos)


weights, biases = all_variables_arch2()



def main_network(x, weights, biases, keep_prob, phase_train):

    if IS_POSITION_BASED:
        x = tf.reshape(x, shape=[-1, 28, 28, 3])
    else:
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'], "conv1", phase_train)

    grid1 = put_kernels_on_grid(weights['wc1'], 8, 4)
    tf.summary.image("merged", grid1, max_outputs=1)
    pool1 = maxpool2d(conv1, "pool1")

    conv2 = conv2d(pool1, weights['wc2'], biases['bc2'], "conv2", phase_train)    
    pool2 = maxpool2d(conv2, "pool2")

    unroll = tf.reshape(pool2, [-1, weights['wfc1'].get_shape().as_list()[0]], name="unroll")
    
    fc1 = tf.add(tf.matmul(unroll, weights['wfc1']), biases['bfc1'])
    fc1_out = tf.nn.relu(fc1, name="relu_fc1")
    fc1_dropped = tf.nn.dropout(fc1_out, keep_prob, name="dropout1")

    fc2 = tf.add(tf.matmul(fc1_dropped, weights['wfc2']), biases['bfc2'])
    fc2_out = tf.nn.relu(fc2, name="relu_fc2")
    fc2_dropped = tf.nn.dropout(fc2_out, keep_prob, name="dropout2")

    out = tf.add(tf.matmul(fc2_dropped, weights['out']), biases['out'])
    return out, conv1, conv2


saver = tf.train.Saver()
pred, conv1, conv2 = main_network(x, weights, biases, keep_prob, phase_train)
    
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
tf.summary.scalar("cost", cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar("accuracy", accuracy)

merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    if IS_POSITION_BASED:
        saver.restore(sess, "./logs/model/position/model.ckpt")
    else:
        saver.restore(sess, "./logs/model/normal/model.ckpt")
    print("Model restored.")

    sess.run(init)
    step = 1
    if IS_POSITION_BASED:
        train_writer = tf.summary.FileWriter('./logs/position_train', sess.graph)
    else:
        train_writer = tf.summary.FileWriter('./logs/normal_train', sess.graph)
    while step <= epochs * 60000/ batch_size:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        train_image = pre_processing(batch_x, interpolation_flag = cv2.INTER_CUBIC, size=(28, 28))

        # contour_info = np.where(batch_x != 0, batch_x, -1)
        # contour_info = np.where(contour_info < 0, contour_info, 1)
        # contour_info = np.where(contour_info > 0, contour_info, 0)

        if IS_POSITION_BASED:
            input_batch =  np.swapaxes(np.swapaxes(np.vstack([[train_image],[[xpos,]*batch_size], [[ypos,]*batch_size]]), 0, 2), 0, 1)
        else:
            input_batch = train_image

        sess.run(optimizer, feed_dict={x: input_batch, y: batch_y, keep_prob: dropout, phase_train: True})
        
        if step % display_steps == 0:
            summary, loss, acc = sess.run([merged, cost, accuracy], feed_dict={x: input_batch, y: batch_y, keep_prob:1.0, phase_train:False})
            train_writer.add_summary(summary, step)

        if step*batch_size % 1000 == 0:
            print "TRAINING IMAGE= {:.1f}".format(step*batch_size)+", EPOCH= {:.5f}".format(step*batch_size//60000)+", loss= {:.6f}".format(loss) + ", Accuracy= {:.5f}".format(acc)

        if step % int(60000/batch_size) == 0:
            validation_images = pre_processing(mnist.validation.images[:validation_examples], interpolation_flag=cv2.INTER_CUBIC, size=(10, 10))
            # validation_images = mnist.test.images[validation_examples:test_examples + validation_examples]

            if IS_POSITION_BASED:
                validation_data = np.swapaxes(np.swapaxes(np.vstack([[validation_images],[[xpos,]*validation_examples], [[ypos,]*validation_examples]]), 0, 2), 0, 1)
            else:
                validation_data = validation_images

            loss, vacc = sess.run([cost, accuracy], feed_dict={x: validation_data, y: mnist.validation.labels[:validation_examples], keep_prob:1.0, phase_train: False})

            print "VALIDATION LOSS= {:.6f}".format(loss) + ", VALIDATION ACCURACY= {:.5f}".format(vacc)

            if vacc >0.9964:
                break
            elif vacc < 0.9780:
                learning_rate = 0.00001


        if step % 500 == 0:
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
    

    test_images = pre_processing(mnist.test.images[:test_examples], interpolation_flag= cv2.INTER_CUBIC, size=(10, 10))
    # test_images = mnist.test.images[:test_examples]

    # test_contours = np.where(mnist.test.images[:test_examples] != 0, mnist.test.images[:test_examples], -1)
    # test_contours = np.where(test_contours < 0, test_contours, 1)
    # test_contours = np.where(test_contours > 0, test_contours, 0)
    
    if IS_POSITION_BASED:
        test_data = np.swapaxes(np.swapaxes(np.vstack([[test_images],[[xpos,]*test_examples], [[ypos,]*test_examples]]), 0, 2), 0, 1)
    else:
        test_data = test_images
    
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: mnist.test.labels[:test_examples], keep_prob:1.0, phase_train: False})