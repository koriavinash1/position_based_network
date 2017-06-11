import tensorflow as tf
import numpy as np
import time
import math
import cv2
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from constants import n_classes, n_input 
from additional_funcs import pre_processing, activation, nonlinear, conv2d, maxpool2d

batch_size = 1
camport = 1

IS_POSITION_BASED = True

cam = cv2.VideoCapture(camport)
time.sleep(0.1)
# some input arrays
nx, ny = (28, 28)
xt = np.linspace(0, 1, nx)
yt = np.linspace(0, 1, ny)

xpos, ypos = np.meshgrid(xt, yt)
xpos = np.square(np.array(xpos, dtype="float32").flatten())
ypos = np.square(np.array(ypos, dtype="float32").flatten())
# print len(xpos), len(ypos)


if IS_POSITION_BASED:
    x = tf.placeholder(tf.float32, shape=(1, n_input, 3))
else:
    x = tf.placeholder(tf.float32, shape=(1, n_input))

# x = tf.Variable(input_batch, tf.float32, name="image_input")
phase_train = tf.placeholder(tf.bool, name='phase_train')


def cap_img():
    ret, im=cam.read()
    cv2.imshow("orig_image",im)
    cv2.waitKey(10)
    return im

def pre_processing(image, flag=0):
    # print "Pre-processing the image...."
    channel_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if flag == 1:
        ret, channel_1 = cv2.threshold(channel_1, 90, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow("test_image", channel_1)
        cv2.waitKey(50)
    resized = cv2.resize(channel_1, (28, 28), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("input_image.png", resized)
    unrolled = np.array(resized, dtype="float32").flatten()
    normalized = np.divide(unrolled, 255)
    return np.array(normalized, ndmin=2)

def find_contours(image):
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    for rect in rects:
        cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        subimages.append(roi)

    cv2.imshow("Resulting Image with Rectangular ROIs", image)
    cv2.waitKey()

def getActivations(sess, layer,stimuli):
    tstimg = np.array(stimuli, ndmin=2)
    if IS_POSITION_BASED:
        test_data = np.swapaxes(np.swapaxes(np.vstack([[tstimg],[[xpos,]*1], [[ypos,]*1]]), 0, 2), 0, 1)
    else:
        test_data = tstimg

    units = sess.run(layer,feed_dict={x: test_data ,phase_train:True})
    plotNNFilter(units)

def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(10, 10))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        # plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
    plt.show()

def define_variable(value, name): 
    return tf.Variable(value, name = name)

def load_weights():
	print "Loading weights and biases for network...."
	with tf.device("/cpu:0"):
	    if IS_POSITION_BASED:
	        weights = {
	            'wc1': define_variable(np.load("./weights/position/wc1.npy"), 'wc1'), 
	            'wc2': define_variable(np.load("./weights/position/wc2.npy"), 'wc2'),
	            'wfc1': define_variable(np.load("./weights/position/wfc1.npy"), 'wfc1'),
	            'wfc2': define_variable(np.load("./weights/position/wfc2.npy"), 'wfc2'), 
	            'out': define_variable(np.load("./weights/position/out.npy"), 'out')
	        }

	        biases = {
	            'bc1': define_variable(np.load("./biases/position/bc1.npy"), 'bc1'),
	            'bc2': define_variable(np.load("./biases/position/bc2.npy"), 'bc2'),
	            'bfc1': define_variable(np.load("./biases/position/bfc1.npy"), 'bfc1'),
	            'bfc2': define_variable(np.load("./biases/position/bfc2.npy"), 'bfc2'),
	            'out': define_variable(np.load("./biases/position/out.npy"), 'out')
	        }
	    else:
	        weights = {
	            'wc1': define_variable(np.load("./weights/normal/wc1.npy"), 'wc1'), 
	            'wc2': define_variable(np.load("./weights/normal/wc2.npy"), 'wc2'),
	            'wfc1': define_variable(np.load("./weights/normal/wfc1.npy"), 'wfc1'),
	            'wfc2': define_variable(np.load("./weights/normal/wfc2.npy"), 'wfc2'), 
	            'out': define_variable(np.load("./weights/normal/out.npy"), 'out')
	        }

	        biases = {
	            'bc1': define_variable(np.load("./biases/normal/bc1.npy"), 'bc1'),
	            'bc2': define_variable(np.load("./biases/normal/bc2.npy"), 'bc2'),
	            'bfc1': define_variable(np.load("./biases/normal/bfc1.npy"), 'bfc1'),
	            'bfc2': define_variable(np.load("./biases/normal/bfc2.npy"), 'bfc2'),
	            'out': define_variable(np.load("./biases/normal/out.npy"), 'out')
	        }
	return weights, biases

def main_network(x, weights, biases, phase_train):
    # Reshape input picture
    
    if IS_POSITION_BASED:
        x = tf.reshape(x, shape=[-1, 28, 28, 3])
    else:
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'], "conv1", phase_train)
    pool1 = maxpool2d(conv1, "pool1")

    conv2 = conv2d(pool1, weights['wc2'], biases['bc2'], "conv2", phase_train)
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
    return (out, conv1, conv2) 

def evaluate(batch_x):

	predict, conv1, conv2 = main_network(x, weights, biases, phase_train)
	correct_pred = tf.nn.softmax(predict)
	result = tf.argmax(correct_pred, 1)

	with tf.Session() as sess:
	    sess.run(init)

	    if IS_POSITION_BASED:
	        input_batch =  np.array(np.swapaxes(np.swapaxes(np.vstack([[batch_x],[[xpos,]*batch_size], [[ypos,]*batch_size]]), 0, 2), 0, 1), dtype="float")
	    else:
	        input_batch = np.array(batch_x, dtype="float")

	    _, result_out = sess.run([correct_pred, result], feed_dict={x: input_batch, phase_train:False})


	    # print "input: {}".format(input_batch.shape) + "  xpos: {}".format(np.array([xpos,]*batch_size, ndmin=2).shape)

	    # print "IMAGE UPLOADED IS:  {}".format(result_out)
	    print "FEATURE MAPS:-->"
	    getActivations(sess, conv1, batch_x)
    	# getActivations(sess, conv1, batch_x)
	return result_out

weights, biases = load_weights()
init = tf.global_variables_initializer()

print "IS_POSITION_BASED: ", IS_POSITION_BASED
subimages = []
load_int = input("Press 1 load an image or 2 to take new image or 3 for real time:  ")
load_str = "abcd123321"

if load_int == 1:
    while not os.path.exists(load_str):
    	load_str = input("Enter image path...  ")
    test_image = cv2.imread(load_str)
    batch_x = pre_processing(test_image)
    result_out = evaluate(batch_x)
    print "IMAGE UPLOADED IS:  {}".format(result_out)
elif load_int == 2:
    t = time.time()
    print "IMAGE WILL BE TAKEN AFTER 5sec."
    while time.time() - t < 10:
        cap_image = cap_img()
    test_image = cap_img()
    cam.release()
    batch_x = pre_processing(test_image, 1)
    result_out = evaluate(batch_x)
    # find_contours(test_image)
    print "IMAGE UPLOADED IS:  {}".format(result_out)
elif load_int == 3:
	while True:
		image = cap_img()
		batch_x = pre_processing(image, 1)
		result_out = evaluate(batch_x)
		cv2.putText(image, str(result_out[0]), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
		cv2.imshow("original image", image)
		# cv2.waitKey(1)
