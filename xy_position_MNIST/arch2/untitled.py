# TODO: add test and validation code

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import numpy as np
import vgg19
import utils
import cv2
import time

from constants import n_classes, n_inputs_100, n_inputs_200, n_inputs_300, learning_rate, batch_size, epochs, display_steps, dropout, test_examples, validation_examples, std_width, std_height, device

from funcs import init_all_sub_variables, defineVariables, init_all_common_variables, preActivation, activation
import extract_video_info
yt8m = extract_video_info.read_data_sets()


print "LEARNING RATE = {}".format(learning_rate)+", BATCH SIZE = {}".format(batch_size)+", EPOCHS = {}".format(epochs)


video_labels = tf.placeholder(tf.bool, shape=(None, n_classes))
encoder_inputs = tf.placeholder(tf.float32, shape=(None, None, 4096))
encoder_hidden_units = tf.placeholder(tf.float32)

phase_train = tf.placeholder(tf.bool, name='phase_train')
keep_prob = tf.placeholder(tf.float32)

def preProcessing(videos):
	feature_videos = []
	print "preprocesing videos", len(videos)

	with tf.device(device):
		vgg = vgg19.Vgg19()
		video_placeholder = tf.placeholder(tf.float32, [1, 224, 224, 3])
		with tf.name_scope("content_vgg"):
			vgg.build(video_placeholder)
		for video in videos:
			feature_video = []
			for frame in video:
				input_subgraph = sess.run(vgg.fc6, feed_dict={video_placeholder: [frame], keep_prob: dropout})
				feature_video.append(input_subgraph[0])
			feature_videos.append(np.array(feature_video, ndmin=2))
			print "updates features in video", len(feature_videos)
		np.array(feature_videos, ndmin=3).shape
	return np.array(feature_videos, ndmin=3)


weights = {
	'wih1': defineVariables([4*300, 24000], "wih1"),
	'wh1h2': defineVariables([24000, 24000], "wh1h2"),
	'wh4o': tf.Variable(tf.truncated_normal([24000, n_classes]), "wh4o")
}

biases = {
	'bi': defineVariables([24000], "bi"),
	'bh1' :defineVariables([24000], "bh1"),
	'bh4': tf.Variable(tf.truncated_normal([n_classes]), "bh4")
}

def main_network(encoder_inputs_embedded, encoder_hidden_units):
	with tf.device(device):
		# size : number of frames = encoder hidden state units
		# feature embedding....	
		# embeddings = tf.Variable(tf.random_uniform([size, encoder_hidden_units], -1.0, 1.0), dtype=tf.float32, name="embedding")
		# encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs, name = "embedded")

		# data encoding...
		fw_encoder_cell = LSTMCell(300)
		bw_encoder_cell = LSTMCell(300)
		print encoder_inputs_embedded.get_shape()
		(encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_encoder_cell, cell_bw=bw_encoder_cell, inputs=encoder_inputs_embedded, dtype=tf.float32)

		encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
		encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
		encoder_final_state = tf.concat((encoder_final_state_c, encoder_final_state_h), 1)

		# MLP for further classification task...
	   	with tf.variable_scope('main_net_layer1') as scope:
			# tf.summary.histogram("weights", weights['wih1'])
			# tf.summary.histogram("biases", biases['bi'])
			fc1 = preActivation(encoder_final_state, weights['wih1'], biases['bi'])
			fc1_out = activation(fc1)
			fc1_dropped = tf.nn.dropout(fc1_out, keep_prob, name="dropout")

		with tf.variable_scope('main_net_layer2') as scope:
			# tf.summary.histogram("weights", weights['wh1h2'])
			# tf.summary.histogram("biases", biases['bh1'])
			fc2 = preActivation(fc1_dropped, weights['wh1h2'], biases['bh1'])
			fc2_out = activation(fc2)
			fc2_dropped = tf.nn.dropout(fc2_out, keep_prob, name="dropout")

		with tf.variable_scope('main_net_layer3') as scope:
			# tf.summary.histogram("weights", weights['wh4o'])
			# tf.summary.histogram("biases", biases['bh4'])
			out = preActivation(fc2_dropped, weights['wh4o'], biases['bh4'])
			# out = activation(out)
	return out

pred = main_network(encoder_inputs, encoder_hidden_units)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=video_labels))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
tf.summary.scalar("cost", cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(tf.cast(video_labels, dtype=tf.float32), 1))

# correct_pred = tf.where(tf.less(tf.subtract(tf.nn.softmax(pred), tf.cast(video_labels, dtype=tf.float32)), 0.15), tf.nn.softmax(pred), tf.zeros_like(video_labels))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar("acc", accuracy)

merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

with tf.device(device):
	with tf.Session() as sess:
		start_time = time.time()
		sess.run(init)
		step = 1
		train_writer = tf.summary.FileWriter('./logs', sess.graph)
		while step <= epochs * 30 / batch_size:
		# batch_100 an array of video frames, video ids, video labels 
			print str(time.time() - start_time) + "sec"
			batch_100, batch_200, batch_300 = yt8m.train.next_batch(batch_size)
			enc_inputs = preProcessing(batch_300[0])
			print enc_inputs.shape
			sess.run(optimizer, feed_dict={video_labels: np.array(batch_200[2].tolist(), ndmin=2), encoder_inputs:enc_inputs, encoder_hidden_units: 300, keep_prob:0.5})


			if step % display_steps == 0:
				summary, loss = sess.run([merged, cost], feed_dict={video_labels: np.array(batch_300[2].tolist(), ndmin=2),  encoder_inputs:enc_inputs, encoder_hidden_units: 300, keep_prob:0.5})
				train_writer.add_summary(summary, step)
				print "loss: {}".format(loss)

	        step += 1

		print "Optimization Finished!"
		print "training time = {}".format(time.time() - start_time)
	pass
pass