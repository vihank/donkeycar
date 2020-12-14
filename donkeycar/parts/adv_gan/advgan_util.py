#!/usr/bin/env python3
'''
Created on Mon Dec 14 2020

@author: vkarnala
'''

import tensorflow as tf
import numpy as np
import os

# helper function for convolution -> instance norm -> relu
def ConvInstNormRelu(x, filters, kernel_size=3, strides=1):
	Conv = tf.layers.conv2d(
						inputs=x,
						filters=filters,
						kernel_size=kernel_size,
						strides=strides,
						padding="same",
						activation=None)

	InstNorm = tf.contrib.layers.instance_norm(Conv)

	return tf.nn.relu(InstNorm)


# helper function for trans convolution -> instance norm -> relu
def TransConvInstNormRelu(x, filters, kernel_size=3, strides=2):
	TransConv = tf.layers.conv2d_transpose(
						inputs=x,
						filters=filters,
						kernel_size=kernel_size,
						strides=strides,
						padding="same",
						activation=None)

	InstNorm = tf.contrib.layers.instance_norm(TransConv)

	return tf.nn.relu(InstNorm)

# helper function for residual block of 2 convolutions with same num filters
# in the same style as ConvInstNormRelu
def ResBlock(x, training, filters=32, kernel_size=3, strides=1):
	conv1 = tf.layers.conv2d(
						inputs=x,
						filters=filters,
						kernel_size=kernel_size,
						strides=strides,
						padding="same",
						activation=None)

	conv1_norm = tf.layers.batch_normalization(conv1, training=training)

	conv1_relu = tf.nn.relu(conv1_norm)

	conv2 = tf.layers.conv2d(
						inputs=conv1_relu,
						filters=filters,
						kernel_size=kernel_size,
						strides=strides,
						padding="same",
						activation=None)

	conv2_norm = tf.layers.batch_normalization(conv2, training=training)


	return x + conv2_norm


def generator(x, training):
	with tf.variable_scope('g_weights', reuse=tf.AUTO_REUSE):
		# input_layer = tf.reshape(x, [-1, 28, 28, 1])

		# define first three conv + inst + relu layers
		c1 = ConvInstNormRelu(x, filters=8, kernel_size=3, strides=1)
		d1 = ConvInstNormRelu(c1, filters=16, kernel_size=3, strides=2)
		d2 = ConvInstNormRelu(d1, filters=32, kernel_size=3, strides=2)

		# define residual blocks
		rb1 = ResBlock(d2, training, filters=32)
		rb2 = ResBlock(rb1, training, filters=32)
		rb3 = ResBlock(rb2, training, filters=32)
		rb4 = ResBlock(rb3, training, filters=32)

		# upsample using conv transpose
		u1 = TransConvInstNormRelu(rb4, filters=16, kernel_size=3, strides=2)
		u2 = TransConvInstNormRelu(u1, filters=8, kernel_size=3, strides=2)

		# final layer block
		out = tf.layers.conv2d_transpose(
						inputs=u2,
						filters=x.get_shape()[-1].value, # or 3 if RGB image
						kernel_size=3,
						strides=1,
						padding="same",
						activation=None)

		# out = tf.contrib.layers.instance_norm(out)

		return tf.nn.tanh(out)


def discriminator(x, training):
	with tf.variable_scope('d_weights', reuse=tf.AUTO_REUSE):
		# input_layer = tf.reshape(x, [-1, 28, 28, 1])

		conv1 = tf.layers.conv2d(
							inputs=x,
							filters=8,
							kernel_size=4,
							strides=2,
							padding="valid",
							activation=None)
		conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)

		
		conv2 = tf.layers.conv2d(
							inputs=conv1,
							filters=16,
							kernel_size=4,
							strides=2,
							padding="valid",
							activation=None)

		in1 = tf.contrib.layers.instance_norm(conv2)
		conv2 = tf.nn.leaky_relu(in1, alpha=0.2)

		conv3 = tf.layers.conv2d(
							inputs=conv2,
							filters=32,
							kernel_size=4,
							strides=2,
							padding="valid",
							activation=None)

		#in2 = tf.contrib.layers.instance_norm(conv3)
		in2 = tf.contrib.layers.instance_norm(conv3)
		conv3 = tf.nn.leaky_relu(in2, alpha=0.2)
		flat = tf.layers.flatten(conv3)
		logits = tf.layers.dense(flat, 1)

		probs = tf.nn.sigmoid(logits)

		return logits, probs


# randomly shuffle a dataset 
def shuffle(X, Y):
	rands = random.sample(range(X.shape[0]),X.shape[0])
	return X[rands], Y[rands]


# get the next batch based on x, y, and the iteration (based on batch_size)
def next_batch(X, Y, i, batch_size):
	idx = i * batch_size
	idx_n = i * batch_size + batch_size
	return X[idx:idx_n], Y[idx:idx_n]


# loss function to encourage misclassification after perturbation
def adv_loss(preds, labels, is_targeted):
	real = tf.reduce_sum(labels * preds, 1)
	other = tf.reduce_max((1 - labels) * preds - (labels * 10000), 1)
	if is_targeted:
		return tf.reduce_sum(tf.maximum(0.0, other - real))
	return tf.reduce_sum(tf.maximum(0.0, real - other))


# loss function to influence the perturbation to be as close to 0 as possible
def perturb_loss(preds, thresh=0.3):
	zeros = tf.zeros((tf.shape(preds)[0]))
	return tf.reduce_mean(tf.maximum(zeros, tf.norm(tf.reshape(preds, (tf.shape(preds)[0], -1)), axis=1) - thresh))

class Target:
	def __init__(self, lr=0.001, epochs=50, n_input=28, n_classes=10, batch_size=16,\
					restore=0):
		self.lr = lr
		self.epochs = epochs
		self.n_input = 28
		self.n_classes = 10
		self.batch_size = batch_size
		self.restore = restore

		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	# randomly shuffle a dataset 
	def shuffle(self, X, Y):
		rands = random.sample(range(X.shape[0]),X.shape[0])
		return X[rands], Y[rands]

	# get the next batch based on x, y, and the iteration (based on batch_size)
	def next_batch(self, X, Y, i, batch_size):
		idx = i * batch_size
		idx_n = i * batch_size + batch_size
		return X[idx:idx_n], Y[idx:idx_n]


	# USAGE:
	# 		- encoder network for vae
	# PARAMS:
	#	x: input data sample
	#	h_hidden: LIST of num. neurons per hidden layer
	def ModelC(self, x):
		with tf.variable_scope('ModelC', reuse=tf.AUTO_REUSE):
			#input_layer = tf.reshape(x, [-1, 28, 28, 1])

			conv1 = tf.layers.conv2d(
								inputs=x,
								filters=32,
								kernel_size=3,
								padding="same",
								activation=tf.nn.relu)
			
			conv2 = tf.layers.conv2d(
								inputs=conv1,
								filters=32,
								kernel_size=3,
								padding="same",
								activation=tf.nn.relu)

			pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

			conv3 = tf.layers.conv2d(
								inputs=pool1,
								filters=64,
								kernel_size=3,
								padding="same",
								activation=tf.nn.relu)

			conv4 = tf.layers.conv2d(
								inputs=conv3,
								filters=64,
								kernel_size=3,
								padding="same",
								activation=tf.nn.relu)

			pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

			pool2_flatten = tf.contrib.layers.flatten(pool2)

			fc1 = tf.layers.dense(inputs=pool2_flatten, units=200, activation=tf.nn.relu)

			fc2 = tf.layers.dense(inputs=fc1, units=200, activation=tf.nn.relu)

			logits = tf.layers.dense(inputs=fc2, units=self.n_classes, activation=None)

			probs = tf.nn.softmax(logits)

			return logits, probs