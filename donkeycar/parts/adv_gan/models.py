#!/usr/bin/env python3
'''
Created on Mon Dec 14 2020

@author: vkarnala
'''
import os
import donkeycar as dk
import tensorflow as tf
import numpy as np
import os, sys
import random
from donkeycar.parts.adv_gan.util import *

# TODO add more model architectures and base structure that basic extends (look at keras.py)

class basic():
    def __init__(self, num_outputs=1, input_shape=(66, 200, 3), *args, **kwargs):
        self.model = default_model(num_outputs, input_shape)
        self.compile()

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss='mse')

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        output = self.model.predict(img_arr)
        return output

def default_model(num_outputs, input_shape):
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

    model = Model(input_shape, out)
    return model

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