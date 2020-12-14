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
from tensorflow.python.keras.utils import to_categorical
from donkeycar.parts.adv_gan.advgan_util import *
from donkeycar.utils import *


def advDriver(X, y, batch_size=128, thresh=0.3, target=-1):
	x_pl = tf.placeholder(tf.float32, [None, X.shape[1], X.shape[2], X.shape[3]]) # image placeholder
	t = tf.placeholder(tf.float32, [None, 10]) # target placeholder
	is_training = tf.placeholder(tf.bool, [])

	is_targeted = False
	if target in range(0, y.shape[-1]):
		is_targeted = True

	perturb = tf.clip_by_value(generator(x_pl, is_training), -thresh, thresh)
	x_perturbed = perturb + x_pl
	x_perturbed = tf.clip_by_value(x_perturbed, 0, 1)

	f = target_model()
	f_real_logits, f_real_probs = f.ModelC(x_pl)
	f_fake_logits, f_fake_probs = f.ModelC(x_perturbed)

	t_vars = tf.trainable_variables()
	f_vars = [var for var in t_vars if 'ModelC' in var.name]
	g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='g_weights')

	sess = tf.Session()

	f_saver = tf.train.Saver(f_vars)
	g_saver = tf.train.Saver(g_vars)
	f_saver.restore(sess, "./weights/target_model/model.ckpt")
	g_saver.restore(sess, tf.train.latest_checkpoint("./weights/generator/"))

	rawpert, pert, fake_l, real_l = sess.run([perturb, x_perturbed, f_fake_probs, f_real_probs], \
												feed_dict={x_pl: X[:32], \
														   is_training: False})
	print('LA: ' + str(np.argmax(y[:32], axis=1)))
	print('OG: ' + str(np.argmax(real_l, axis=1)))
	print('PB: ' + str(np.argmax(fake_l, axis=1)))

	correct_prediction = tf.equal(tf.argmax(f_fake_probs, 1), tf.argmax(t, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	accs = []
	total_batches_test = int(X.shape[0] / batch_size)
	for i in range(total_batches_test):
		batch_x, batch_y = next_batch(X, y, i, batch_size)

		if is_targeted:
			targets = np.full((batch_y.shape[0],), target)
			batch_y = np.eye(y.shape[-1])[targets]

		acc, fake_l, x_pert = sess.run([accuracy, f_fake_probs, x_perturbed], feed_dict={x_pl: batch_x, t: batch_y, is_training: False})
		accs.append(acc)

	print('accuracy of test set: {}'.format(sum(accs) / len(accs)))

	f, axarr = plt.subplots(2,2)
	axarr[0,0].imshow(np.squeeze(X[3]), cmap='Greys_r')
	axarr[0,1].imshow(np.squeeze(pert[3]), cmap='Greys_r')
	axarr[1,0].imshow(np.squeeze(X[4]), cmap='Greys_r')
	axarr[1,1].imshow(np.squeeze(pert[4]), cmap='Greys_r')
	plt.show()