#!/usr/bin/env python3
'''
Created on Mon Dec 14 2020

@author: vkarnala
'''

import tensorflow as tf
import numpy as np
import os
from donkeycar.utils import *

def get_adv_model_by_type(model_type):
	'''
	given the string model_type and the configuration settings in cfg
    create a Keras model and return it.
	'''
	from donkeycar.parts.adv_gan.models import basic
	
	if model_type is None:
		model_type = cfg.DEFAULT_ADV_MODEL_TYPE
	print("\"get_model_by_type\" model Type is: {}".format(model_type))
	input_shape = (cfg.IMAGE_H, cfg.IMAGE_W, cfg.IMAGE_DEPTH)
	
	if model_type == "basic":
		advGen = basic(num_outputs=1, input_shape=input_shape)
	else:
		raise Exception("unknown model type: %s" % model_type)
	
	return advGen

def dataDivide(save_best, opts, data, batch_size, isTrainSet=True, min_records_to_train=1000):
        
        num_records = len(data)

        while True:

            if isTrainSet and opts['continuous']:
                '''
                When continuous training, we look for new records after each epoch.
                This will add new records to the train and validation set.
                '''
                records = gather_records(cfg, tub_names, opts)
                if len(records) > num_records:
                    collate_records(records, gen_records, opts)
                    new_num_rec = len(data)
                    if new_num_rec > num_records:
                        print('picked up', new_num_rec - num_records, 'new records!')
                        num_records = new_num_rec 
                        save_best.reset_best()
                if num_records < min_records_to_train:
                    print("not enough records to train. need %d, have %d. waiting..." % (min_records_to_train, num_records))
                    time.sleep(10)
                    continue

            batch_data = []

            keys = list(data.keys())

            random.shuffle(keys)

            kl = opts['keras_pilot']

            if type(kl.model.output) is list:
                model_out_shape = (2, 1)
            else:
                model_out_shape = kl.model.output.shape

            if type(kl.model.input) is list:
                model_in_shape = (2, 1)
            else:    
                model_in_shape = kl.model.input.shape

            has_imu = type(kl) is KerasIMU
            has_bvh = type(kl) is KerasBehavioral
            img_out = type(kl) is KerasLatent
            loc_out = type(kl) is KerasLocalizer
            
            if img_out:
                import cv2

            for key in keys:

                if not key in data:
                    continue

                _record = data[key]

                if _record['train'] != isTrainSet:
                    continue

                if continuous:
                    #in continuous mode we need to handle files getting deleted
                    filename = _record['image_path']
                    if not os.path.exists(filename):
                        data.pop(key, None)
                        continue

                batch_data.append(_record)

                if len(batch_data) == batch_size:
                    inputs_img = []
                    inputs_imu = []
                    inputs_bvh = []
                    angles = []
                    throttles = []
                    out_img = []
                    out_loc = []
                    out = []

                    for record in batch_data:
                        #get image data if we don't already have it
                        if record['img_data'] is None:
                            filename = record['image_path']
                            
                            img_arr = load_scaled_image_arr(filename, cfg)

                            if img_arr is None:
                                break
                            
                            if aug:
                                img_arr = augment_image(img_arr)

                            if cfg.CACHE_IMAGES:
                                record['img_data'] = img_arr
                        else:
                            img_arr = record['img_data']
                            
                        if img_out:                            
                            rz_img_arr = cv2.resize(img_arr, (127, 127)) / 255.0
                            out_img.append(rz_img_arr[:,:,0].reshape((127, 127, 1)))

                        if loc_out:
                            out_loc.append(record['location'])
                            
                        if has_imu:
                            inputs_imu.append(record['imu_array'])
                        
                        if has_bvh:
                            inputs_bvh.append(record['behavior_arr'])

                        inputs_img.append(img_arr)
                        angles.append(record['angle'])
                        throttles.append(record['throttle'])
                        out.append([record['angle'], record['throttle']])

                    if img_arr is None:
                        continue

                    img_arr = np.array(inputs_img).reshape(batch_size,\
                        cfg.TARGET_H, cfg.TARGET_W, cfg.TARGET_D)

                    if has_imu:
                        X = [img_arr, np.array(inputs_imu)]
                    elif has_bvh:
                        X = [img_arr, np.array(inputs_bvh)]
                    else:
                        X = [img_arr]

                    if img_out:
                        y = [out_img, np.array(angles), np.array(throttles)]
                    elif out_loc:
                        y = [ np.array(angles), np.array(throttles), np.array(out_loc)]
                    elif model_out_shape[1] == 2:
                        y = [np.array([out]).reshape(batch_size, 2) ]
                    else:
                        y = [np.array(angles), np.array(throttles)]

                    yield X, y

                    batch_data = []
	


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