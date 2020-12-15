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
from donkeycar.utils import *


def advTrainer(cfg, tub_names, model_in_path, model_out_path, model_type):
    '''
    trains a discriminator from given model and uses discriminator to
    train perturbation generator
    '''
    if model_type is None:
        model_type = cfg.DEFAULT_ADV_MODEL_TYPE

    if (model_in_path and not '.h5' == model_in_path[-3:]) or (model_out_path and not '.h5' == model_out_path[-3:]):
        raise Exception("Model filename should end with .h5")

    gen_records = {}
    opts = { 'cfg' : cfg}

    #place holders for some reason
    x_pl = tf.placeholder(tf.float32, [None, X.shape[1], X.shape[2], X.shape[3]]) # image placeholder
    t = tf.placeholder(tf.float32, [None, y.shape[-1]]) # target placeholder
    is_training = tf.placeholder(tf.bool, [])

    # gather target model
    # run time for the target model
    kl = get_model_by_type('dave2', cfg)
    if '.h5' in model_in_path:
        load_model(kl, model_in_path)
    else:
        print("ERR>> Unknown extension type on model file!!")
        return

    # gather generator model
    # train time for the generator model
    advGen = get_adv_model_by_type(model_type, cfg)
    advGen.compile()

    if cfg.PRINT_MODEL_SUMMARY:
        print(advGen.model.summary())
    
    opts['adv_gen'] = advGen
    opts['model_type'] = model_type

    # Gather data from disk
    extract_data_from_pickles(cfg, tub_names)

    records = gather_records(cfg, tub_names, opts, verbose=True)
    print('collating %d records ...' % (len(records)))
    collate_records(records, gen_records, opts)
    
    # targeted vs untargeted attack
    is_targeted = False

    thresh = 0.3

    train_gen = dataDivide(save_best, opts, gen_records, cfg.BATCH_SIZE, True)
    val_gen = dataDivide(save_best, opts, gen_records, cfg.BATCH_SIZE, False)

    total_records = len(gen_records)

    num_train = 0
    num_val = 0

    for key, _record in gen_records.items():
        if _record['train'] == True:
            num_train += 1
        else:
            num_val += 1

    print("train: %d, val: %d" % (num_train, num_val))
    print('total records: %d' %(total_records))
    
    if not continuous:
        steps_per_epoch = num_train // cfg.BATCH_SIZE
    else:
        steps_per_epoch = 100
    
    val_steps = num_val // cfg.BATCH_SIZE
    print('steps_per_epoch', steps_per_epoch)

    cfg.model_type = model_type

    #TODO change the training below to match donkeycar training if possible

    '''
    # init generator probably not needed with earlier load_model()
    generator(model_type)

	# generate perturbation, add to original input image(s)
    perturb = tf.clip_by_value(generator(x_pl, is_training), -thresh, thresh)
    x_perturbed = perturb + x_pl
    x_perturbed = tf.clip_by_value(x_perturbed, 0, 1)

	# pass real and perturbed image to discriminator and the target model
    d_real_angle, d_real_throt = discriminator(x_pl, is_training)
    d_fake_angle, d_fake_throt = discriminator(x_perturbed, is_training)
	
	# pass real and perturbed images to the model we are trying to fool
    f_real_angle, f_real_thrott = kl.run(x_pl)
    f_fake_angle, f_fake_thrott = kl.run(x_perturbed)

	
	# generate labels for discriminator (optionally smooth labels for stability)
    smooth = 0.0
    d_labels_real = tf.ones_like(d_real_probs) * (1 - smooth)
    d_labels_fake = tf.zeros_like(d_fake_probs)

	#-----------------------------------------------------------------------------------
	# LOSS DEFINITIONS
	# discriminator loss
    d_loss_real = tf.losses.mean_squared_error(predictions=d_real_probs, labels=d_labels_real)
    d_loss_fake = tf.losses.mean_squared_error(predictions=d_fake_probs, labels=d_labels_fake)
    d_loss = d_loss_real + d_loss_fake

	# generator loss
    g_loss_fake = tf.losses.mean_squared_error(predictions=d_fake_probs, labels=tf.ones_like(d_fake_probs))

	# perturbation loss (minimize overall perturbation)
    l_perturb = perturb_loss(perturb, thresh)

	# adversarial loss (encourage misclassification)
    l_adv = adv_loss(f_fake_probs, t, is_targeted)

	# weights for generator loss function
    alpha = 1.0
    beta = 5.0
    g_loss = l_adv + alpha*g_loss_fake + beta*l_perturb 

	# ----------------------------------------------------------------------------------
	# gather variables for training/restoring
    t_vars = tf.trainable_variables()
    f_vars = [var for var in t_vars if 'ModelC' in var.name]
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='g_weights')

	# define optimizers for discriminator and generator
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        d_opt = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)
        g_opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(g_loss, var_list=g_vars)

	# create saver objects for the target model, generator, and discriminator
	saver = tf.train.Saver(f_vars)
	g_saver = tf.train.Saver(g_vars)
	d_saver = tf.train.Saver(d_vars)

	init  = tf.global_variables_initializer()

	sess  = tf.Session()
	sess.run(init)

	# load the pretrained target model
    try:
        saver.restore(sess, "./weights/target_model/model.ckpt")
    except:
        print("make sure to train the target model first...")
        sys.exit(1)
        
    total_batches = int(X.shape[0] / batch_size)
    
    for epoch in range(0, epochs):
        X, y = shuffle(X, y)
        loss_D_sum = 0.0
        loss_G_fake_sum = 0.0
        loss_perturb_sum = 0.0
        loss_adv_sum = 0.0
        
        for i in range(total_batches):
            
            batch_x, batch_y = next_batch(X, y, i, batch_size)

			# if targeted, create one hot vectors of the target
            if is_targeted:
                targets = np.full((batch_y.shape[0],), target)
                batch_y = np.eye(y.shape[-1])[targets]

			# train the discriminator first n times
            for _ in range(1):
                _, loss_D_batch = sess.run([d_opt, d_loss], feed_dict={x_pl: batch_x, \
																	   is_training: True})

			# train the generator n times
            for _ in range(1):
                _, loss_G_fake_batch, loss_adv_batch, loss_perturb_batch = \
									sess.run([g_opt, g_loss_fake, l_adv, l_perturb], \
												feed_dict={x_pl: batch_x, \
														   t: batch_y, \
														   is_training: True})
            loss_D_sum += loss_D_batch
            loss_G_fake_sum += loss_G_fake_batch
            loss_perturb_sum += loss_perturb_batch
            loss_adv_sum += loss_adv_batch
            
        print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f, \
			    \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
				(epoch + 1, loss_D_sum/total_batches, loss_G_fake_sum/total_batches,
				loss_perturb_sum/total_batches, loss_adv_sum/total_batches))
        
        if epoch % 10 == 0:
            g_saver.save(sess, "weights/generator/gen.ckpt")
            d_saver.save(sess, "weights/discriminator/disc.ckpt")

	# evaluate the test set
	correct_prediction = tf.equal(tf.argmax(f_fake_probs, 1), tf.argmax(t, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	accs = []
	total_batches_test = int(X_test.shape[0] / batch_size)
    for i in range(total_batches_test):
        batch_x, batch_y = next_batch(X_test, y_test, i, batch_size)
        acc, x_pert = sess.run([accuracy, x_perturbed], feed_dict={x_pl: batch_x, t: batch_y, is_training: False})
        accs.append(acc)

	print('accuracy of test set: {}'.format(sum(accs) / len(accs)))

	print('finished training, saving weights')
	g_saver.save(sess, "weights/generator/gen.ckpt")
	d_saver.save(sess, "weights/discriminator/disc.ckpt")
    '''