#!/usr/bin/env python3
'''
Created on Mon Dec 14 2020

@author: vkarnala
'''

import random
import numpy as np
from donkeycar.parts.adv_gan.util import get_adv_model_by_type
from donkeycar.parts.adv_gan.models import build_disc
from donkeycar.utils import get_model_by_type, load_model, extract_data_from_pickles, gather_records, collate_records, load_scaled_image_arr
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import binary_accuracy
from keras import backend as K
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from keras.utils import to_categorical


def advTrainer(cfg, tub_names, model_in_path, model_out_path, model_type):
    '''
    trains a discriminator from given model and uses discriminator to
    train perturbation generator
    '''
    if model_type is None:
        model_type = cfg.DEFAULT_ADV_MODEL_TYPE
    
    cfg.model_type = model_type

    if (model_in_path and not '.h5' == model_in_path[-3:]) or (model_out_path and not '.h5' == model_out_path[-3:]):
        raise Exception("Model filename should end with .h5")

    gen_records = {}
    opts = { 'cfg' : cfg}
    input_shape = (cfg.IMAGE_H, cfg.IMAGE_W, cfg.IMAGE_DEPTH)
    
    # gather target model
    # run time for the target model
    kl = get_model_by_type('dave2', cfg)
    
    kl.model.trainable = False

    # gather generator model
    # train time for the generator model
    advGen = get_adv_model_by_type(cfg, model_type)
    advGen.compile()

    optim = opt(cfg.DISC_OPTIMIZER, cfg.DISC_LR)
    inputs = Input(shape=input_shape)
    output = build_disc(advGen(inputs))
    disc = Model(inputs, output)
    disc.compile(loss=keras.losses.binary_crossentropy, optimizer=optim)

    optim = opt(cfg.STACKED_OPTIMIZER, cfg.STACKED_LR)
    kl_ang_out, _ = kl(advGen(inputs))
    stacked = Model(inputs=inputs, outputs=[advGen(inputs), disc(advGen(inputs)), kl_ang_out])
    stacked.compile(loss=[genLoss, keras.losses.binary_crossentropy, keras.losses.binary_crossentropy], optimizer=optim)

    if cfg.PRINT_MODEL_SUMMARY:
        print(advGen.model.summary())
        print(disc.summary())

    opts['adv_gen'] = advGen
    opts['model_type'] = model_type
    
    if '.h5' in model_in_path:
        load_model(kl, model_in_path)
    else:
        print("ERR>> Unknown extension type on model file!!")
        return

    # Gather data from disk
    extract_data_from_pickles(cfg, tub_names)

    records = gather_records(cfg, tub_names, opts, verbose=True)
    print('collating %d records ...' % (len(records)))
    collate_records(records, gen_records, opts)
    
    #shuffle and get ready for training
    keys = list(gen_records.keys())
    random.shuffle(keys)
    x_train = []
    y_train_thrott = []
    batch_data = []
    for key in keys:
        _record = gen_records[key]
        batch_data.append(_record)
        for record in batch_data:
            if record['img_data'] is None:
                filename = record['image_path']
                img_arr = load_scaled_image_arr(filename, cfg)

                if img_arr is None:
                    break

            else:
                img_arr = record['img_data']
                
            x_train.append(img_arr)
            y_train_thrott.append(record['angle'])
        batch_data = []
    
    total_records = len(gen_records)
    print('total records: %d' %(total_records))

    batch_size = cfg.ADV_BATCH_SIZE
    num_batches = len(x_train)//batch_size
    if len(x_train) % batch_size != 0:
        num_batches += 1
    print('There are %d batches of data' % num_batches)

    for epoch in range(cfg.ADV_EPOCH):
        print("Epoch " + str(epoch))
        batch_index = 0

        for batch in range(num_batches - 1):
            start = batch_size*batch_index
            end = batch_size*(batch_index+1)
            batches = get_batches(start, end, x_train, y_train_thrott, advGen)
            train_D_on_batch(batches, disc)
            train_stacked_on_batch(batches, stacked, disc, kl)
            batch_index += 1


        start = batch_size*batch_index
        end = len(x_train)
        x_batch, Gx_batch, y_batch = get_batches(start, end, x_train, y_train_thrott, advGen)

        (d_loss, d_acc) = train_D_on_batch((x_batch, Gx_batch, y_batch))
        (g_loss, hinge_loss, gan_loss, adv_loss) = train_stacked_on_batch((x_batch, Gx_batch, y_batch))

        target_acc = kl.test_on_batch(Gx_batch, to_categorical(y_batch))[1]
        target_predictions = kl.predict_on_batch(Gx_batch) #(96,2)

        misclassified = np.where(y_batch.reshape((len(x_train) % batch_size, )) != np.argmax(target_predictions, axis=1))[0]
        print(np.array(misclassified).shape)
        print(misclassified)

        print("Discriminator -- Loss:%f\tAccuracy:%.2f%%\nGenerator -- Loss:%f\nHinge Loss: %f\nTarget Loss: %f\tAccuracy:%.2f%%" %(d_loss, d_acc*100., gan_loss, hinge_loss, adv_loss, target_acc*100.))


def train_D_on_batch(batches, disc):
    x_batch, Gx_batch, _ = batches

    #for each batch:
        #predict noise on generator: G(z) = batch of fake images
        #train fake images on discriminator: D(G(z)) = update D params per D's classification for fake images
        #train real images on disciminator: D(x) = update D params per classification for real images

    #Update D params
    disc.trainable = True
    d_loss_real = disc.train_on_batch(x_batch, np.random.uniform(0.9, 1.0, size=(len(x_batch), 1)) ) #real=1, positive label smoothing
    d_loss_fake = disc.train_on_batch(Gx_batch, np.zeros((len(Gx_batch), 1)) ) #fake=0
    d_loss = 0.5*np.add(d_loss_real, d_loss_fake)

    return d_loss #(loss, accuracy) tuple


def train_stacked_on_batch(batches, stacked, disc, kl):
    x_batch, _, y_batch = batches
    flipped_y_batch = 1.-y_batch

    #for each batch:
        #train fake images on discriminator: D(G(z)) = update G params per D's classification for fake images

    #Update only G params
    disc.trainable = False
    kl.model.trainable = False
    stacked_loss = stacked.train_on_batch(x_batch, [x_batch, np.ones((len(x_batch), 1)), to_categorical(flipped_y_batch)] )
    #stacked_loss = self.stacked.train_on_batch(x_batch, [x_batch, np.ones((len(x_batch), 1)), to_categorical(y_batch)] )
    #input to full GAN is original image
    #output 1 label for generated image is original image
    #output 2 label for discriminator classification is real/fake; G wants D to mark these as real=1
    #output 3 label for target classification is 1/3; g wants to flip these so 1=1 and 3=0
    return stacked_loss #(total loss, hinge loss, gan loss, adv loss) tuple

def opt(cfg_opt, cfg_lr):
    if cfg_opt == 'adam':
        opt = Adam(cfg_lr)
    else:
        opt = SGD(cfg_lr)
    return opt

def genLoss(y_true, y_pred):
    return K.mean(K.maximum(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)) - 0.3, 0), axis=-1)

def custom_acc(y_true, y_pred):
    return binary_accuracy(K.round(y_true), K.round(y_pred))

def get_batches(start, end, x_train, y_train_thrott, advGen):
        x_batch = np.array(x_train[start:end])
        Gx_batch = np.array(advGen.predict_on_batch(x_batch))
        y_batch = np.array(y_train_thrott[start:end])
        return x_batch, Gx_batch, y_batch