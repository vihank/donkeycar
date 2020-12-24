#!/usr/bin/env python3
'''
Created on Mon Dec 14 2020

@author: vkarnala
'''


import random
import numpy as np
from donkeycar.parts.advmodels import build_disc
from donkeycar.utils import get_model_by_type, load_model, extract_data_from_pickles, gather_records, \
                            collate_records, load_scaled_image_arr, get_adv_model_by_type
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import to_categorical
import matplotlib.pyplot as plt
import time
import datetime


def advTrain(cfg, tub_names, model_in_path, model_out_path, model_type):
    '''
    trains a discriminator from given model and uses discriminator to
    train perturbation generator
    '''
    if model_type is None:
        model_type = cfg.ADV_DEFAULT_MODEL_TYPE
    
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

    optim = opt(cfg.ADV_OPTIMIZER, cfg.ADV_LR)
    kl_ang_out, _ = kl(advGen(inputs))
    stacked = Model(inputs=inputs, outputs=[advGen(inputs), disc(advGen(inputs)), kl_ang_out])
    stacked.compile(loss=[genLoss, keras.losses.binary_crossentropy, discLoss], optimizer=optim)

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
    batch_data = []
    batch_size = cfg.ADV_BATCH_SIZE
    for key in keys:
        if not key in gen_records:
            continue

        batch_data.append(gen_records[key])

    total_records = len(gen_records)
    print('total records: %d' %(total_records))

    num_batches = len(batch_data)//batch_size
    if len(batch_data) % batch_size != 0:
        num_batches += 1
    print('There are %d batches of data' % num_batches)

    #set values needed for training
    history = {}
    train_thresh = cfg.ADV_THRESH
    history['gen_loss'] = []
    history['d_loss'] = []
    history['t_acc'] = []
    history['t_loss'] = []
    missclassified = []
    Gx_batch = None
    x_batch = None
    advEarlyStop = cfg.ADV_EARLY_STOP
    if advEarlyStop:
        earlyStop = CustomEarlyStopping(min_delta=cfg.ADV_MIN_DELTA, patience=cfg.ADV_EARLY_PATIENCE)
        earlyStop.on_train_begin()
    start = time.time()

    #Start training
    for epoch in range(cfg.ADV_EPOCH):
        print('\n\n\n\n')
        print("Epoch " + str(epoch))
        batch_index = 0
        missclassified = []
        Gx_batch = None
        x_batch = None

        for batch in range(num_batches - 1):
            start = batch_size*batch_index
            end = batch_size*(batch_index+1)
            batches = get_batches(cfg, start, end, batch_data, advGen)
            train_D_on_batch(batches, disc)
            train_stacked_on_batch(batches, stacked, disc, kl)
            batch_index += 1

        #collect batch of data left after previous batches
        start = batch_size*batch_index
        end = len(batch_data)
        last_batch  = get_batches(cfg, start, end, batch_data, advGen)
        x_batch, Gx_batch, y_batch = last_batch

        d_loss = train_D_on_batch((x_batch, Gx_batch, None), disc)
        (_, hinge_loss, gan_loss, adv_loss) = train_stacked_on_batch(last_batch, stacked, disc, kl)
        
        #calculate accuracy of the target on generated images
        target_predictions = kl.model.predict_on_batch(Gx_batch)

        kl.compile()
        for ind in range(len(Gx_batch)):
            if y_batch[ind]-train_thresh <= target_predictions[0][ind] <= y_batch[ind]+train_thresh:
                missclassified.append(ind)

        target_acc = (batch_size-len(missclassified))/batch_size
        history['gen_loss'].append(gan_loss)
        history['d_loss'].append(d_loss)
        history['t_loss'].append(adv_loss)
        history['t_acc'].append(target_acc)
        

        print("Disciminator -- Loss: %f \nGenerator -- Loss: %f\nHinge Loss: %f\nTarget Loss: %f\tAccuracy:%.2f%%" %(d_loss, gan_loss, hinge_loss, adv_loss, target_acc*100.))

        if advEarlyStop:
            stop = earlyStop.on_epoch_end(epoch, advGen, history)
            
            if stop:
                earlyStop.on_train_end()
                break
    
    train_duration = time.time()-start
    print('Training completed in %s.' % str(datetime.timedelta(seconds=round(train_duration))))
    if cfg.ADV_SHOW_PLOT:
        plt.figure(1)

        # summarize history for loss
        plt.plot(history['gen_loss'])
        plt.plot(history['d_loss'])
        plt.plot(history['t_loss'])
        plt.plot(history['t_acc'])

        plt.title('GAN Model Training')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Generator Loss', 'Discriminator Loss', 'Target Loss', 'Target Accuracy'], loc='upper right')

        plt.savefig(model_out_path + '_loss_acc_%f.png' % (history['gen_loss'][-1]))
        plt.show()

        show_generated_images(Gx_batch[missclassified], 'Missclassified Images', model_out_path)
        show_generated_images(Gx_batch, 'Generated Images', model_out_path)
        show_generated_images(x_batch, 'Original Images', model_out_path)

        

    #save the model at the defined model_out path
    advGen.model.save(model_out_path)
    
class CustomEarlyStopping():
  """Stop training when a monitored quantity has stopped improving.

  Arguments:
      min_delta: Minimum change in the monitored quantity
          to qualify as an improvement, i.e. an absolute
          change of less than min_delta, will count as no
          improvement.
      patience: Number of epochs with no improvement
          after which training will be stopped.
      restore_best_weights: Whether to restore model weights from
          the epoch with the best value of the monitored quantity.
          If False, the model weights obtained at the last step of
          training are used.
  """

  def __init__(self,
               min_delta=0,
               patience=0,
               restore_best_weights=True):
    
    self.patience = patience
    self.min_delta = abs(min_delta)
    self.wait = 0
    self.stopped_epoch = 0
    self.restore_best_weights = restore_best_weights
    self.best_weights = None
    self.monitor_op = np.less


  def on_train_begin(self):
    # Allow instances to be re-used
    self.wait = 0
    self.stopped_epoch = 0
    self.best = np.Inf

  def on_epoch_end(self, epoch, gen, logs=None):
    current = logs['gen_loss'][-1]
    if current is None:
        return False
    if self.monitor_op(current - self.min_delta, self.best):
        self.best = current
        self.wait = 0
        if self.restore_best_weights:
            self.best_weights = gen.model.get_weights()
        return False
    else:
        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if self.restore_best_weights:
                print('Restoring model weights from the end of the best epoch.')
                gen.model.set_weights(self.best_weights)
            return True

  def on_train_end(self):
    if self.stopped_epoch > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


def show_generated_images(batch, name, path):
    plt.figure

    rows, columns = 5, 5
    _ , axs = plt.subplots(rows, columns)
    plt.title(name)
    cnt = 0
    for i in range(rows):
        for j in range(columns):
            axs[i,j].imshow(batch[cnt], interpolation='none')
            axs[i,j].axis('off')
            cnt += 1

    plt.savefig(path[:-3] + '_' + name + '.png')
    plt.show()
    

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
    #||G(x) - x||_2 - c, where c is user-defined. Here it is set to 0.3

def discLoss(y_true, y_pred):
    #custom binary cross entropy
    return K.max(y_pred, 0) - y_pred * y_true + K.log(1 + K.exp(-abs(y_pred)))

def custom_acc(y_true, y_pred):
    return binary_accuracy(K.round(y_true), K.round(y_pred))

def get_batches(cfg, start, end, batch_data, advGen):
    x_train = []
    y_batch = []
    
    for record in batch_data[start:end]:
        if record['img_data'] is None:
            filename = record['image_path']
            img_arr = load_scaled_image_arr(filename, cfg)

            if img_arr is None:
                break
            
            if cfg.CACHE_IMAGES:
                record['img_data'] = img_arr

        else:
            img_arr = record['img_data']
            
        x_train.append(img_arr)
        y_batch.append(record['angle'])

    x_batch = np.array(x_train)
    Gx_batch = np.array(advGen.model.predict_on_batch(x_batch))
    y_batch = np.array(y_batch)
    return x_batch, Gx_batch, y_batch