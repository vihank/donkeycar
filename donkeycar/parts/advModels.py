#!/usr/bin/env python3
'''
Created on Mon Dec 14 2020

@author: vkarnala
'''



import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.python.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation,  LeakyReLU, Input, Dense,\
                         BatchNormalization, Conv2D, Conv2DTranspose, BatchNormalization, Cropping2D

if tf.__version__ == '1.13.1':
    from tensorflow import ConfigProto, Session

    # Override keras session to work around a bug in TF 1.13.1
    # Remove after we upgrade to TF 1.14 / TF 2.x.
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = Session(config=config)
    keras.backend.set_session(session)

# TODO add more model architectures and base structure that basic extends (look at keras.py)

class KerasGAN(object):
    '''
    Base class for Keras models that will provide steering and throttle to guide a car.
    '''
    def __init__(self):
        self.model = None
        self.optimizer = "adam"
 
    def load(self, model_path):
        self.model = keras.models.load_model(model_path, compile=False)

    def shutdown(self):
        pass

    def compile(self):
        pass

    def set_optimizer(self, optimizer_type, rate, decay, beta_1=0.9, beta_2=0.9, epsilon=None):
        if optimizer_type == "adam":
            self.model.optimizer = keras.optimizers.Adam(lr=rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)
        elif optimizer_type == "sgd":
            self.model.optimizer = keras.optimizers.SGD(lr=rate, decay=decay)
        elif optimizer_type == "rmsprop":
            self.model.optimizer = keras.optimizers.RMSprop(lr=rate, decay=decay, epsilon=epsilon)
        else:
            raise Exception("unknown optimizer type: %s" % optimizer_type)

    def __call__(self, input):
        return self.model(input)
    

class Basic(KerasGAN):
    def __init__(self, input_shape=(66, 200, 3), *args, **kwargs):
        super(Basic, self).__init__(*args, **kwargs)
        self.model = default_generator(input_shape=input_shape)
        self.compile()

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss='mse')

    def run(self, img_arr):
        output = self.model.predict(img_arr)
        return output

    

class G2(KerasGAN):
    def __init__(self, num_outputs=1, input_shape=(66, 200, 3), *args, **kwargs):
        super(G2, self).__init__(*args, **kwargs)
        self.model = model_2(num_outputs, input_shape)
        self.compile()

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss='mse')

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        output = self.model.predict(img_arr)
        return output
    
def model_2(num_outputs, input_shape):
    #this is the next model
    model = Model()
    return model

def default_generator(input_shape):
    input = Input(shape=input_shape)
    #c3s1-8
    G = Conv2D(filters=8, kernel_size=(3,3), padding='same')(input)
    #G = InstanceNormalization()(G)
    G = Activation('relu')(G)

    #d16
    G = Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), padding='same')(G)
    #G = InstanceNormalization()(G)
    G = Activation('relu')(G)

    #d32
    G = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding='same')(G)
    #G = InstanceNormalization()(G)
    G = Activation('relu')(G)

    residual = G
    #four r32 blocks
    for _ in range(4):
        G = Conv2D(filters=32, kernel_size=(3,3), padding='same')(G)
        G = BatchNormalization()(G)
        G = Activation('relu')(G)
        G = Conv2D(filters=32, kernel_size=(3,3), padding='same')(G)
        G = BatchNormalization()(G)
        G = layers.add([G, residual])
        residual = G

    #u16
    G = Conv2DTranspose(filters=16, kernel_size=(3,3), strides=(2,2), padding='same')(G)
    #G = InstanceNormalization()(G)
    G = Activation('relu')(G)

    #u8
    G = Conv2DTranspose(filters=8, kernel_size=(3,3), strides=(2,2), padding='same')(G)
    #G = InstanceNormalization()(G)
    G = Activation('relu')(G)

    #c3s1-3
    G = Conv2D(filters=3, kernel_size=(3,3), padding='same')(G)
    #G = InstanceNormalization()(G)
    G = Activation('relu')(G)

    G = Cropping2D(cropping=((1, 1), (0, 0)))(G)
    
    G = layers.add([G, input])

    model = Model(inputs=input, outputs=G)
    return model

def build_disc(inputs):

        D = Conv2D(32, 4, strides=(2,2))(inputs)
        D = LeakyReLU()(D)
        D = Dropout(0.4)(D)
        D = Conv2D(64, 4, strides=(2,2))(D)
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)
        D = Dropout(0.4)(D)
        D = Flatten()(D)
        D = Dense(64)(D)
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)
        D = Dense(1, activation='sigmoid')(D)
        return D
