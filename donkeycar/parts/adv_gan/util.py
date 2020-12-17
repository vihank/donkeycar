#!/usr/bin/env python3
'''
Created on Mon Dec 14 2020

@author: vkarnala
'''

import tensorflow as tf
import numpy as np
import os
from donkeycar.utils import *
import random

def get_adv_model_by_type(cfg, model_type):
    '''
    given the string model_type and the configuration settings in cfg
    create a Keras model and return it.
    '''
    if model_type is None:
        model_type = cfg.DEFAULT_ADV_MODEL_TYPE
    print("\"get_adv_model_by_type\" model Type is: {}".format(model_type))
    input_shape = (cfg.IMAGE_H, cfg.IMAGE_W, cfg.IMAGE_DEPTH)
    
    if model_type == "basic":
        from donkeycar.parts.adv_gan.models import Basic
        advGen = Basic(input_shape=input_shape)
    else:
        raise Exception("unknown model type: %s" % model_type)
    
    return advGen