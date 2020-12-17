#!/usr/bin/env python3
'''
Created on Sun Nov 29 2020

@author: vkarnala
'''
from donkeycar.utils import load_model

#TODO implement an adversarial method to train and infrerence

def advTrain(cfg, tub_names, model_in_path, model_out_path, model_type):
    from donkeycar.parts.adv_gan.advgan_train import advTrainer
    #model_in_path is the path to the existing driving model
    #model_out_path is the path to the output generator model
    #model_type is the type of generator defaults to the adv generator default in myconfig
    advTrainer(cfg, tub_names, model_in_path, model_out_path, model_type)

class advDrive():
    def __init__(self, cfg, adv):
        self.cfg = cfg
        self.advPath = adv
        load_model(self.advPath)
        
    def run(self, img_arr):
        return #TODO image output of gan taken from code below