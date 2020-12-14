#!/usr/bin/env python3
'''
Created on Sun Nov 29 2020

@author: vkarnala
'''
import os
import donkeycar as dk
import tensorflow as tf
import numpy as np

import os, sys
import random

#TODO implement an adversarial method to train and infrerence

def advTrain(cfg, tub_names, model_in_path, model_out_path, model_type):
	from donkeycar.parts.adv_gan.advgan_train import advTrainer
	advTrainer(cfg, tub_names, model_in_path, model_out_path, model_type)

def advDrive():
	from donkeycar.parts.adv_gan.advgan_drive import advDriver
	advDriver()