#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car

Usage:
    advmanage.py (drive) [--model=<model>] [--js] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer|latent|dave2)] [--meta=<key:value> ...] [--myconfig=<filename>] [--adv]
    advmanage.py (train) [--tub=<tub1,tub2,..tubn>] [--file=<file> ...] (--model=<model>) [--transfer=<model>] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer|dave2)] [--continuous] [--aug] [--myconfig=<filename>]


Options:
    -h --help               Show this screen.
    --js                    Use physical joystick.
    -f --file=<file>        A text file containing paths to tub files, one per line. Option may be used more than once.
    --meta=<key:value>      Key/Value strings describing describing a piece of meta data about this drive. Option may be used more than once.
    --myconfig=filename     Specify myconfig file to use. [default: myconfig.py]
"""
import os
import time

from docopt import docopt

import donkeycar as dk

#import parts
from donkeycar.parts.transform import TriggeredCallback, DelayedTrigger
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.controller import LocalWebController
from donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.parts.file_watcher import FileWatcher
from donkeycar.utils import *

#python advmanage.py drive --model models/1.h5 --adv

def drive(cfg, model_path=None, model_type=None, adv=False, meta=[]):
    '''
    Construct a working robotic vehicle from many parts.
    Each part runs as a job in the Vehicle loop, calling either
    it's run or run_threaded method depending on the constructor flag `threaded`.
    All parts are updated one after another at the framerate given in
    cfg.DRIVE_LOOP_HZ assuming each part finishes processing in a timely manner.
    Parts may have named outputs and inputs. The framework handles passing named outputs
    to parts requesting the same named input.
    '''

    if cfg.DONKEY_GYM:
        #the simulator will use cuda and then we usually run out of resources
        #if we also try to use cuda. so disable for donkey_gym.
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    #Initialize car
    V = dk.vehicle.Vehicle()

    print("cfg.CAMERA_TYPE", cfg.CAMERA_TYPE)
    inputs = []
    threaded = True
    if cfg.DONKEY_GYM:
        from donkeycar.parts.dgym import DonkeyGymEnv 
        cam = DonkeyGymEnv(cfg.DONKEY_SIM_PATH, host=cfg.SIM_HOST, env_name=cfg.DONKEY_GYM_ENV_NAME, conf=cfg.GYM_CONF, delay=cfg.SIM_ARTIFICIAL_LATENCY)
        inputs = ['angle', 'throttle']
    elif cfg.CAMERA_TYPE == "IMAGE_LIST":
        from donkeycar.parts.camera import ImageListCamera
        cam = ImageListCamera(path_mask=cfg.PATH_MASK)
    else:
        raise(Exception("Unkown camera type: %s" % cfg.CAMERA_TYPE))

    V.add(cam, inputs=inputs, outputs=['cam/image_array', 'env/info'], threaded=threaded)


    #This web controller will create a web server that is capable
    #of managing steering, throttle, and modes, and more.
    ctr = LocalWebController(port=cfg.WEB_CONTROL_PORT, mode=cfg.WEB_INIT_MODE)
    
    V.add(ctr,
        inputs=['cam/image_array', 'tub/num_records'],
        outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
        threaded=True)

    #this throttle filter will allow one tap back for esc reverse
    th_filter = ThrottleFilter()
    V.add(th_filter, inputs=['user/throttle'], outputs=['user/throttle'])

    #See if we should even run the pilot module.
    #This is only needed because the part run_condition only accepts boolean
    class PilotCondition:
        def run(self, mode):
            if mode == 'user':
                return False
            else:
                return True

    V.add(PilotCondition(), inputs=['user/mode'], outputs=['run_pilot'])

    if model_type is None:
            model_type = cfg.DEFAULT_MODEL_TYPE

    def get_record_alert_color(num_records):
        col = (0, 0, 0)
        for count, color in cfg.RECORD_ALERT_COLOR_ARR:
            if num_records >= count:
                col = color
        return col

    class RecordTracker:
        def __init__(self):
            self.last_num_rec_print = 0
            self.dur_alert = 0
            self.force_alert = 0

        def run(self, num_records):
            if num_records is None:
                return 0

            if self.last_num_rec_print != num_records or self.force_alert:
                self.last_num_rec_print = num_records

                if num_records % 10 == 0:
                    print("recorded", num_records, "records")

                if num_records % cfg.REC_COUNT_ALERT == 0 or self.force_alert:
                    self.dur_alert = num_records // cfg.REC_COUNT_ALERT * cfg.REC_COUNT_ALERT_CYC
                    self.force_alert = 0

            if self.dur_alert > 0:
                self.dur_alert -= 1

            if self.dur_alert != 0:
                return get_record_alert_color(num_records)

            return 0

    rec_tracker_part = RecordTracker()
    V.add(rec_tracker_part, inputs=["tub/num_records"], outputs=['records/alert'])

    # Watch for following
    '''
        inputs=['cam/image_array'],
        outputs=[inf_input],
        run_condition='run_pilot')
    '''

    class AdvAttack:

        def __init__(self, kl, attack_freq):
            self.kl = kl
            self.attack_freq = attack_freq

        
        def adversarial_pattern(self, image, label):
            image = tf.cast(image, tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(image)
                prediction = self.kl.model.predict(image)
                loss = tf.keras.losses.MSE(label, prediction)
            
            gradient = tape.gradient(loss, image)
            
            signed_grad = tf.sign(gradient)
            
            return signed_grad

        def run(self, img, num_rec):
            if num_rec != None and num_rec % self.attack_freq == 0:
                ang = self.kl.model.predict(img)
                
                grad = self.adversarial_pattern(img, ang)
                
                return img + (grad[0]*0.5 + 0.5), img

    inputs=[inf_input]


    if model_path:
        #When we have a model, first create an appropriate Keras part
        kl = get_model_by_type(model_type, cfg)

        model_reload_cb = None

        if '.h5' in model_path or '.uff' in model_path or 'tflite' in model_path or '.pkl' in model_path:
            #when we have a .h5 extension
            #load everything from the model file
            load_model(kl, model_path)

            def reload_model(filename):
                load_model(kl, filename)

            model_reload_cb = reload_model
        else:
            print("ERR>> Unknown extension type on model file!!")
            return

        #these parts will reload the model file, but only when ai is running so we don't interrupt user driving
        V.add(FileWatcher(model_path), outputs=['modelfile/dirty'], run_condition="ai_running")
        V.add(DelayedTrigger(100), inputs=['modelfile/dirty'], outputs=['modelfile/reload'], run_condition="ai_running")
        V.add(TriggeredCallback(model_path, model_reload_cb), inputs=["modelfile/reload"], run_condition="ai_running")

        outputs=['pilot/angle', 'pilot/throttle']

        if cfg.TRAIN_LOCALIZER:
            outputs.append("pilot/loc")

        if adv:
            import tensorflow as tf

            V.add(AdvAttack(kl, cfg.ADV_ATTACK), inputs=[inf_input, "tub/num_records"], outputs=[inf_input, 'img/old'], run_condition='run_pilot')

        V.add(kl, inputs=inputs,
            outputs=outputs,
            run_condition='run_pilot')

    #Choose what inputs should change the car.
    class DriveMode:
        def run(self, mode,
                    user_angle, user_throttle,
                    pilot_angle, pilot_throttle):
            if mode == 'user':
                return user_angle, user_throttle

            elif mode == 'local_angle':
                return pilot_angle if pilot_angle else 0.0, user_throttle

            else:
                return pilot_angle if pilot_angle else 0.0, pilot_throttle * cfg.AI_THROTTLE_MULT if pilot_throttle else 0.0

    V.add(DriveMode(),
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'],
          outputs=['angle', 'throttle'])


    class AiRunCondition:
        '''
        A bool part to let us know when ai is running.
        '''
        def run(self, mode):
            if mode == "user":
                return False
            return True

    V.add(AiRunCondition(), inputs=['user/mode'], outputs=['ai_running'])

    #Ai Recording
    class AiRecordingCondition:
        '''
        return True when ai mode, otherwize respect user mode recording flag
        '''
        def run(self, mode, recording):
            if mode == 'user':
                return recording
            return True

    if cfg.RECORD_DURING_AI:
        V.add(AiRecordingCondition(), inputs=['user/mode', 'recording'], outputs=['recording'])

    class envInfoHandler:
        '''
        Breaks down info returned from environment for storage in tub
        '''
        def __init__(self, cfg):
            self.attack_freq = cfg.ADV_ATTACK
            self.img_h = cfg.IMAGE_H
            self.img_w = cfg.IMAGE_W
            self.img_d = cfg.IMAGE_DEPTH

        def run(self, info, img, img_old, num_rec):
            cte=info['cte']
            pos_x = info['pos'][0]
            pos_z = info['pos'][2]

            if num_rec != None and num_rec % self.attack_freq == 0:
                temp_img = img
                img = img_old
                img_adv = temp_img
            else:
                img_adv = np.zeros((self.img_h, self.img_w, self.img_d))

            return cte, pos_x, pos_z, img, img_adv

    #add tub to save data
    inputs=['cam/image_array',
            'user/angle', 'user/throttle',
            'user/mode']

    types=['image_array',
           'float', 'float',
           'str']

    if cfg.DONKEY_GYM:
        V.add(envInfoHandler(cfg), inputs=['env/info', inf_input, 'img/old', 'tub/num_records'], outputs=['env/cte','env/pos_x', 'env/pos_z', inf_input, 'adv/img'])
        inputs.append('env/cte')
        inputs.append('env/pos_x')
        inputs.append('env/pos_z')
        inputs.append('adv/img')
        types.append('float')
        types.append('float')
        types.append('float')
        types.append('image_array')

    th = TubHandler(path=cfg.DATA_PATH)
    tub = th.new_tub_writer(inputs=inputs, types=types, user_meta=meta)
    V.add(tub, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')


    print("You can now go to http://localhost:%d to drive your car." % cfg.WEB_CONTROL_PORT)

    #run the vehicle for 20 seconds
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ,
            max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config(myconfig=args['--myconfig'])

    if args['drive']:
        model_type = args['--type']

        drive(cfg, model_path=args['--model'],
              model_type=model_type,
              adv=args['--adv'],
              meta=args['--meta'])

    if args['train']:
        from train import multi_train, preprocessFileList

        tub = args['--tub']
        model = args['--model']
        transfer = args['--transfer']
        model_type = args['--type']
        continuous = args['--continuous']
        aug = args['--aug']
        dirs = preprocessFileList( args['--file'] )

        if tub is not None:
            tub_paths = [os.path.expanduser(n) for n in tub.split(',')]
            dirs.extend( tub_paths )

        if model_type is None:
            model_type = cfg.DEFAULT_MODEL_TYPE
            print("using default model type of", model_type)

        multi_train(cfg, dirs, model, transfer, model_type, continuous, aug)