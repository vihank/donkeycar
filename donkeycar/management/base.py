
import argparse
import os
import shutil
import socket
import stat
import sys
from socket import *
from pathlib import Path

from progress.bar import IncrementalBar
import donkeycar as dk
from donkeycar.management.joystick_creator import CreateJoystick
from donkeycar.management.tub import TubManager
from donkeycar.utils import *

PACKAGE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TEMPLATES_PATH = os.path.join(PACKAGE_PATH, 'templates')


def make_dir(path):
    real_path = os.path.expanduser(path)
    print('making dir ', real_path)
    if not os.path.exists(real_path):
        os.makedirs(real_path)
    return real_path


def load_config(config_path):

    '''
    load a config from the given path
    '''
    conf = os.path.expanduser(config_path)

    if not os.path.exists(conf):
        print("No config file at location: %s. Add --config to specify\
                location or run from dir containing config.py." % conf)
        return None

    try:
        cfg = dk.load_config(conf)
    except:
        print("Exception while loading config from", conf)
        return None

    return cfg


class BaseCommand(object):
    pass


class CreateCar(BaseCommand):
    
    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='createcar', usage='%(prog)s [options]')
        parser.add_argument('--path', default=None, help='path where to create car folder')
        parser.add_argument('--template', default=None, help='name of car template to use')
        parser.add_argument('--overwrite', action='store_true', help='should replace existing files')
        parser.add_argument('--adv', action='store_true', help='if you want adversarial potential')
        
        parsed_args = parser.parse_args(args)
        return parsed_args
        
    def run(self, args):
        args = self.parse_args(args)
        self.create_car(path=args.path, template=args.template, overwrite=args.overwrite, adv=args.adv)
  
    def create_car(self, path, template='complete', overwrite=False, adv=False):
        """
        This script sets up the folder structure for donkey to work.
        It must run without donkey installed so that people installing with
        docker can build the folder structure for docker to mount to.
        """

        # these are neeeded incase None is passed as path
        path = path or '~/mycar'
        template = template or 'basic'
        print("Creating car folder: {}".format(path))
        path = make_dir(path)
        
        print("Creating data & model folders.")
        folders = ['models', 'data', 'logs']
        folder_paths = [os.path.join(path, f) for f in folders]   
        for fp in folder_paths:
            make_dir(fp)
            
        # add car application and config files if they don't exist
        app_template_path = os.path.join(TEMPLATES_PATH, template+'.py')
        config_template_path = os.path.join(TEMPLATES_PATH, 'cfg_' + template + '.py')
        myconfig_template_path = os.path.join(TEMPLATES_PATH, 'myconfig.py')
        train_template_path = os.path.join(TEMPLATES_PATH, 'train.py')
        calibrate_template_path = os.path.join(TEMPLATES_PATH, 'calibrate.py')
        car_app_path = os.path.join(path, 'manage.py')
        car_adv_path = os.path.join(path, 'advmanage.py')
        car_config_path = os.path.join(path, 'config.py')
        mycar_config_path = os.path.join(path, 'myconfig.py')
        train_app_path = os.path.join(path, 'train.py')
        calibrate_app_path = os.path.join(path, 'calibrate.py')
        
        if adv:
            app_template_path = os.path.join(TEMPLATES_PATH, 'advcomplete.py')
            if os.path.exists(car_adv_path) and not overwrite:
                print('Adversarial car app already exists. Delete it and rerun createcar to replace.')
            else:
                print("Copying adversarial car application template: {}".format(template))
                shutil.copyfile(app_template_path, car_adv_path)
        
        else:            
            if os.path.exists(car_app_path) and not overwrite:
                print('Car app already exists. Delete it and rerun createcar to replace.')
            else:
                print("Copying car application template: {}".format(template))
                shutil.copyfile(app_template_path, car_app_path)

        if os.path.exists(car_config_path) and not overwrite:
            print('Car config already exists. Delete it and rerun createcar to replace.')
        else:
            print("Copying car config defaults. Adjust these before starting your car.")
            shutil.copyfile(config_template_path, car_config_path)

        if os.path.exists(train_app_path) and not overwrite:
            print('Train already exists. Delete it and rerun createcar to replace.')
        else:
            print("Copying train script. Adjust these before starting your car.")
            shutil.copyfile(train_template_path, train_app_path)
            os.chmod(train_app_path, stat.S_IRWXU)

        if os.path.exists(calibrate_app_path) and not overwrite:
            print('Calibrate already exists. Delete it and rerun createcar to replace.')
        else:
            print("Copying calibrate script. Adjust these before starting your car.")
            shutil.copyfile(calibrate_template_path, calibrate_app_path)
            os.chmod(calibrate_app_path, stat.S_IRWXU)

        if not os.path.exists(mycar_config_path):
            print("Copying my car config overrides")
            shutil.copyfile(myconfig_template_path, mycar_config_path)
            # now copy file contents from config to myconfig, with all lines
            # commented out.
            cfg = open(car_config_path, "rt")
            mcfg = open(mycar_config_path, "at")
            copy = False
            for line in cfg:
                if "import os" in line:
                    copy = True
                if copy: 
                    mcfg.write("# " + line)
            cfg.close()
            mcfg.close()
 
        print("Donkey setup complete.")


class UpdateCar(BaseCommand):
    '''
    always run in the base ~/mycar dir to get latest
    '''

    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='update', usage='%(prog)s [options]')
        parser.add_argument('--template', default=None, help='name of car template to use')
        parser.add_argument('--adv', action='store_true', help='if you want adversarial potential')
        parsed_args = parser.parse_args(args)
        return parsed_args
        
    def run(self, args):
        args = self.parse_args(args)
        cc = CreateCar()
        cc.create_car(path=".", overwrite=True, template=args.template, adv=args.adv)
        

class FindCar(BaseCommand):
    def parse_args(self, args):
        pass        

    def run(self, args):
        print('Looking up your computer IP address...')
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8",80))
        ip = s.getsockname()[0] 
        print('Your IP address: %s ' %s.getsockname()[0])
        s.close()
        
        print("Finding your car's IP address...")
        cmd = "sudo nmap -sP " + ip + "/24 | awk '/^Nmap/{ip=$NF}/B8:27:EB/{print ip}'"
        print("Your car's ip address is:" )
        os.system(cmd)


class CalibrateCar(BaseCommand):    
    
    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='calibrate', usage='%(prog)s [options]')
        parser.add_argument('--channel', help="The channel you'd like to calibrate [0-15]")
        parser.add_argument('--address', default='0x40', help="The i2c address you'd like to calibrate [default 0x40]")
        parser.add_argument('--bus', default=None, help="The i2c bus you'd like to calibrate [default autodetect]")
        parser.add_argument('--pwmFreq', default=60, help="The frequency to use for the PWM")
        parser.add_argument('--arduino', dest='arduino', action='store_true', help='Use arduino pin for PWM (calibrate pin=<channel>)')
        parser.set_defaults(arduino=False)
        parsed_args = parser.parse_args(args)
        return parsed_args

    def run(self, args):
        args = self.parse_args(args)
        channel = int(args.channel)

        if args.arduino == True:
            from donkeycar.parts.actuator import ArduinoFirmata

            arduino_controller = ArduinoFirmata(servo_pin=channel)
            print('init Arduino PWM on pin %d' %(channel))
            input_prompt = "Enter a PWM setting to test ('q' for quit) (0-180): "
        else:
            from donkeycar.parts.actuator import PCA9685
            from donkeycar.parts.sombrero import Sombrero

            s = Sombrero()

            busnum = None
            if args.bus:
                busnum = int(args.bus)
            address = int(args.address, 16)
            print('init PCA9685 on channel %d address %s bus %s' %(channel, str(hex(address)), str(busnum)))
            freq = int(args.pwmFreq)
            print("Using PWM freq: {}".format(freq))
            c = PCA9685(channel, address=address, busnum=busnum, frequency=freq)
            input_prompt = "Enter a PWM setting to test ('q' for quit) (0-1500): "
            print()
        while True:
            try:
                val = input(input_prompt)
                if val == 'q' or val == 'Q':
                    break
                pmw = int(val)
                if args.arduino == True:
                    arduino_controller.set_pulse(channel,pmw)
                else:
                    c.run(pmw)
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt received, exit.")
                break
            except Exception as ex:
                print("Oops, {}".format(ex))


class MakeMovieShell(BaseCommand):
    '''
    take the make movie args and then call make movie command
    with lazy imports
    '''
    def __init__(self):
        self.deg_to_rad = math.pi / 180.0

    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='makemovie')
        parser.add_argument('--tub', help='The tub to make movie from')
        parser.add_argument('--out', default='tub_movie.mp4', help='The movie filename to create. default: tub_movie.mp4')
        parser.add_argument('--config', default='./config.py', help='location of config file to use. default: ./config.py')
        parser.add_argument('--model', default=None, help='the model to use to show control outputs')
        parser.add_argument('--type', default=None, required=False, help='the model type to load')
        parser.add_argument('--salient', action="store_true", help='should we overlay salient map showing activations')
        parser.add_argument('--start', type=int, default=0, help='first frame to process')
        parser.add_argument('--end', type=int, default=-1, help='last frame to process')
        parser.add_argument('--scale', type=int, default=2, help='make image frame output larger by X mult')
        parser.add_argument('--draw-user-input', default=True, action='store_false', help='show user input on the video')
        parsed_args = parser.parse_args(args)
        return parsed_args, parser

    def run(self, args):
        '''
        Load the images from a tub and create a movie from them.
        Movie
        '''
        args, parser = self.parse_args(args)

        from donkeycar.management.makemovie import MakeMovie

        mm = MakeMovie()
        mm.run(args, parser)


class TubCheck(BaseCommand):
    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='tubcheck', usage='%(prog)s [options]')
        parser.add_argument('tubs', nargs='+', help='paths to tubs')
        parser.add_argument('--fix', action='store_true', help='remove problem records')
        parser.add_argument('--delete_empty', action='store_true', help='delete tub dir with no records')
        parsed_args = parser.parse_args(args)
        return parsed_args

    def check(self, tub_paths, fix=False, delete_empty=False):
        '''
        Check for any problems. Looks at tubs and find problems in any records or images that won't open.
        If fix is True, then delete images and records that cause problems.
        '''
        cfg = load_config('config.py')
        tubs = gather_tubs(cfg, tub_paths)

        for tub in tubs:
            tub.check(fix=fix)
            if delete_empty and tub.get_num_records() == 0:
                import shutil
                print("removing empty tub", tub.path)
                shutil.rmtree(tub.path)

    def run(self, args):
        args = self.parse_args(args)
        self.check(args.tubs, args.fix, args.delete_empty)


class ShowHistogram(BaseCommand):

    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='tubhist', usage='%(prog)s [options]')
        parser.add_argument('--tub', nargs='+', help='paths to tubs')
        parser.add_argument('--record', default=None, help='name of record to create histogram')
        parser.add_argument('--out', default=None, help='path where to save histogram end with .png')
        parsed_args = parser.parse_args(args)
        return parsed_args

    def show_histogram(self, tub_paths, record_name, out):
        '''
        Produce a histogram of record type frequency in the given tub
        '''
        from matplotlib import pyplot as plt
        from donkeycar.parts.tub_v2 import Tub
        import pandas as pd

        base_path = Path(os.path.expanduser(tub_paths)).absolute().as_posix()
        output = out or os.path.basename(tub_paths)
        tub = Tub(base_path)
        records = list(tub)
        angle = []
        pos_x = []
        pos_z = []
        cte = []
        data = []

        for record in records[400:-100]:
            if record_name is None:
                if 'env/cte' in record:
                    cte.append(record['env/cte'])

                    if 'env/pos_x' in record:
                        pos_x.append(record['env/pos_x'])
                        pos_z.append(record['env/pos_z'])

                    if 'angle' in record:
                        angle.append(record['angle'])
            else:
                data.append(record[record_name])


        if record_name is None:
            cte_ang_df = pd.DataFrame({'env/cte': cte, 'angle': angle})
            pos_df = pd.DataFrame({'env/pos_x': pos_x, 'env/pos_z': pos_z})
            try:
                pos = plt
                pos.plot(pos_df['env/pos_x'], pos_df['env/pos_z'])
                pos.xlabel('X Position')
                pos.ylabel('Y Position')
                pos.title('Position of Car')
                filename = output + '_pos.png'
                plt.savefig(filename)
            except Exception as e:
                print(e)
            cte_ang_df.hist(bins=50)
            plt.figtext(0.55, 0.6, cte_ang_df['env/cte'].describe().to_string())
        else:
            data_df = pd.DataFrame({record_name, data})
            data_df[record_name].hist(bins=50)
  
        try:
            if out is not None:
                filename = output
            else:
                if record_name is not None:
                    filename = output + '_hist_%s.png' % record_name.replace('/', '_')
                else:
                    filename = output + '_hist.png'
            plt.savefig(filename)
            print('saving image to:', filename)
        except Exception as e:
            print(e)
        plt.show()

    def run(self, args):
        args = self.parse_args(args)
        args.tub = ','.join(args.tub)
        self.show_histogram(args.tub, args.record, args.out)

class Stats(BaseCommand):
    '''
    Run f-test on cte data, comparing two tubs
    '''

    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='tubhist', usage='%(prog)s [options]')
        parser.add_argument('--tub1', nargs='+', help='paths to tub 1')
        parser.add_argument('--tub2', nargs='+', help='paths to tub 2')
        parsed_args = parser.parse_args(args)
        return parsed_args

    def fTest(self, tub1, tub2):
        from donkeycar.parts.tub_v2 import Tub
        import numpy as np
        import scipy.stats as stats
        
        tub1_path = Path(os.path.expanduser(tub1)).absolute().as_posix()
        tub2_path = Path(os.path.expanduser(tub2)).absolute().as_posix()
        tub1 = Tub(tub1_path)
        tub2 = Tub(tub2_path)
        records1 = list(tub1)
        records2 = list(tub2)
        cte1 = []
        cte2 = []

        for record in records1[400:-100]:
            if 'env/cte' in record:
                cte1.append(record['env/cte'])

        for record in records2[400:-100]:
            if 'env/cte' in record:
                cte2.append(record['env/cte'])

        print('Gathered CTE values')

        var1 = np.var(cte1, ddof=1)
        var2 = np.var(cte2, ddof=1)
        df1 = len(cte1)-1
        df2 = len(cte2)-1

        if var1 > var2:
            f_value = var1/var2
            p_value = 1-stats.f.cdf(f_value, df1, df2)
        else:
            f_value = var2/var1
            p_value = 1-stats.f.cdf(f_value, df2, df1)

        print('The F value is: %s' % f_value)
        print('The P-value is: %s' % p_value)
                

    def run(self, args):
        args = self.parse_args(args)
        args.tub1 = ','.join(args.tub1)
        args.tub2 = ','.join(args.tub2)
        self.fTest(args.tub1, args.tub2)

class VirtualDrive(BaseCommand):
    '''
    Gather images from tub and run model with attack on it to see the difference in steering angles
    '''
    def parse_args(aelf, args):
        parser = argparse.ArgumentParser(prog='virtualdrive', usage='%(prog)s [options]')
        parser.add_argument('--tub', nargs='+', help='paths to tubs')
        parser.add_argument('--model', default='./models/drive.h5', help='path to model')
        parser.add_argument('--type', default='dave2', help='type of model (linear|categorical|rnn|imu|behavior|3d|dave2|vgg|resnet)')
        parser.add_argument('--out', defaults='.', help='location of the output image')

    def run(self, args):
        from matplotlib import pyplot as plt
        from donkeycar.parts.tub_v2 import Tub
        from donkecar.parts.advattack import AdvAttack
        from donkeycar.utils import get_model_by_type
        import pandas as pd

        args = self.parse_args(args)
        cfg = load_config('myconfig.py')
        tub_path = args.tub
        out = args.out

        base_path = Path(os.path.expanduser(tub_path)).absolute().as_posix()
        output = out or os.path.basename(tub_path)
        tub = Tub(base_path)
        records = list(tub)
        imgs = []

        kl = get_model_by_type(args.type, cfg)

        attacker = AdvAttack(kl)

        for record in records[200:-100]:
            if 'cam/image_array' in record:
                imgs.append(record['cam/image_array'])

        print('Gathered data')
        

        for img in imgs:
            ang, adv_ang = attacker(img)

        '''
        for img in images:
            run the attack on the image
            run model on image
            run model on attacked image
            compare both angles
            stats?
            save?
        '''



class ShowCnnActivations(BaseCommand):

    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt

    def get_activations(self, image_path, model_path, cfg):
        '''
        Extracts features from an image

        returns activations/features
        '''
        from tensorflow.python.keras.models import load_model, Model

        model_path = os.path.expanduser(model_path)
        image_path = os.path.expanduser(image_path)

        model = load_model(model_path, compile=False)
        image = load_image(image_path, cfg)[None, ...]

        conv_layer_names = self.get_conv_layers(model)
        input_layer = model.get_layer(name='img_in').input
        activations = []      
        for conv_layer_name in conv_layer_names:
            output_layer = model.get_layer(name=conv_layer_name).output

            layer_model = Model(inputs=[input_layer], outputs=[output_layer])
            activations.append(layer_model.predict(image)[0])
        return activations

    def create_figure(self, activations):
        import math
        cols = 6

        for i, layer in enumerate(activations):
            fig = self.plt.figure()
            fig.suptitle('Layer {}'.format(i+1))

            print('layer {} shape: {}'.format(i+1, layer.shape))
            feature_maps = layer.shape[2]
            rows = math.ceil(feature_maps / cols)

            for j in range(feature_maps):
                self.plt.subplot(rows, cols, j + 1)

                self.plt.imshow(layer[:, :, j])
        
        self.plt.show()

    def get_conv_layers(self, model):
        conv_layers = []
        for layer in model.layers:
            if layer.__class__.__name__ == 'Conv2D':
                conv_layers.append(layer.name)
        return conv_layers

    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='cnnactivations', usage='%(prog)s [options]')
        parser.add_argument('--image', help='path to image')
        parser.add_argument('--model', default=None, help='path to model')
        parser.add_argument('--config', default='./config.py', help='location of config file to use. default: ./config.py')
        
        parsed_args = parser.parse_args(args)
        return parsed_args

    def run(self, args):
        args = self.parse_args(args)
        cfg = load_config(args.config)
        activations = self.get_activations(args.image, args.model, cfg)
        self.create_figure(activations)


class ShowPredictionPlots(BaseCommand):

    def plot_predictions(self, cfg, tub_paths, model_path, limit, model_type):
        '''
        Plot model predictions for angle and throttle against data from tubs.

        '''
        import matplotlib.pyplot as plt
        import pandas as pd

        model_path = os.path.expanduser(model_path)
        model = dk.utils.get_model_by_type(model_type, cfg)
        # This just gets us the text for the plot title:
        if model_type is None:
            model_type = cfg.DEFAULT_MODEL_TYPE
        model.load(model_path)

        user_angles = []
        user_throttles = []
        pilot_angles = []
        pilot_throttles = []       

        from donkeycar.parts.tub_v2 import Tub
        from pathlib import Path

        base_path = Path(os.path.expanduser(tub_paths)).absolute().as_posix()
        tub = Tub(base_path)
        records = list(tub)
        records = records[:limit]
        bar = IncrementalBar('Inferencing', max=len(records))

        for record in records:
            img_filename = os.path.join(base_path, Tub.images(), record['cam/image_array'])
            img = load_image(img_filename, cfg)
            user_angle = float(record["user/angle"])
            user_throttle = float(record["user/throttle"])
            pilot_angle, pilot_throttle = model.run(img)

            user_angles.append(user_angle)
            user_throttles.append(user_throttle)
            pilot_angles.append(pilot_angle)
            pilot_throttles.append(pilot_throttle)
            bar.next()

        angles_df = pd.DataFrame({'user_angle': user_angles, 'pilot_angle': pilot_angles})
        throttles_df = pd.DataFrame({'user_throttle': user_throttles, 'pilot_throttle': pilot_throttles})

        fig = plt.figure()

        title = "Model Predictions\nTubs: " + tub_paths + "\nModel: " + model_path + "\nType: " + model_type
        fig.suptitle(title)

        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        angles_df.plot(ax=ax1)
        throttles_df.plot(ax=ax2)

        ax1.legend(loc=4)
        ax2.legend(loc=4)

        plt.savefig(model_path + '_pred.png')
        plt.show()

    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='tubplot', usage='%(prog)s [options]')
        parser.add_argument('--tub', nargs='+', help='The tub to make plot from')
        parser.add_argument('--model', default=None, help='model for predictions')
        parser.add_argument('--limit', type=int, default=1000, help='how many records to process')
        parser.add_argument('--type', default=None, help='model type')
        parser.add_argument('--config', default='./config.py', help='location of config file to use. default: ./config.py')
        parsed_args = parser.parse_args(args)
        return parsed_args

    def run(self, args):
        args = self.parse_args(args)
        args.tub = ','.join(args.tub)
        cfg = load_config(args.config)
        self.plot_predictions(cfg, args.tub, args.model, args.limit, args.type)


class Train(BaseCommand):

    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='train', usage='%(prog)s [options]')
        parser.add_argument('--tub', nargs='+', help='tub data for training')
        parser.add_argument('--model', default=None, help='output model name')
        parser.add_argument('--type', default=None, help='model type')
        parser.add_argument('--config', default='./config.py', help='location of config file to use. default: ./config.py')
        parser.add_argument('--framework', choices=['tensorflow', 'pytorch', None], required=False, help='the AI framework to use (tensorflow|pytorch). Defaults to config.DEFAULT_AI_FRAMEWORK')
        parser.add_argument('--checkpoint', type=str, help='location of checkpoint to resume training from')
        parsed_args = parser.parse_args(args)
        return parsed_args

    def run(self, args):
        args = self.parse_args(args)
        args.tub = ','.join(args.tub)
        cfg = load_config(args.config)
        framework = args.framework if args.framework else cfg.DEFAULT_AI_FRAMEWORK

        if framework == 'tensorflow':
            from donkeycar.pipeline.training import train
            train(cfg, args.tub, args.model, args.type)
        elif framework == 'pytorch':
            from donkeycar.parts.pytorch.torch_train import train
            train(cfg, args.tub, args.model, args.type,
                  checkpoint_path=args.checkpoint)
        else:
            print("Unrecognized framework: {}. Please specify one of 'tensorflow' or 'pytorch'".format(framework))


def execute_from_command_line():
    """
    This is the function linked to the "donkey" terminal command.
    """
    commands = {
            'createcar': CreateCar,
            'findcar': FindCar,
            'calibrate': CalibrateCar,
            'tubclean': TubManager,
            'tubhist': ShowHistogram,
            'tubplot': ShowPredictionPlots,
            'tubcheck': TubCheck,
            'makemovie': MakeMovieShell,            
            'createjs': CreateJoystick,
            'cnnactivations': ShowCnnActivations,
            'update': UpdateCar,
            'research': Stats,
            'train': Train,
    }
    
    args = sys.argv[:]

    if len(args) > 1 and args[1] in commands.keys():
        command = commands[args[1]]
        c = command()
        c.run(args[2:])
    else:
        dk.utils.eprint('Usage: The available commands are:')
        dk.utils.eprint(list(commands.keys()))
        
    
if __name__ == "__main__":
    execute_from_command_line()
