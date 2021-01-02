#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car

Usage:
    manage.py (drive) [--model=<model>] [--js] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer|latent)] [--camera=(single|stereo)] [--meta=<key:value> ...] [--myconfig=<filename>]
    manage.py (train) [--tub=<tub1,tub2,..tubn>] [--file=<file> ...] (--model=<model>) [--transfer=<model>] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer)] [--continuous] [--aug] [--myconfig=<filename>]


Options:
    -h --help               Show this screen.
    --js                    Use physical joystick.
    -f --file=<file>        A text file containing paths to tub files, one per line. Option may be used more than once.
    --meta=<key:value>      Key/Value strings describing describing a piece of meta data about this drive. Option may be used more than once.
    --myconfig=filename     Specify myconfig file to use.
                            [default: myconfig.py]
"""
import os
import time

from docopt import docopt
import numpy as np

import donkeycar as dk

# import parts
from donkeycar.parts.transform import Lambda, TriggeredCallback, DelayedTrigger
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.controller import LocalWebController, \
    JoystickController, WebFpv
from donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.parts.behavior import BehaviorPart
from donkeycar.parts.file_watcher import FileWatcher
from donkeycar.parts.launch import AiLaunch
from donkeycar.utils import *
import cv2
from pyaudio import *
import wave
import sys
import pygame
def play1(music):
    print('playing music')
    time.sleep(5)
    pygame.mixer.init()
    s=pygame.mixer.Sound(r'ring_0'+str(music)+'.wav')
    s.set_volume(1)
    s.play()
    time.sleep(5)
def play(music):
    # 用文本文件记录wave模块解码每一帧所产生的内容。注意这里不是保存为二进制文件
    mus=music
    if(mus==0):
        return
    if(mus==1):
        dump_buff_file = open(r"ring_01.dup", 'w')
        chunk = 512  # 指定WAV文件的大小
        wf = wave.open(r"ring_01.wav", 'rb')  # 打开WAV文件
    if (mus == 2):
        dump_buff_file = open(r"ring_02.dup", 'w')
        chunk = 512  # 指定WAV文件的大小
        wf = wave.open(r"ring_02.wav", 'rb')  # 打开WAV文件
    if (mus == 3):
        dump_buff_file = open(r"ring_03.dup", 'w')
        chunk = 512  # 指定WAV文件的大小
        wf = wave.open(r"ring_03.wav", 'rb')  # 打开WAV文件
    p = PyAudio()  # 初始化PyAudio模块

    # 打开一个数据流对象，解码而成的帧将直接通过它播放出来，我们就能听到声音啦
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                    rate=wf.getframerate(), output=True)

    data = wf.readframes(chunk)  # 读取第一帧数据
    #print(data)  # 以文本形式打印出第一帧数据，实际上是转义之后的十六进制字符串

    # 播放音频，并使用while循环继续读取并播放后面的帧数
    # 结束的标志为wave模块读到了空的帧
    while data != b'':
        stream.write(data)  # 将帧写入数据流对象中，以此播放之
        data = wf.readframes(chunk)  # 继续读取后面的帧
        dump_buff_file.write(str(data) + "\n---------------------------------------\n")  # 将读出的帧写入文件中，每一个帧用分割线隔开以便阅读

    stream.stop_stream()  # 停止数据流
    stream.close()  # 关闭数据流
    p.terminate()  # 关闭 PyAudio
    print('play函数结束！')

'''********************************************************************************************************'''


def drive(cfg, model_path=None, use_joystick=False, model_type=None, camera_type='single', meta=[]):
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
        # the simulator will use cuda and then we usually run out of resources
        # if we also try to use cuda. so disable for donkey_gym.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if model_type is None:
        if cfg.TRAIN_LOCALIZER:
            model_type = "localizer"
        elif cfg.TRAIN_BEHAVIORS:
            model_type = "behavior"
        else:
            model_type = cfg.DEFAULT_MODEL_TYPE

    # Initialize car
    V = dk.vehicle.Vehicle()

    print("cfg.CAMERA_TYPE", cfg.CAMERA_TYPE)
    if camera_type == "stereo":

        if cfg.CAMERA_TYPE == "WEBCAM":
            from donkeycar.parts.camera import Webcam

            camA = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam=0)
            camB = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam=1)

        elif cfg.CAMERA_TYPE == "CVCAM":
            from donkeycar.parts.cv import CvCam

            camA = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam=0)
            camB = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, iCam=1)
        else:
            raise (Exception("Unsupported camera type: %s" % cfg.CAMERA_TYPE))

        V.add(camA, outputs=['cam/image_array_a'], threaded=True)
        V.add(camB, outputs=['cam/image_array_b'], threaded=True)

        from donkeycar.parts.image import StereoPair

        V.add(StereoPair(), inputs=['cam/image_array_a', 'cam/image_array_b'],
              outputs=['cam/image_array'])
    elif cfg.CAMERA_TYPE == "D435":
        from donkeycar.parts.realsense435i import RealSense435i
        cam = RealSense435i(
            enable_rgb=cfg.REALSENSE_D435_RGB,
            enable_depth=cfg.REALSENSE_D435_DEPTH,
            enable_imu=cfg.REALSENSE_D435_IMU,
            device_id=cfg.REALSENSE_D435_ID)
        V.add(cam, inputs=[],
              outputs=['cam/image_array', 'cam/depth_array',
                       'imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                       'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z'],
              threaded=True)

    else:
        if cfg.DONKEY_GYM:
            from donkeycar.parts.dgym import DonkeyGymEnv

        inputs = []
        threaded = True
        if cfg.DONKEY_GYM:
            from donkeycar.parts.dgym import DonkeyGymEnv
            cam = DonkeyGymEnv(cfg.DONKEY_SIM_PATH, host=cfg.SIM_HOST, env_name=cfg.DONKEY_GYM_ENV_NAME,
                               conf=cfg.GYM_CONF, delay=cfg.SIM_ARTIFICIAL_LATENCY)
            threaded = True
            inputs = ['angle', 'throttle']
        elif cfg.CAMERA_TYPE == "PICAM":
            from donkeycar.parts.camera import PiCamera
            cam = PiCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH,
                           framerate=cfg.CAMERA_FRAMERATE, vflip=cfg.CAMERA_VFLIP, hflip=cfg.CAMERA_HFLIP)
        elif cfg.CAMERA_TYPE == "WEBCAM":
            from donkeycar.parts.camera import Webcam
            cam = Webcam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        elif cfg.CAMERA_TYPE == "CVCAM":
            from donkeycar.parts.cv import CvCam
            cam = CvCam(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        elif cfg.CAMERA_TYPE == "CSIC":
            from donkeycar.parts.camera import CSICamera
            cam = CSICamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH,
                            framerate=cfg.CAMERA_FRAMERATE, gstreamer_flip=cfg.CSIC_CAM_GSTREAMER_FLIP_PARM)
        elif cfg.CAMERA_TYPE == "V4L":
            from donkeycar.parts.camera import V4LCamera
            cam = V4LCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH,
                            framerate=cfg.CAMERA_FRAMERATE)
        elif cfg.CAMERA_TYPE == "MOCK":
            from donkeycar.parts.camera import MockCamera
            cam = MockCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
        elif cfg.CAMERA_TYPE == "IMAGE_LIST":
            from donkeycar.parts.camera import ImageListCamera
            cam = ImageListCamera(path_mask=cfg.PATH_MASK)
        else:
            raise (Exception("Unkown camera type: %s" % cfg.CAMERA_TYPE))

        V.add(cam, inputs=inputs, outputs=['cam/image_array'], threaded=threaded)

    if use_joystick or cfg.USE_JOYSTICK_AS_DEFAULT:
        # modify max_throttle closer to 1.0 to have more power
        # modify steering_scale lower than 1.0 to have less responsive steering
        if cfg.CONTROLLER_TYPE == "MM1":
            from donkeycar.parts.robohat import RoboHATController
            ctr = RoboHATController(cfg)
        elif "custom" == cfg.CONTROLLER_TYPE:
            #
            # custom controller created with `donkey createjs` command
            #
            from my_joystick import MyJoystickController
            ctr = MyJoystickController(
                throttle_dir=cfg.JOYSTICK_THROTTLE_DIR,
                throttle_scale=cfg.JOYSTICK_MAX_THROTTLE,
                steering_scale=cfg.JOYSTICK_STEERING_SCALE,
                auto_record_on_throttle=cfg.AUTO_RECORD_ON_THROTTLE)
            ctr.set_deadzone(cfg.JOYSTICK_DEADZONE)
        else:
            from donkeycar.parts.controller import get_js_controller

            ctr = get_js_controller(cfg)

            if cfg.USE_NETWORKED_JS:
                from donkeycar.parts.controller import JoyStickSub
                netwkJs = JoyStickSub(cfg.NETWORK_JS_SERVER_IP)
                V.add(netwkJs, threaded=True)
                ctr.js = netwkJs

        V.add(ctr,
              inputs=['cam/image_array', 'tub/num_records', 'showbutton'],
              outputs=['user/angle', 'user/throttle', 'user/mode', 'recording', 'wim', 'wtg', 'gon'],
              threaded=True)

    else:
        # This web controller will create a web server that is capable
        # of managing steering, throttle, and modes, and more.
        ctr = LocalWebController(port=cfg.WEB_CONTROL_PORT, mode=cfg.WEB_INIT_MODE)

        V.add(ctr,
              inputs=['cam/image_array', 'tub/num_records','showbutton'],
              outputs=['user/angle', 'user/throttle', 'user/mode', 'recording','wim','wtg','gon'],
              threaded=True)

    # this throttle filter will allow one tap back for esc reverse
    th_filter = ThrottleFilter()
    V.add(th_filter, inputs=['user/throttle'], outputs=['user/throttle'])

    # See if we should even run the pilot module.
    # This is only needed because the part run_condition only accepts boolean
    class PilotCondition:
        def run(self, mode):
            if mode == 'user':
                return False
            else:
                return True

    V.add(PilotCondition(), inputs=['user/mode'], outputs=['run_pilot'])

    class LedConditionLogic:
        def __init__(self, cfg):
            self.cfg = cfg

        def run(self, mode, recording, recording_alert, behavior_state, model_file_changed, track_loc):
            # returns a blink rate. 0 for off. -1 for on. positive for rate.

            if track_loc is not None:
                led.set_rgb(*self.cfg.LOC_COLORS[track_loc])
                return -1

            if model_file_changed:
                led.set_rgb(self.cfg.MODEL_RELOADED_LED_R, self.cfg.MODEL_RELOADED_LED_G, self.cfg.MODEL_RELOADED_LED_B)
                return 0.1
            else:
                led.set_rgb(self.cfg.LED_R, self.cfg.LED_G, self.cfg.LED_B)

            if recording_alert:
                led.set_rgb(*recording_alert)
                return self.cfg.REC_COUNT_ALERT_BLINK_RATE
            else:
                led.set_rgb(self.cfg.LED_R, self.cfg.LED_G, self.cfg.LED_B)

            if behavior_state is not None and model_type == 'behavior':
                r, g, b = self.cfg.BEHAVIOR_LED_COLORS[behavior_state]
                led.set_rgb(r, g, b)
                return -1  # solid on

            if recording:
                return -1  # solid on
            elif mode == 'user':
                return 1
            elif mode == 'local_angle':
                return 0.5
            elif mode == 'local':
                return 0.1
            return 0

    if cfg.HAVE_RGB_LED and not cfg.DONKEY_GYM:
        from donkeycar.parts.led_status import RGB_LED
        led = RGB_LED(cfg.LED_PIN_R, cfg.LED_PIN_G, cfg.LED_PIN_B, cfg.LED_INVERT)
        led.set_rgb(cfg.LED_R, cfg.LED_G, cfg.LED_B)

        V.add(LedConditionLogic(cfg),
              inputs=['user/mode', 'recording', "records/alert", 'behavior/state', 'modelfile/modified', "pilot/loc"],
              outputs=['led/blink_rate'])

        V.add(led, inputs=['led/blink_rate'])

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

    if cfg.AUTO_RECORD_ON_THROTTLE and isinstance(ctr, JoystickController):
        # then we are not using the circle button. hijack that to force a record count indication
        def show_record_acount_status():
            rec_tracker_part.last_num_rec_print = 0
            rec_tracker_part.force_alert = 1

        ctr.set_button_down_trigger('circle', show_record_acount_status)

    # Sombrero
    if cfg.HAVE_SOMBRERO:
        from donkeycar.parts.sombrero import Sombrero
        s = Sombrero()

    # IMU
    if cfg.HAVE_IMU:
        from donkeycar.parts.imu import IMU
        imu = IMU(sensor=cfg.IMU_SENSOR, dlp_setting=cfg.IMU_DLP_CONFIG)
        V.add(imu, outputs=['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z'], threaded=True)

    class ImgPreProcess():
        '''
        preprocess camera image for inference.
        normalize and crop if needed.
        '''

        def __init__(self, cfg):
            self.cfg = cfg

        def run(self, img_arr):
            return normalize_and_crop(img_arr, self.cfg)

    if "coral" in model_type:
        inf_input = 'cam/image_array'
    else:
        inf_input = 'cam/normalized/cropped'
        V.add(ImgPreProcess(cfg),
              inputs=['cam/image_array'],
              outputs=[inf_input],
              run_condition='run_pilot')

    # Use the FPV preview, which will show the cropped image output, or the full frame.
    if cfg.USE_FPV:
        V.add(WebFpv(), inputs=['cam/image_array'], threaded=True)

    # Behavioral state
    if cfg.TRAIN_BEHAVIORS:
        bh = BehaviorPart(cfg.BEHAVIOR_LIST)
        V.add(bh, outputs=['behavior/state', 'behavior/label', "behavior/one_hot_state_array"])
        try:
            ctr.set_button_down_trigger('L1', bh.increment_state)
        except:
            pass

        inputs = [inf_input, "behavior/one_hot_state_array"]
    # IMU
    elif model_type == "imu":
        assert (cfg.HAVE_IMU)
        # Run the pilot if the mode is not user.
        inputs = [inf_input,
                  'imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                  'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z']
    else:
        inputs = [inf_input]

    def load_model(kl, model_path):
        start = time.time()
        print('loading model', model_path)
        kl.load(model_path)
        print('finished loading in %s sec.' % (str(time.time() - start)))

    def load_weights(kl, weights_path):
        start = time.time()
        try:
            print('loading model weights', weights_path)
            kl.model.load_weights(weights_path)
            print('finished loading in %s sec.' % (str(time.time() - start)))
        except Exception as e:
            print(e)
            print('ERR>> problems loading weights', weights_path)

    def load_model_json(kl, json_fnm):
        start = time.time()
        print('loading model json', json_fnm)
        from tensorflow.python import keras
        try:
            with open(json_fnm, 'r') as handle:
                contents = handle.read()
                kl.model = keras.models.model_from_json(contents)
            print('finished loading json in %s sec.' % (str(time.time() - start)))
        except Exception as e:
            print(e)
            print("ERR>> problems loading model json", json_fnm)

    if model_path:
        # When we have a model, first create an appropriate Keras part
        kl = dk.utils.get_model_by_type(model_type, cfg)

        model_reload_cb = None

        if '.h5' in model_path or '.uff' in model_path or 'tflite' in model_path or '.pkl' in model_path:
            # when we have a .h5 extension
            # load everything from the model file
            load_model(kl, model_path)

            def reload_model(filename):
                load_model(kl, filename)

            model_reload_cb = reload_model

        elif '.json' in model_path:
            # when we have a .json extension
            # load the model from there and look for a matching
            # .wts file with just weights
            load_model_json(kl, model_path)
            weights_path = model_path.replace('.json', '.weights')
            load_weights(kl, weights_path)

            def reload_weights(filename):
                weights_path = filename.replace('.json', '.weights')
                load_weights(kl, weights_path)

            model_reload_cb = reload_weights

        else:
            print("ERR>> Unknown extension type on model file!!")
            return

        # this part will signal visual LED, if connected
        V.add(FileWatcher(model_path, verbose=True), outputs=['modelfile/modified'])

        # these parts will reload the model file, but only when ai is running so we don't interrupt user driving
        V.add(FileWatcher(model_path), outputs=['modelfile/dirty'], run_condition="ai_running")
        V.add(DelayedTrigger(100), inputs=['modelfile/dirty'], outputs=['modelfile/reload'], run_condition="ai_running")
        V.add(TriggeredCallback(model_path, model_reload_cb), inputs=["modelfile/reload"], run_condition="ai_running")

        outputs = ['pilot/angle', 'pilot/throttle']

        if cfg.TRAIN_LOCALIZER:
            outputs.append("pilot/loc")

        V.add(kl, inputs=inputs,
              outputs=outputs,
              run_condition='run_pilot')

    if cfg.STOP_SIGN_DETECTOR:
        from donkeycar.parts.object_detector.stop_sign_detector import StopSignDetector
        V.add(StopSignDetector(cfg.STOP_SIGN_MIN_SCORE, cfg.STOP_SIGN_SHOW_BOUNDING_BOX),
              inputs=['cam/image_array', 'pilot/throttle'], outputs=['pilot/throttle', 'cam/image_array'])

    # Choose what inputs should change the car.
        # Choose what inputs should change the car.
        '''下面代码能实现的重要条件是小车一开始就得处于原来的自动驾驶状态
        '''

        # wci=梁的where car is
        # wim=蔡的where i am， wtg=蔡的where to go, gon=蔡接收到button的值后，另一个变量go or not
    class Picmatch:
        def __init__(self):
            self.mylocation = '0'
            self.color = 'green'
            self.template = []
            self.green_sign = 0
            self.yellow_sign = 0
            self.time1 = time.time()
            self.img = np.ones(shape=(120, 160, 3))
            print('start!!!')
            self.template = cv2.imread(self.color + '_tem.jpg', 1)

        def run_threaded(self, img):
            self.img = img
            return self.mylocation

        def update(self):
            while True:
                self.mylocation = self.run(self.img)

        def run(self, img):
            try:
                img = np.array(img)
                img = np.array(img, dtype='uint8')
                img = img[70:120]
            except:
                return '0'
                # 相关系数匹配方法: cv2.TM_CCOEFF
            res1 = cv2.matchTemplate(img, self.template, cv2.TM_CCOEFF_NORMED)
            min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
            if (max_val1 >= 0.62):
                if (time.time() - self.time1 >= 2):
                    self.green_sign += 1
                    if (self.green_sign == 7):
                        self.green_sign = 1
                self.time1 = time.time()
                self.mylocation = str(self.green_sign)
            #if (self.mylocation != '0'):
                #print(self.mylocation)
                #print(max_val1)
            return self.mylocation

    V.add(Picmatch(), inputs=['cam/image_array'], outputs=['wci'], threaded=True)


    class AfterPictureMatch:
        def __init__(self):
            self.extra = 0.
            self.showbutton = 0.  # '输入位置后点此等待小车接应！'
            self.wci = '0'
            self.wim = ''
            self.wtg = ''
            self.gon = 0
            self.music_sign = 0
            self.pre_music_sign=0

        def run_threaded(self, wci, wim, wtg, gon):
            self.wci = wci
            self.wim = wim
            self.wtg = wtg
            self.gon = gon
            return self.extra, self.showbutton

        def update(self):
            while True:
                self.extra, self.showbutton= self.run(self.wci, self.wim, self.wtg, self.gon)

        def run(self, wci, wim, wtg, gon):
            #    def run(self,wim,wtg,gon):
            print('wci:', wci, 'wim:', wim, 'wtg:', wtg, 'gon:', gon)
            if gon == 0:
                # gon默认为0
                if wim == wci:
                    print('可以准备出发')
                    self.showbutton = 2.
                    self.extra = 0
                    # '点此即可以出发！'
                    # 蔡小程序接收到button值，蔡默认为0的gon改为1
                    # play(2)#提示音我在这里哦，点击按钮即可出发
                    if(self.pre_music_sign!=2):
                        play(2)
                        self.pre_music_sign=2
                    return self.extra, self.showbutton
                elif wim != wci:
                    if (wim == ''):
                        self.extra = 0
                        self.showbutton = 0
                        self.pre_music_sign=0
                        return self.extra, self.showbutton
                    else:
                        print('出发地不等于所在地')
                        self.showbutton = 1.  # '请等待小车到达！'
                        self.extra = 1.
                        # play(1)#"提示音：小车正在自动驾驶中"
                        if (self.pre_music_sign != 1):
                            play(1)
                            self.pre_music_sign =1
                        return self.extra, self.showbutton
            else:
                if wci == wtg:
                    print('到达目的地')
                    self.extra = 0
                    self.showbutton = 3.  # '已经到达！'，蔡gon由1改为0
                    # play(3)#提示音：已经到达祝主人一路顺风
                    if(self.pre_music_sign!=3):
                        play(3)
                        self.pre_music_sign=3
                    return self.extra, self.showbutton
                elif wci != wtg:
                    print('在前进过程中')
                    self.extra = 1
                    self.showbutton = 1.  # '请等待小车到达！'
                    # play(1)#"提示音：小车正在自动驾驶中"
                    if(self.pre_music_sign!=4):
                        play(1)
                        self.pre_music_sign=4
                    return self.extra, self.showbutton

    V.add(AfterPictureMatch(),
          # inputs=['wim','wtg','gon'],
          inputs=['wci', 'wim', 'wtg', 'gon'],
          outputs=['extra', 'showbutton'], threaded=True)

    class PlayMusic:
        def __init__(self):
            self.play_sign=0
            self.pre_play_sign=0
        def run_threaded(self, music):
            self.play_sign=music
            if(music!=0 and self.pre_play_sign==0):
                play(music)
                self.pre_play_sign=music
            if(music==0):
                self.pre_play_sign=0
        def update(self):
            pass
    #V.add(PlayMusic,inputs=['music'],outputs=[],threaded=True)

    class Play_Music:
        def __init__(self):
            self.pre_play_sign=0
        def run(self,music=0):
            self.play_sign=music
            if(music!=0 and self.pre_play_sign==0):
                play(music)
                self.pre_play_sign=music
            if(music==0):
                self.pre_play_sign=0
    #V.add(Play_Music,inputs=['music'],outputs=[],)

    class DriveMode:
        def run(self,mode,extra,
                         user_angle,user_throttle,
                         pilot_angle,pilot_throttle):
            if mode == 'local_angle':
                return pilot_angle if pilot_angle else 0.0, user_throttle
            elif mode=='user' or extra==0:
            #extra默认为0，由梁的类输出的一个切换参量
               return user_angle,user_throttle
            #已停止模式+进入沟里，控制驶出
            else:
               return pilot_angle if pilot_angle else 0.0, pilot_throttle * cfg.AI_THROTTLE_MULT if pilot_throttle else 0.0
            #已进入自动驾驶模式
            #下面是extra参量的来源
    V.add(DriveMode(),
          inputs=['user/mode', 'extra','user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'],
          outputs=['angle', 'throttle'])

    # to give the car a boost when starting ai mode in a race.
    aiLauncher = AiLaunch(cfg.AI_LAUNCH_DURATION, cfg.AI_LAUNCH_THROTTLE, cfg.AI_LAUNCH_KEEP_ENABLED)

    V.add(aiLauncher,
          inputs=['user/mode', 'throttle'],
          outputs=['throttle'])

    if isinstance(ctr, JoystickController):
        ctr.set_button_down_trigger(cfg.AI_LAUNCH_ENABLE_BUTTON, aiLauncher.enable_ai_launch)

    class AiRunCondition:
        '''
        A bool part to let us know when ai is running.
        '''

        def run(self, mode):
            if mode == "user":
                return False
            return True

    V.add(AiRunCondition(), inputs=['user/mode'], outputs=['ai_running'])

    # Ai Recording
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

    # Drive train setup
    if cfg.DONKEY_GYM or cfg.DRIVE_TRAIN_TYPE == "MOCK":
        pass
    elif cfg.DRIVE_TRAIN_TYPE == "SERVO_ESC":
        from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle

        steering_controller = PCA9685(cfg.STEERING_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
        steering = PWMSteering(controller=steering_controller,
                               left_pulse=cfg.STEERING_LEFT_PWM,
                               right_pulse=cfg.STEERING_RIGHT_PWM)

        throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
        throttle = PWMThrottle(controller=throttle_controller,
                               max_pulse=cfg.THROTTLE_FORWARD_PWM,
                               zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                               min_pulse=cfg.THROTTLE_REVERSE_PWM)

        V.add(steering, inputs=['angle'], threaded=True)
        V.add(throttle, inputs=['throttle'], threaded=True)


    elif cfg.DRIVE_TRAIN_TYPE == "DC_STEER_THROTTLE":
        from donkeycar.parts.actuator import Mini_HBridge_DC_Motor_PWM

        steering = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_LEFT, cfg.HBRIDGE_PIN_RIGHT)
        throttle = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_FWD, cfg.HBRIDGE_PIN_BWD)

        V.add(steering, inputs=['angle'])
        V.add(throttle, inputs=['throttle'])


    elif cfg.DRIVE_TRAIN_TYPE == "DC_TWO_WHEEL":
        from donkeycar.parts.actuator import TwoWheelSteeringThrottle, Mini_HBridge_DC_Motor_PWM

        left_motor = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_LEFT_FWD, cfg.HBRIDGE_PIN_LEFT_BWD)
        right_motor = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_RIGHT_FWD, cfg.HBRIDGE_PIN_RIGHT_BWD)
        two_wheel_control = TwoWheelSteeringThrottle()

        V.add(two_wheel_control,
              inputs=['throttle', 'angle'],
              outputs=['left_motor_speed', 'right_motor_speed'])

        V.add(left_motor, inputs=['left_motor_speed'])
        V.add(right_motor, inputs=['right_motor_speed'])

    elif cfg.DRIVE_TRAIN_TYPE == "SERVO_HBRIDGE_PWM":
        from donkeycar.parts.actuator import ServoBlaster, PWMSteering
        steering_controller = ServoBlaster(cfg.STEERING_CHANNEL)  # really pin
        # PWM pulse values should be in the range of 100 to 200
        assert (cfg.STEERING_LEFT_PWM <= 200)
        assert (cfg.STEERING_RIGHT_PWM <= 200)
        steering = PWMSteering(controller=steering_controller,
                               left_pulse=cfg.STEERING_LEFT_PWM,
                               right_pulse=cfg.STEERING_RIGHT_PWM)

        from donkeycar.parts.actuator import Mini_HBridge_DC_Motor_PWM
        motor = Mini_HBridge_DC_Motor_PWM(cfg.HBRIDGE_PIN_FWD, cfg.HBRIDGE_PIN_BWD)

        V.add(steering, inputs=['angle'], threaded=True)
        V.add(motor, inputs=["throttle"])

    elif cfg.DRIVE_TRAIN_TYPE == "MM1":
        from donkeycar.parts.robohat import RoboHATDriver
        V.add(RoboHATDriver(cfg), inputs=['angle', 'throttle'])

    elif cfg.DRIVE_TRAIN_TYPE == "PIGPIO_PWM":
        from donkeycar.parts.actuator import PWMSteering, PWMThrottle, PiGPIO_PWM
        steering_controller = PiGPIO_PWM(cfg.STEERING_PWM_PIN, freq=cfg.STEERING_PWM_FREQ,
                                         inverted=cfg.STEERING_PWM_INVERTED)
        steering = PWMSteering(controller=steering_controller,
                               left_pulse=cfg.STEERING_LEFT_PWM,
                               right_pulse=cfg.STEERING_RIGHT_PWM)

        throttle_controller = PiGPIO_PWM(cfg.THROTTLE_PWM_PIN, freq=cfg.THROTTLE_PWM_FREQ,
                                         inverted=cfg.THROTTLE_PWM_INVERTED)
        throttle = PWMThrottle(controller=throttle_controller,
                               max_pulse=cfg.THROTTLE_FORWARD_PWM,
                               zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                               min_pulse=cfg.THROTTLE_REVERSE_PWM)
        V.add(steering, inputs=['angle'], threaded=True)
        V.add(throttle, inputs=['throttle'], threaded=True)

    # OLED setup
    if cfg.USE_SSD1306_128_32:
        from donkeycar.parts.oled import OLEDPart
        auto_record_on_throttle = cfg.USE_JOYSTICK_AS_DEFAULT and cfg.AUTO_RECORD_ON_THROTTLE
        oled_part = OLEDPart(cfg.SSD1306_128_32_I2C_BUSNUM, auto_record_on_throttle=auto_record_on_throttle)
        V.add(oled_part, inputs=['recording', 'tub/num_records', 'user/mode'], outputs=[], threaded=True)

    # add tub to save data

    inputs = ['cam/image_array',
              'user/angle', 'user/throttle',
              'user/mode']

    types = ['image_array',
             'float', 'float',
             'str']

    if cfg.TRAIN_BEHAVIORS:
        inputs += ['behavior/state', 'behavior/label', "behavior/one_hot_state_array"]
        types += ['int', 'str', 'vector']

    if cfg.CAMERA_TYPE == "D435" and cfg.REALSENSE_D435_DEPTH:
        inputs += ['cam/depth_array']
        types += ['gray16_array']

    if cfg.HAVE_IMU or (cfg.CAMERA_TYPE == "D435" and cfg.REALSENSE_D435_IMU):
        inputs += ['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                   'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z']

        types += ['float', 'float', 'float',
                  'float', 'float', 'float']

    if cfg.RECORD_DURING_AI:
        inputs += ['pilot/angle', 'pilot/throttle']
        types += ['float', 'float']

    th = TubHandler(path=cfg.DATA_PATH)
    tub = th.new_tub_writer(inputs=inputs, types=types, user_meta=meta)
    V.add(tub, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')

    if cfg.PUB_CAMERA_IMAGES:
        from donkeycar.parts.network import TCPServeValue
        from donkeycar.parts.image import ImgArrToJpg
        pub = TCPServeValue("camera")
        V.add(ImgArrToJpg(), inputs=['cam/image_array'], outputs=['jpg/bin'])
        V.add(pub, inputs=['jpg/bin'])

    if type(ctr) is LocalWebController:
        if cfg.DONKEY_GYM:
            print("You can now go to http://localhost:%d to drive your car." % cfg.WEB_CONTROL_PORT)
        else:
            print("You can now go to <your hostname.local>:%d to drive your car." % cfg.WEB_CONTROL_PORT)
    elif isinstance(ctr, JoystickController):
        print("You can now move your joystick to drive your car.")
        # tell the controller about the tub
        ctr.set_tub(tub)

        if cfg.BUTTON_PRESS_NEW_TUB:
            def new_tub_dir():
                V.parts.pop()
                tub = th.new_tub_writer(inputs=inputs, types=types, user_meta=meta)
                V.add(tub, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')
                ctr.set_tub(tub)

            ctr.set_button_down_trigger('cross', new_tub_dir)
        ctr.print_controls()

    # run the vehicle for 20 seconds
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ,
            max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config(myconfig=args['--myconfig'])

    if args['drive']:
        model_type = args['--type']
        camera_type = args['--camera']

        drive(cfg, model_path=args['--model'], use_joystick=args['--js'],
              model_type=model_type, camera_type=camera_type,
              meta=args['--meta'])

    if args['train']:
        from train import multi_train, preprocessFileList

        tub = args['--tub']
        model = args['--model']
        transfer = args['--transfer']
        model_type = args['--type']
        continuous = args['--continuous']
        aug = args['--aug']
        dirs = preprocessFileList(args['--file'])

        if tub is not None:
            tub_paths = [os.path.expanduser(n) for n in tub.split(',')]
            dirs.extend(tub_paths)

        if model_type is None:
            model_type = cfg.DEFAULT_MODEL_TYPE
            print("using default model type of", model_type)

        multi_train(cfg, dirs, model, transfer, model_type, continuous, aug)

