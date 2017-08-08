import zwoasi
import time
from xraycam.camconfig import cameraconfig
from xraycam.zmq_comm import send_array
import zmq
import logging
from multiprocess import Process

logger = logging.getLogger('Zwo')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('zwo-logging.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

#Address for messages from ventilator to worker
ventilator_addr = "tcp://127.0.0.1:5555"

#Directory for camera drivers
ZWO_LIB_DIRECTORY = '/home/xrays/zwodriver/lib/x64/libASICamera2.so'

zwoasi.init(ZWO_LIB_DIRECTORY)


def config_xray_camera(camera, config = cameraconfig, delay=0.0):
    startconfig = camera.get_control_values()
    for k in config:
        if k is 'ImageType':
            set_xray_camera_image_type(camera,config[k])
        elif startconfig[k] != config[k]['set_value']:
            print('Initializing',k,'to value:',config[k]['set_value'])
            camera.set_control_value(config[k]['controlname'],config[k]['set_value'])
            time.sleep(delay)
    return confirm_xray_camera_config(camera,config,delay = delay)

def confirm_xray_camera_config(camera, config, delay=0.0, autofalse = True):
    currentconfig = camera.get_control_values()
    for k in config:
        if k is not 'ImageType':
            assert currentconfig[k] == config[k]['set_value'],\
            "Error, camera setting doesn't match config:"+str(k)
            time.sleep(delay)
            if autofalse:
                if k in ('Gain','Exposure'):
                    assert not camera.get_control_value(config[k]['controlname'])[1],\
                    'Error, '+str(k)+' is set to auto.'
    return currentconfig

def set_xray_camera_image_type(camera, imgtype='RAW8'):
    """Set the image type of the camera.  For monochrome chips, valid image types are RAW8 and RAW16."""
    imgmap = {'RAW8':0,'RAW16':2}
    if camera.get_image_type() != imgmap[imgtype]:
        camera.set_image_type(imgmap[imgtype])
    time.sleep(0.25)
    assert camera.get_image_type() == imgmap[imgtype],'Image type not set properly.'

def launch_camera(doconfig=True):
    camera = zwoasi.Camera(0)
    if doconfig:
        config_xray_camera(camera)
    return camera

def capture_worker(flags = 0, copy = True, track = False):
    logger.info('Launching camera...')
    camera = launch_camera()
    time.sleep(0.5)
    camera.start_video_capture()

    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    sender.bind(ventilator_addr)
    try:
        while True:
            frame = camera.capture_video_frame()
            send_array(sender, frame, flags = flags, copy = copy, track = track)
    except KeyboardInterrupt:
        logger.info('Camera worker received interrupt, stopping.')
    finally:
        camera.stop_video_capture()
        camera.close()
        sender.close()
        context.term()

def start_capture_process():
    p = Process(target=capture_worker)
    p.daemon = True
    p.start()
    global process
    process = p
    return p

process = None