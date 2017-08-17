import zwoasi
import time
from xraycam.config import cameraconfig, ZWO_LIB_DIRECTORY
from xraycam.zmq_comm import send_array
import zmq
import logging
import multiprocess

logger = logging.getLogger('Zwo')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('zwo-logging.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

#Address for messages from ventilator to worker
ventilator_addr = "tcp://127.0.0.1:5555"

#Initialize zwo library
zwoasi.init(ZWO_LIB_DIRECTORY)

#shared process namespace for communicating camera status
camstatus = multiprocess.Manager().Namespace()
camstatus.value = None

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
    time.sleep(0.5)
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
    p = multiprocess.Process(target=capture_worker)
    p.daemon = True
    p.start()
    global process
    process = p
    return p

def check_status():
    try:
        status = {
        'Gain':camstatus.value['Gain'],
        'Exposure':camstatus.value['Exposure'],
        'HighSpeedMode':camstatus.value['HighSpeedMode'],
        'Temperature':camstatus.value['Temperature']/10
        }
        return status
    except TypeError:
        raise RuntimeError('Camera setting read error, wait one second and try again.')

class CaptureProcess(multiprocess.Process):
    """Background process responsible for grabbing frames from camera.

    Create the process, then call its .start() method to start the camera.
    To turn off or reset the camera, call shutdown().
    """
    def __init__(self, ns = camstatus, statusrefresh=20):
        multiprocess.Process.__init__(self)
        self.camstatus = camstatus
        self.exit = multiprocess.Event()
        self.statusrefresh = statusrefresh
        self.daemon = False

    def run(self):
        self.camera = launch_camera()
        time.sleep(0.5)
        self.camera.start_video_capture()

        context = zmq.Context()
        sender = context.socket(zmq.PUSH)
        sender.bind(ventilator_addr)

        try:
            i = 0
            while not self.exit.is_set():
                #grab and send frame
                frame = self.camera.capture_video_frame()
                send_array(sender, frame)

                #occasionally load camera status
                i += 1
                if i == self.statusrefresh:
                    self.camstatus.value = self.camera.get_control_values()
                    i = 0

        finally:
            self.camera.stop_video_capture()
            self.camera.close()
            sender.close()
            context.term()

    def shutdown(self):
        """Sets the exit flag on the process, causing the camera 
        to stop the exposure and close.
        """
        self.exit.set()

process = None