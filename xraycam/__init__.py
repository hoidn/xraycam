import time
from . import zwocapture
from . import config
from . import zwo
from .zwocapture import check_status
import multiprocess

camera = zwocapture.CaptureProcess()

def start():
    """Initiates back-end processes, and begins continuously pulling
    frames from the camera.
    """
    global camera
    zwo.init_workers()
    camera.start()

def shutdown():
    """Stops camera feed, closes camera, and shuts down back-end processes.
    """
    global camera
    camera.shutdown()
    zwocapture.camstatus = multiprocess.Manager().Namespace() #reinstantiaiting namespace to fix broken-pipe issue
    zwocapture.camstatus.value = None
    camera = zwocapture.CaptureProcess()
    time.sleep(0.5)
    zwo.shutdown_workers()

def set_save_directory(directory):
    """Sets the save directory for the current session.  To change the default 
    save directory, modify 'saveconfig' in config.py"""
    config.saveconfig['Directory'] = directory

def get_process_pids():
    try :
        processes = {
            'camera':camera.pid,
            'workers':[x.pid for x in zwo.workers],
            'sink':zwo.sink.pid
        }
        return processes
    except AttributeError:
        raise RuntimeError('Processes haven\'t been started by start() yet')

def set_gain(value):
    if not (value < 600) & (value > 0):
        raise ValueError('Gain must be in the range: (0,600)')
    config.cameraconfig['Gain']['set_value'] = value
    shutdown()
    time.sleep(1)
    start()