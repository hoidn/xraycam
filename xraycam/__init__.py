import time
from . import zwocapture
from . import config
from . import zwo
from .zwocapture import check_status

def start():
	"""Initiates back-end processes, and begins continuously pulling
	frames from the camera.
	"""
	global camera
	camera = zwocapture.CaptureProcess()
	zwo.init_workers()
	camera.start()

def shutdown():
    """Stops camera feed, closes camera, and shuts down back-end processes.
    """
    global camera
    camera.shutdown()
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