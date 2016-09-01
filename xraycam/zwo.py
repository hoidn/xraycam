import zmq
import time
import numpy as np
from ctypes import c_int
import numpy.ctypeslib as npct

from multiprocess import Process
import os

from . import zmq_comm
from . import utils

PKG_NAME = __name__.split('.')[0]

# communication with ZWO camera capture program
context = zmq.Context()

def do_decluster(arr2d, threshold, dtype = np.uint8):
    # TODO: does ascontiguousarray slow things down if arr2d is already
    # contiguous?
    # TODO: add support for uint8 arrays in the c module
    arr1d = np.ascontiguousarray(arr2d.ravel(), dtype = np.uint16)
    declustered = np.zeros_like(arr1d)
    arr_uint = npct.ndpointer(dtype = np.uint16, ndim = 1, flags = 'C_CONTIGUOUS')
    # load the c extension
    try:
        libcd = npct.load_library("libclusters", utils.resource_path("../lib/",  PKG_NAME))
    except:
        print (os.listdir('.'))
        print (os.listdir('../lib/'))
        raise

    libcd.searchFrame_array.restype = None
    libcd.searchFrame_array.argtypes = [arr_uint, arr_uint, c_int, c_int, c_int]

    dimx, dimy = np.shape(arr2d)
    libcd.searchFrame_array(declustered, arr1d, dimx, dimy, threshold)
    return declustered.reshape(np.shape(arr2d))

def make_worker_function(threshold, window_min = 0, window_max = 255, decluster = False):
    def worker_function(arr):
        new = arr.copy() # TODO: zero-copy
        new[new < window_min] = 0
        new[new > window_max] = 0
        if decluster:
            return do_decluster(new, threshold)
        else:
            return new
    def worker_process():
        zmq_comm.launch_worker(worker_function)
    return worker_process

def launch_process(f):
    p = Process(target = f, args = ())
    p.daemon = True
    p.start()
    return p

def sink_function(current, arr):
    #arr = np.frombuffer(msg, dtype = 'uint8').reshape(10, 10)
    import time
    #time.sleep(0.1)
    if current is None:
        print ('hello world')
        #return np.zeros((10, 10), dtype = 'uint32')
        return arr.astype('uint32')
    else:
        print ('shape: ', current.shape, 'sum: ', np.sum(current))
        return current + arr
def sink_process():
    zmq_comm.start_sink_routine(sink_function)

# TODO: common base class for datarun, or at least an abstract base
# class defining the interface.
class DataRun:
    """
    TODO.
    """
    def __init__(self, run_name = '', photon_value = 45., window_min = 0,
            window_max = 255, threshold = 10, decluster = False, htime = '10s'):
        if run_name is None:
            self.name = str(time.time())
        else:
            self.name = run_name

        self._time_start = time.time()
        #self._total_time = self.time_to_numexposures(htime) # Total exposure time

        self.photon_value = photon_value
        self.initial_array = self.get_array()

        # TODO: test multiple-worker configuration
        worker_function = make_worker_function(threshold, window_min = window_min,
            window_max = window_max, decluster = decluster)
        # Kill the old worker and launch a new one

        worker.terminate()
        loc['worker'] = launch_process(worker_function)

    def stop(self):
        """
        Stop the acquisition.
        """
        final_array = self.get_array()
        self.get_array = lambda: final_array
        self._total_time = time.time() - self._time_start

    def acquisistion_time(self):
        elapsed = time.time() - self._time_start 
        #if elapsed > self._total_time:
        try:
            return self._total_time
        except AttributeError:
            return elapsed

    def get_array(self):
        """
        Get the sum of exposures in this data run.
        """
        # TODO: package the server-side code 
        # TODO: implement communication to the sink so that its output
        # can be reset to 0 from here.
        socket = context.socket(zmq.SUB)
        socket.connect(zmq_comm.client_addr)
        socket.setsockopt(zmq.SUBSCRIBE, b'')
        result = zmq_comm.recv_array(socket)
        socket.close()
        try:
            return result - self.initial_array
        except AttributeError:
            return result

    def get_histograms(self):
        raise NotImplementedError()

# initial worker
loc = locals()
worker = launch_process(make_worker_function(0))

launch_process(sink_process)

# Launch OAcapture
# TODO: This process shouldn't persist when the parent dies.
# set daemon attribute to True so that child processes die with the parent
os.system('oacapture &> /dev/null &')
