import zmq
import time
import numpy as np
from ctypes import c_int
import numpy.ctypeslib as npct
import dill
import humanfriendly

from multiprocess import Process
from threading import Thread
import os

from . import zmq_comm
from . import utils

PKG_NAME = __name__.split('.')[0]

NCORES = 5

# communication with ZWO camera capture program
context = zmq.Context()

def cache_put(obj, key = None, cachedir = 'cache/'):
    """
    Push an object to a persistent cache stored on disk.
    """
    if not os.path.isdir(cachedir):
        os.makedirs(cachedir)
    if key is None:
        key = utils.hash_obj(obj)
    with open(cachedir + '/' + key, 'wb') as f:
        dill.dump(obj, f)

def cache_get(key, cachedir = 'cache/'):
    with open(cachedir + '/' + key, 'rb') as f:
        obj = dill.load(f)
    return obj

def do_decluster(arr2d, threshold, dtype = np.uint8):
    # TODO: does ascontiguousarray slow things down if arr2d is already
    # contiguous?
    # TODO: add support for uint8 arrays in the c module
    arr1d = np.ascontiguousarray(arr2d.ravel(), dtype = np.uint8)
    declustered = np.zeros(np.shape(arr1d), dtype = np.uint32)
    arr_uint = npct.ndpointer(dtype = np.uint8, ndim = 1, flags = 'C_CONTIGUOUS')
    arr_uint32 = npct.ndpointer(dtype = np.uint32, ndim = 1, flags = 'C_CONTIGUOUS')
    # load the c extension
    libcd = npct.load_library("libclusters", utils.resource_path("../lib/",  PKG_NAME))

    libcd.searchFrame_array_8.restype = None
    libcd.searchFrame_array_8.argtypes = [arr_uint32, arr_uint, c_int, c_int, c_int]

    dimx, dimy = np.shape(arr2d)
    libcd.searchFrame_array_8(declustered, arr1d, dimx, dimy, threshold)
    return declustered.reshape(np.shape(arr2d))

def make_worker_function(threshold, window_min = 0, window_max = 255, decluster = True):
    def worker_function(arr):
        """
        Input:
            arr : np.ndarray of type uint8

        Returns np.ndarray of the same type and shape.
        """
        new = arr.copy() # TODO: zero-copy
        if decluster:
            new = do_decluster(arr, threshold)
        new[new > window_max] = 0
        new[new < window_min] = 0
        return new
    def worker_process():
        zmq_comm.launch_worker(worker_function)
    return worker_process

dummy_worker = make_worker_function(0, decluster = False)
def init_workers(worker = dummy_worker):
    """
    Start a dummy worker process that keeps up with frames from the ventilator.
    """
    loc['workers'] = [launch_process(worker) for n in range(NCORES)]

def kill_workers():
    controller = context.socket(zmq.PUB)
    controller.bind(zmq_comm.controller_addr)
    # Delay to avoid slow joiner syndrome
    time.sleep(0.2)
    print("controller sending KILL signal")
    controller.send(b'KILL')
    controller.close()

def replace_workers(worker_function):
    """
    Replace all current workers by processes running worker_function.
    """
    kill_workers()
    # Delay to synchronize with kill signal
    time.sleep(0.05)

    new_workers = []
    while workers:
        workers.pop()
        new_workers.append(launch_process(worker_function))
    while new_workers:
        workers.append(new_workers.pop())

def launch_process(f):
    p = Process(target = f, args = ())
    p.daemon = True
    p.start()
    return p

def sink_function(current, arr):
    #arr = np.frombuffer(msg, dtype = 'uint8').reshape(10, 10)
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

def init_sink():
    launch_process(sink_process)

class ZRun:
    """
    TODO.
    """
    def __init__(self, run_prefix = '',  window_min = 0,
            window_max = 255, threshold = 10, decluster = True, htime = None):
            self.name = run_prefix
            self.attrs = ['_time_start', 'initial_array', '_total_time', '_final_array']
            # TODO: entire dict in one file
            try:
                attrs = self.attrs
                keys = [run_prefix + a for a in attrs]
                values = [cache_get(k, cachedir = 'cache/') for k, a in zip(keys, attrs)]
                for a, v in zip(attrs, values):
                    self.__dict__[a] = v
                print ("Loaded from cache.")
            # Do an actual data collection
            except FileNotFoundError:
                self._time_start = time.time()
                self.initial_array = self.get_array()

                worker_function = make_worker_function(threshold, window_min = window_min,
                    window_max = window_max, decluster = decluster)

                # Kill the old workers and launch new ones
                replace_workers(worker_function)

                if htime is not None:
                    def timeit():
                        print( "starting acquisition")
                        time.sleep(humanfriendly.parse_timespan(htime))
                        self.stop()
                        print("stopped acquisistion")
                    t_thread = Thread(target = timeit, args = ())
                    t_thread.start()



    def stop(self):
        """
        Stop the acquisition.
        """
        self._final_array = self.get_array()
        self._total_time = time.time() - self._time_start

        keys = [self.name + a for a in self.attrs]
        for k, a in zip(keys, self.attrs):
            cache_put(self.__dict__[a], key = k)

        self.replace_workers(dummy_worker)

    def acquisition_time(self):
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
        try:
            return self._final_array
        except AttributeError:
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

# Launch OAcapture
# TODO: This process shouldn't persist when the parent dies.
def init_ventilator():
    os.system('oacapture &> /dev/null &')

def init():
    """
    Initialize all processes.
    """
    init_workers()
    init_sink()
    init_ventilator()

# For modifying module-level variables inside functions
loc = locals()
workers = []

