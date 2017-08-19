import zmq
import time
import dill
import humanfriendly
import json
import numpy as np

from multiprocess import Process
from threading import Thread, Event
import os

from . import zmq_comm
from . import declustering
from . import config
from . import zwocapture

NCORES = 1
CAPTURE = 'zwopython'

# communication with ZWO camera capture program
context = zmq.Context()

def cache_put(obj, key = None, cachedir = 'cache/'):
    """
    Push an object to a persistent cache stored on disk.
    """
    if not os.path.isdir(cachedir):
        os.makedirs(cachedir)
    if key is None:
        raise ValueError("Need cache key to write to cache.")
    with open(cachedir + '/' + key, 'wb') as f:
        dill.dump(obj, f)

def cache_get(key, cachedir = 'cache/'):
    with open(cachedir + '/' + key, 'rb') as f:
        obj = dill.load(f)
    return obj


def make_worker_function(threshold, window_min = 0, window_max = 255, decluster = True):
    def worker_function(arr):
        new = arr.copy() # TODO: zero-copy
        if decluster:
            new = declustering.do_decluster(arr, threshold)
        new[new > window_max] = 0
        new[new < window_min] = 0
        return new
    def worker_process():
        if CAPTURE == 'oacapture':
            zmq_comm.launch_worker_oa(worker_function)
        elif CAPTURE == 'zwopython':
            zmq_comm.launch_worker(worker_function)
    return worker_process

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
        #print ('shape: ', current.shape, 'sum: ', np.sum(current))
        return current + arr
def sink_process():
    zmq_comm.start_sink_routine(sink_function)

def init_workers():
    """
    Start a dummy worker process that keeps up with frames from the ventilator.
    """
    global workers
    workers = [launch_process(make_worker_function(0, decluster = False)) 
                for n 
                in range(NCORES)]
    global sink
    sink = launch_process(sink_process)

def shutdown_workers():
    for w in workers:
        w.terminate()
    sink.terminate()

def replace_workers(worker_function):
    """
    Replace all current workers by processes running worker_function.
    """
    # Terminate and replace the current worker processes
    new_workers = []
    def replace_worker():
        new_workers.append(launch_process(worker_function))
        old = workers.pop()
        old.terminate()
        
    while workers:
        replace_worker()
    while new_workers:
        workers.append(new_workers.pop())

def _validate_savedir():
    #Ensure save configuration valid
    if not os.path.isdir(config.saveconfig['Directory']):
        raise IOError('Save directory does not exist.  Check saveconfig.')

class ZRun:
    """
    TODO.
    """
    def __init__(self, run_prefix = '', window_min = 0, window_max = 255, 
        threshold = 0, decluster = True, duration = None, loadonly = False, 
        saveonstop = True, photon_value = 1):

        _validate_savedir()
        
        self.name = run_prefix
        self.window_min = window_min
        self.window_max = window_max
        self.threshold  = threshold
        self.decluster = decluster
        self.duration = duration
        self.loadonly = loadonly
        self.saveonstop = saveonstop

        try:
            self.load()
            print ("Run loaded from disk.")

        except FileNotFoundError:
            if not loadonly:
                self._time_start = time.time()

                worker_function = make_worker_function(threshold, 
                    window_min = window_min, window_max = window_max, 
                    decluster = decluster)

                # Kill the old workers and launch new ones
                replace_workers(worker_function)

                self.initial_array = self.get_array()
                self.initialcamstatus = zwocapture.check_status()


                if duration is not None:
                    def timeit(e):
                        print("starting acquisition")
                        e.wait(duration)
                        if not e.is_set():
                            self.stop()
                            print("stopped acquisition")
                    self.stopevent = Event()                      
                    t_thread = Thread(target = timeit, args = (self.stopevent,))
                    t_thread.start()
            else:
                raise FileNotFoundError('Loadonly set true, but file not found.')

    def stop(self):
        """Stop the acquisition.

        The run parameters (name, time, keywords) and
        resulting array are written to disk after stopping.

        Parameters are saved as: run_prefix_parameters
        Final array is saved as: run_prefix_array
        """
        try:
            if self.final_array.any():
                print('Run is already stopped.')
        except AttributeError:
            self._total_time = time.time() - self._time_start
            self.final_array = self.get_array()
            self.finalcamstatus = zwocapture.check_status()

            if self.saveonstop:
                self.save()

            if self.duration is not None:
                self.stopevent.set()

            replace_workers(dummy_worker)

    def save_parameters(self):
        """Utility function that saves parameters of the run to disk.
        This function is called after self.stop().
        """
        savedict={}
        for k,v in self.__dict__.items():
            if type(v) is not np.ndarray:
                if type(v) is not Event:
                    savedict[k]=v
        with open(config.saveconfig['Directory']+self.name+'_parameters','w') as file:
            json.dump(savedict, file)

    def save(self):
        """Saves the run parameters and array to disk.
        """
        np.save(config.saveconfig['Directory']+self.name+'_array',self.final_array)
        self.save_parameters()

    def load(self):
        """Loads the run parameters and array from disk.
        """
        with open(config.saveconfig['Directory']+self.name+'_parameters','r') as file:
            self.__dict__ = json.load(file)
        self.final_array = np.load(config.saveconfig['Directory']+self.name+'_array.npy')


    def acquisition_time(self):
        """Returns the length of time run has exposed for.
        """
        elapsed = time.time() - self._time_start 
        try:
            return self._total_time
        except AttributeError:
            return elapsed

    def get_array(self):
        """
        Get the sum of exposures in this data run.
        """
        try:
            return self.final_array
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

    def block_until_complete(self, waitperiod = 0.1):
        prev_time = self.acquisition_time()
        time.sleep(waitperiod)
        while True:
            cur_time = self.acquisition_time()
            if cur_time != prev_time:
                prev_time = cur_time
                time.sleep(waitperiod)
            else:
                break

dummy_worker = make_worker_function(0, decluster = False)
