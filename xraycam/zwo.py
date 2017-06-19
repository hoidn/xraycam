import zmq
import time
import dill
import humanfriendly

from multiprocess import Process
from threading import Thread
import os

from . import zmq_comm
from . import utils
from . import declustering

PKG_NAME = __name__.split('.')[0]

NCORES = 1

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


def make_worker_function(threshold, window_min = 0, window_max = 255, decluster = True):
    def worker_function(arr):
        new = arr.copy() # TODO: zero-copy
        if decluster:
            new = declustering.do_decluster(arr, threshold)
        new[new > window_max] = 0
        new[new < window_min] = 0
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
    loc = locals()
    workers = [launch_process(make_worker_function(0, decluster = False)) for n in range(NCORES)]

class ZRun:
    """
    TODO.
    """
    def __init__(self, run_prefix = '', runparam = {}, window_min = 0,
            window_max = 255, threshold = 10, decluster = True, htime = None, norunparam = False, loadonly = False):
            self.name = run_prefix
            if norunparam:
                self.attrs = ['_time_start', 'initial_array', '_total_time', '_final_array']
            else:
                self.attrs = ['_time_start', 'initial_array', '_total_time', '_final_array','_run_parameters']
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
                if not loadonly:
                    self._time_start = time.time()
                    self.initial_array = self.get_array()
                    self._run_parameters = runparam

                    worker_function = make_worker_function(threshold, window_min = window_min,
                        window_max = window_max, decluster = decluster)

                    # Kill the old workers and launch new ones
                    self.replace_workers(worker_function)

                    if htime is not None:
                        def timeit():
                            print( "starting acquisition")
                            time.sleep(humanfriendly.parse_timespan(htime))
                            self.stop()
                            print("stopped acquisition")
                        t_thread = Thread(target = timeit, args = ())
                        t_thread.start()
                else:
                    print("Loadonly set true, but file not found in cache.")


    def replace_workers(self, worker_function):
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
        #loc['worker'] = launch_process(worker_function)


    def stop(self):
        """
        Stop the acquisition.
        """
        try:
            if self._final_array.any():
                print('Run already stopped.')
        except AttributeError:
            self._total_time = time.time() - self._time_start
            self._final_array = self.get_array()
            # self._total_time = time.time() - self._time_start

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

# initial worker
loc = locals()
dummy_worker = make_worker_function(0, decluster = False)
workers = [launch_process(dummy_worker) for n in range(NCORES)]

time.sleep(0.2)
launch_process(sink_process)

# Launch OAcapture
# TODO: This process shouldn't persist when the parent dies.
# set daemon attribute to True so that child processes die with the parent
#os.system('oacapture &> /dev/null &')
