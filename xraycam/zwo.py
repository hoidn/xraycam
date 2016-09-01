import zmq
import time
import numpy as np

from multiprocess import Process
import os

from . import utils
from .camalysis import _plot_histogram
from . import zmq_comm

# communication with ZWO camera capture program
context = zmq.Context()

def worker_function(arr):
    return arr.copy()
def worker_process():
    zmq_comm.launch_worker(worker_function)

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
    def __init__(self, run_name = '', photon_value = 45., htime = '10s'):
        if run_name is None:
            self.name = str(time.time())
        else:
            self.name = run_name

        self._time_start = time.time()
        #self._total_time = self.time_to_numexposures(htime) # Total exposure time

        self.photon_value = photon_value
        self.initial_array = self.get_array()

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

# TODO: test multiple-worker configuration
# set daemon attribute to True so that child processes die with the parent
p1 = Process(target = worker_process, args = ())
p1.daemon = True
p1.start()

p2 = Process(target = sink_process, args = ())
p2.daemon = True
p2.start()

# Launch OAcapture
# TODO: This process shouldn't persist when the parent dies.
os.system('oacapture &> /dev/null &')
