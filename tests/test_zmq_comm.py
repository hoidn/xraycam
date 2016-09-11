from multiprocess import Process
import zmq
from xraycam import zmq_comm
from xraycam import zwo
import numpy as np
import sys
import time

# TODO: finish this

def start_ventilator(delay = 0.):
    import time
    context = zmq.Context()

    # Socket to receive messages on
    sender = context.socket(zmq.PUSH)
    sender.bind(zmq_comm.ventilator_addr)
    sender.setsockopt(zmq.CONFLATE, 1)

    #test_arr = np.ones((10, 10), dtype = 'uint8')
    # 10x10 array of ones
    test_buffer =  b'\x0a\x00\x0a\x00' + 100 * b'\x01'
    while True:
        time.sleep(delay)
        sender.send(test_buffer, copy = True, track = False)
        #print ('sending array')
        #time.sleep(0.1)

def start_ventilator_bigarr(delay = 0.):
    """
    push out 1920 x 1080 arrays filled by 10s.
    """
    context = zmq.Context()

    # Socket to receive messages on
    sender = context.socket(zmq.PUSH)
    sender.bind(zmq_comm.ventilator_addr)
    sender.setsockopt(zmq.CONFLATE, 1)
    test_buffer =  b'\x38\x04\x80\x07' + (1920 * 1080) * b'\x02'
    while True:
        time.sleep(delay)
        sender.send(test_buffer, copy = True, track = False)
#    test_arr = 10 * np.ones((1080, 1920), dtype = np.uint8)
#    print ( test_arr )
#    print(test_arr.shape)
#    while True:
#        zmq_comm.send_array(sender, test_arr, copy = True, track = False)
#        time.sleep(delay)

def worker_function(arr):
    return 2 * arr.copy()

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

def client_process():
    # Set up communication with ZWO camera capture program
    context = zmq.Context()
    def poll():
        socket = context.socket(zmq.SUB)
        socket.connect(zmq_comm.client_addr)
        socket.setsockopt(zmq.SUBSCRIBE, b'')
        result = zmq_comm.recv_array(socket)
        socket.close()
        return result
    while True:
        print ('client waiting.....')
        arr = poll()
        print ('CLIENT: shape: ', arr.shape, 'sum: ', np.sum(arr))
        time.sleep(3)
        

def test1():
    """
    Run the ventilator defined in this function.
    """
    Process(target = start_ventilator, args = ()).start()
    #time.sleep(0.1)
    Process(target = worker_process, args = ()).start()
    Process(target = sink_process, args = ()).start()

def test2():
    """
    Launch planetary_imager and process its data
    """
    import os
    Process(target = worker_process, args = ()).start()
    Process(target = sink_process, args = ()).start()
    os.system('planetary_imager &')

def test_client():
    """
    Launch planetary_imager and process its data
    """
    Process(target = start_ventilator, args = ()).start()
    Process(target = worker_process, args = ()).start()
    Process(target = sink_process, args = ()).start()
    Process(target = client_process, args = ()).start()

def test_worker_sink():
    """
    Launch planetary_imager and process its data
    """
    from multiprocess import Process
    import os
    Process(target = worker_process, args = ()).start()
    Process(target = sink_process, args = ()).start()

def test_kill():
    zwo.init_workers()
    zwo.init_sink()
    Process(target = start_ventilator).start()
    time.sleep(1)
    while True:
        zwo.kill_workers()
        time.sleep(0.5)
        zwo.init_workers()
        time.sleep(0.5)

def test_replace(delay = 0.):
    zwo.init_workers()
    zwo.init_sink()
    Process(target = start_ventilator, args = (delay,)).start()
    while True:
        time.sleep(10)
        zwo.replace_workers(zwo.dummy_worker)

def test_replace_decluster(delay = 0.):
    worker = zwo.make_worker_function(1, decluster = True)
    #Process(target = worker).start()
    zwo.init_workers(worker)
    zwo.init_sink()
    Process(target = start_ventilator_bigarr, args = (delay,)).start()
    while True:
        time.sleep(10)
        zwo.replace_workers(worker)
