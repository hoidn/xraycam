import zmq
import zmq_comm
import numpy as np
import sys
import time

# TODO: finish this

def start_ventilator():
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
        time.sleep(0.1)
        sender.send(test_buffer, copy = True, track = False)
        #print ('sending array')
        #time.sleep(0.1)

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
        #print ('shape: ', current.shape, 'sum: ', np.sum(current))
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
    from multiprocess import Process
    Process(target = start_ventilator, args = ()).start()
    #time.sleep(0.1)
    Process(target = worker_process, args = ()).start()
    Process(target = sink_process, args = ()).start()

def test2():
    """
    Launch planetary_imager and process its data
    """
    from multiprocess import Process
    import os
    Process(target = worker_process, args = ()).start()
    Process(target = sink_process, args = ()).start()
    os.system('planetary_imager &')

def test_client():
    """
    Launch planetary_imager and process its data
    """
    from multiprocess import Process
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

#test2()
