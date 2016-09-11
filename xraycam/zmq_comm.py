import os
import numpy as np
import zmq
import signal

# Address for messages from ventilator to workers
ventilator_addr = "tcp://127.0.0.1:5555"

# Address for messages from workers to sink
sink_addr =  "tcp://127.0.0.1:5556"

# Address for PUB messages from the sink to client applications
server_addr = "tcp://127.0.0.1:5557"

client_addr = "tcp://127.0.0.1:5558"

controller_addr = "tcp://127.0.0.1:5559"

def send_array(socket, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    #print(md)
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    md = socket.recv_json(flags=flags)
    #print("recv: ", md)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    A = np.frombuffer(msg, dtype=md['dtype'])
    return A.reshape(md['shape'])


def launch_worker(f, flags = 0, copy = True, track = False):
    """
    Run a worker that evaluates the function f on numpy arrays parsed
    from ventilator messages and sends the results, assumed to also be a
    numpy array, to the sink.

    Incoming message data must be in the following format: uint16
    (height), uint16 (width), uint8 array[].
    """
    print("Worker LAUNCHING")
    nframe = 0
    import struct
    context = zmq.Context()

    # Socket to receive messages on
    receiver = context.socket(zmq.PULL)
    receiver.connect(ventilator_addr)

    # Socket to send messages to
    sender = context.socket(zmq.PUSH)
    sender.connect(sink_addr)

    controller = context.socket(zmq.SUB)
    controller.connect(controller_addr)
    controller.setsockopt(zmq.SUBSCRIBE, b'')

    poller = zmq.Poller()
    poller.register(receiver, zmq.POLLIN)
    poller.register(controller, zmq.POLLIN)
    #print("Worker waiting for data...")

    def _exit():
        receiver.close()
        sender.close()
        controller.close()
        context.term()
        os.kill(os.getpid(), signal.SIGTERM)
    try:
        while True:
            socks = dict(poller.poll())
            if receiver in socks and socks[receiver] == zmq.POLLIN:
                message_in = receiver.recv(copy = copy, track = track)
                height, width = struct.unpack('HH', message_in[:4])
                arr_in = np.frombuffer(message_in[4:], dtype = 'uint8').reshape((height, width))
#                print('dims: ', arr_in.shape)
#                print('dtype: ', arr_in.dtype)

                arr_out = f(arr_in)
                send_array(sender, arr_out, flags = flags, copy = copy, track = track)
                #print("Worker received frame %d" % nframe)
                nframe += 1
            if controller in socks and socks[controller] == zmq.POLLIN:
                msg = controller.recv()
                if msg == b'KILL':
                    print("Worker received KILL signal")
                    _exit()

    except KeyboardInterrupt:
        print("Worker received interrupt, stopping.")
    finally:
        _exit()

def start_sink_routine(f, flags = 0, copy = True, track = False):
    """
    f(x, y) is an update function that takes a current value, x,
    and incremental value, y (i.e. a single result from a worker).

    f must be able to initialize on x == None.

    This function continually evaluates f on worker messages and
    sends its most recent output to the PUB socket.
    """
    context = zmq.Context()

    # Socket to receive messages on
    receiver = context.socket(zmq.PULL)
    receiver.bind(sink_addr)

    # Socket to send messages to
    publisher = context.socket(zmq.PUB)
    publisher.bind(client_addr)
    publisher.setsockopt(zmq.CONFLATE, 1)
    output = None
    try:
        while True:
            arr_in = recv_array(receiver, flags = flags, copy = copy, track = track)
            output = f(output, arr_in)
            send_array(publisher, output, flags = flags, copy = copy, track = track)
    except KeyboardInterrupt:
        print("Sink received interrupt, stopping.")
    finally:
        publisher.close()
        receiver.close()
        context.term()
