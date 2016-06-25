import _thread
import time
from threading import Thread

import os
def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

def threaded_func(arg):
    if os.path.exists('foo'):
        os.remove('foo')
    touch('foo')


def test_thread2():
    thread = Thread(target = threaded_func, args = ('b'))
    thread.start()
    time.sleep(.1)
    assert os.path.exists('foo')
    #thread.join()
    os.remove('foo')

