u"""
Functions for controlling logging and other output.
"""



from __future__ import absolute_import
import os
import logging
from io import open
from itertools import imap


# Control redirection of stdout:
suppress_root_print = False
stdout_to_file = False

logfile_path = u'log.txt'

logging.basicConfig(filename = logfile_path, level = logging.DEBUG)

class conditional_decorator(object):
    u"""
    From http://stackoverflow.com/questions/10724854/how-to-do-a-conditional-decorator-in-python-2-6
    """
    def __init__(self, dec, condition):
        self.decorator = dec
        self.condition = condition

    def __call__(self, func):
        if not self.condition:
            # Return the function unchanged, not decorated.
            return func
        return self.decorator(func)

def isroot():
    u"""
    Return true if the MPI core rank is 0 and false otherwise.
    """
    if u'OMPI_COMM_WORLD_RANK' not in u' '.join(list(os.environ.keys())):
        return True
    else:
        rank = int(os.environ[u'OMPI_COMM_WORLD_RANK'])
        return (rank == 0)
    
def ifroot(func):
    u"""
    Decorator that causes the decorated function to execute only if
    the MPI core rank is 0.
    """
    def inner(*args, **kwargs):
        if isroot():
            return func(*args, **kwargs)
    return inner

def stdout_to_file(path = None):
    u"""
    Decorator that causes stdout to be redirected to a text file during the
    modified function's invocation.
    """
    if path is None:
        path = u'mecana.log'
    def decorator(func):
        import sys
        def new_func(*args, **kwargs):
            stdout = sys.stdout
            sys.stdout = open(path, u'w')
            result = func(*args, **kwargs)
            sys.stdout.close()
            sys.stdout = stdout
            return result
        return new_func
    return decorator

@conditional_decorator(ifroot, suppress_root_print)
def log(*args):
    def newargs():
        if stdout_to_file:
            return (u'PID ', os.getpid(), u': ') + args
        else:
            return args
    logging.info(u' '.join(imap(unicode, args)))
    #print(*newargs())
