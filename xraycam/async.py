from __future__ import absolute_import
from Queue import Queue
import threading


class IterThread(object):
    u"""
    Evaluate the elements of an iterable in a background thread, incrementally
    aggregating return values. Instances of this class behave as lists containing all
    so-far evaluated elements of the input (as of the moment of invocation of the
    instance's __iter__ method).
    """
    def __init__(self, iterable):
        t = threading.Thread(target = self._eval_iter, args = (iterable,))
        t.daemon = True
        t.start()
        self.thread = t
        self.results = []
        self.q = Queue()

    def _eval_iter(self, _iterable):
        u"""
        Arguments:
            _iterable : iterable
        Iterates through _iterable, putting returned values in q.
        """
        for elem in _iterable:
            self.q.put(elem)

    def __iter__(self):
        while not self.q.empty():
            self.results.append(self.q.get())
        return iter(list(self.results))

    def __getitem__(self, key):
        return list(self.__iter__()).__getitem__(key)
