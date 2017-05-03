import threading, time
import numpy as np
from . import camcontrol

class angle_scan:

    def __init__(self, duration, anglerange, stepsize, prefix, **kwargs):
        self.duration = duration
        self.anglerange = anglerange
        self.stepsize = stepsize
        self.prefix = prefix
        self.runset = camcontrol.RunSet()
        self.kwargs = kwargs
        self.anglelist = np.arange(self.anglerange[0],self.anglerange[1],self.stepsize)

    def generate_actionlist(self):
        prefixes = [self.prefix + '_%d' % deg for deg in self.anglelist]
        self.actionlist = []
        for angle in self.anglelist:
            datarunfunc = lambda: camcontrol.DataRun(
                prefix=self.prefix+'_%d'%angle, runparam={'angle':angle},**self.kwargs)
            movefunc = ardstep.go_to_degree(angle)
            self.actionlist.append[[datarunfunc,movefunc]]

    def run_scan(self):
        from xraycam import async
        self.generate_actionlist()
        self.runset.dataruns = async.IterThread(ActionSequence(self.actionlist))



class ActionSequence:
    """
    Takes in a nested list of the form 
    [[action1,datarun1],[action2,datarun2],...]
    Takes action1, then evaluates datarun1 etc.
    """

    def __init__(self,actionlist):
        self.actionlist = actionlist
        self.current = None
    
    def __iter__(self):
        return self

    def _wait_current_complete(self):
        """
        Wait until the current run has completed.
        """
        import time
        # Wait until current run is complete
        if self.current is not None:
            prev_time = self.current.acquisition_time()
            while True:
                cur_time = self.current.acquisition_time()
                if cur_time != prev_time:
                    prev_time = cur_time
                    time.sleep(1)
                else:
                    break

    def __next__(self):
        try:
            action, datarun = self.actionlist.pop(0)
            run = datarun()
            self.current = run
        except IndexError:
            raise StopIteration
        self._wait_current_complete()
        action()
        return run
