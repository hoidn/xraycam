import threading, time
from . import camcontrol

class angle_scan():

    def __init__(self, duration, anglerange, stepsize, filename):

        self.duration = duration
        self.anglerange = anglerange
        self.stepsize = stepsize
        self.filename = filename
        self.runset = camcontrol.RunSet()

    def run_scan(self):
        self.runset

class scan_thread(threading.Thread):

    def __init__(self,stuff):
        self.stuff = stuff

class ActionSequence:

    
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
        self._wait_current_complete()
        try:
            run = self.funcalls.pop(0)()
            self.current = run
        except IndexError:
            raise StopIteration
        return run
