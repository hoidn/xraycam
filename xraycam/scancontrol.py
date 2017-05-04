import threading, time
import numpy as np
from . import camcontrol

def _check_for_data_files(prefixlist):
    import os
    cachecontents = os.listdir('cache')
    exists=[]
    for f in prefixlist:
        if f+'_final_array' in cachecontents:
            exists.append([True])
        else:
            exists.append([False])
    return exists


class angle_scan:

    def __init__(self, duration, anglerange, stepsize, prefix, **kwargs):
        self.duration = duration
        self.anglerange = anglerange
        self.stepsize = stepsize
        self.prefix = prefix
        self.runset = camcontrol.RunSet()
        self.kwargs = kwargs
        self.anglelist = np.arange(self.anglerange[0],
            self.anglerange[1]+self.stepsize,self.stepsize)
        self.prefixes = [self.prefix+'_%d' % angle for angle in self.anglelist]
        self.runthread = None
        self.continuescan = False

    def generate_actionlist(self):
        self.actionlist = []
        for ang in self.anglelist:
            datarunfunc = lambda angle = ang: camcontrol.DataRun(
                run_prefix=self.prefix+'_%d'%angle, htime=self.duration,
                runparam={'angle':angle},**self.kwargs)
            movefunc = lambda angle = ang: ardstep.go_to_degree(angle)
            self.actionlist.append([movefunc,datarunfunc])

    def run_scan(self):
        try:
            self.load_scan()
        except IOError:
            self.generate_actionlist()
            self.runthread = ScanThread(self.runset,self.actionlist)
            self.runthread.start()

    def load_scan(self,doreload=False):
        checklist = _check_for_data_files(self.prefixes)
        if not all(checklist):
            if not any(checklist):
                raise IOError('Files not found, cannot load data.')
            else:
                raise IOError('Some files found, scan partially complete. \n \
                    Set continuescan attribute to True to resume scan.')
        else:
            if self.runset.dataruns is None or doreload:
                self.runset.dataruns=[]
                self.generate_actionlist()
                for a in self.actionlist:
                    action, datarun = a
                    self.runset.insert(datarun())
            else:
                print('All files already loaded. Run load_scan with doreload=True to reload files.')



    def angle_plot(self,start=0,end=-1,show=True,**kwargs):
        if self.runset.dataruns is None:
            print('Error: datafiles not loaded.')
        else:
            angles = [x.runparam['angle'] for x in self.runset.dataruns]
            counts = [x.counts_per_second(start=start,end=end) for x in self.runset.dataruns]
            camcontrol.plt.plot(angles,counts,**kwargs)
            camcontrol.plt.xlabel('Angle (deg)')
            camcontrol.plt.ylabel('Counts/sec in region '+str(start)+':'+str(end))
            if show:
                camcontrol.plt.show()

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
        self._wait_current_complete()
        try:
            action, datarun = self.actionlist.pop(0)
            action()
            run = datarun()
            self.current = run
        except IndexError:
            raise StopIteration
        return run

class ScanThread(threading.Thread):
    
    def __init__(self,runset,actionlist):
        threading.Thread.__init__(self)
        self.runset = runset
        self.actionlist = actionlist
        self.current = None

    def run(self):
        for el in self.actionlist:
            self._wait_current_complete()
            action, datarun = el
            print('moving before scan')
            action()
            dr = datarun()
            print('scan started')
            self.current = dr
            self.runset.insert(dr)
        print('Congratulations, scan is complete! Have a nice day ;)')

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




