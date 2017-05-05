import threading, time
import numpy as np
from . import camcontrol
from arduinostepper import arduinostepper as ardstep

def _check_for_data_files(prefixlist):
    import os
    cachecontents = os.listdir('cache')
    exists=[]
    for f in prefixlist:
        if f+'_final_array' in cachecontents:
            exists.append(True)
        else:
            exists.append(False)
    return exists


class ScanAndAction:

    def __init__(self, runset, prefix, movetype, movelist, movefunc, continuescan, duration, **kwargs):
        self.runset = runset
        self.prefix = prefix
        self.movetype = movetype
        self.movelist = movelist
        self.movefunc = movefunc
        self.runthread = None
        self.continuescan  = continuescan
        self.duration = duration
        self.kwargs = kwargs

    def make_prefix(self,value):
        if self.movetype == 'angle':
            r= self.prefix+'_{:d}'.format(value)+self.movetype
        else:
            r= self.prefix+'_{:.2f}'.format(value)+self.movetype
        return r

    def generate_actionlist(self):
        self.actionlist = []
        for move in self.movelist:
            datarunfuncs = lambda m = move: camcontrol.DataRun(
                run_prefix=self.make_prefix(m), htime=self.duration,
                runparam={self.movetype:m},**self.kwargs)
            movefuncs = lambda m = move: self.movefunc(m)
            self.actionlist.append([movefuncs,datarunfuncs])    

    def run_scan(self):
        try:
            self.load_scan()
        except IOError:
            self.generate_actionlist()
            self.runthread = ScanThread(self.runset,self.actionlist)
            self.runthread.start()

    def load_scan(self,doreload=False):
        checklist = _check_for_data_files([self.make_prefix(m) for m in self.movelist])
        if not all(checklist):
            if not any(checklist):
                raise IOError('Files not found, cannot load data.')
            else:
                raise Exception('Some files found, scan partially complete. \n \
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

    def insert_scan(self,movevalue):
        self.movefunc(movevalue)
        self.runset.insert(camcontrol.DataRun(run_prefix=self.make_prefix(movevalue),htime=self.duration,
            runparam={self.movetype:movevalue},**self.kwargs))

class AngleScan:

    def __init__(self, duration, anglerange, stepsize, prefix, **kwargs):
        self.duration = duration
        self.anglerange = anglerange
        self.stepsize = stepsize
        self.prefix = prefix
        self.runset = camcontrol.RunSet()
        self.kwargs = kwargs
        self.anglelist = []
        # self.prefixes = []
        for angle in np.arange(self.anglerange[0],self.anglerange[1]+self.stepsize,self.stepsize):
            self.anglelist.append(angle)
            # self.prefixes.append(self.prefix+'_%d'+ % angle)
        self.continuescan = False
        self.scanandaction = ScanAndAction(self.runset,self.prefix,'angle',self.anglelist,
            ardstep.go_to_degree,self.continuescan,self.duration,**kwargs)


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

    def load_scan(self,doreload = False):
        self.scanandaction.load_scan(doreload=doreload)

    def run_scan(self):
        self.scanandaction.run_scan()

class CameraScan:

    def __init__(self, duration, distancerange, stepsize, prefix, **kwargs):
        self.duration = duration
        self.distancerange = distancerange
        self.stepsize = stepsize
        self.prefix = prefix
        self.runset = camcontrol.RunSet()
        self.kwargs = kwargs
        self.distlist = []
        # self.prefixes = []
        for dist in np.arange(self.distancerange[0],self.distancerange[1]+self.stepsize,self.stepsize):
            self.distlist.append(dist)
            # self.prefixes.append(self.prefix+'_%d' % dist)
        self.continuescan = False
        self.scanandaction = ScanAndAction(self.runset,self.prefix,'mm_camera',self.distlist,
            ardstep.go_to_mm,self.continuescan,self.duration,**kwargs)


    # def angle_plot(self,start=0,end=-1,show=True,**kwargs):
    #     if self.runset.dataruns is None:
    #         print('Error: datafiles not loaded.')
    #     else:
    #         angles = [x.runparam['angle'] for x in self.runset.dataruns]
    #         counts = [x.counts_per_second(start=start,end=end) for x in self.runset.dataruns]
    #         camcontrol.plt.plot(angles,counts,**kwargs)
    #         camcontrol.plt.xlabel('Angle (deg)')
    #         camcontrol.plt.ylabel('Counts/sec in region '+str(start)+':'+str(end))
    #         if show:
    #             camcontrol.plt.show()

    def load_scan(self,doreload = False):
        self.scanandaction.load_scan(doreload=doreload)

    def run_scan(self):
        self.scanandaction.run_scan()


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
        self._wait_current_complete()
        print('Congratulations, scan complete! Have a nice day ;)')

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




