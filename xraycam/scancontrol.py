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

def _do_nothing():
    pass

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
            camcontrol.plt.plot(angles,counts,label='angle_scan',**kwargs)
            camcontrol.plt.xlabel('Angle (deg)')
            camcontrol.plt.ylabel('Counts/sec in region '+str(start)+':'+str(end))
            if show:
                camcontrol.plt.show()

    def load_scan(self,doreload = False):
        self.scanandaction.load_scan(doreload=doreload)

    def run_scan(self):
        self.scanandaction.run_scan()

    def stop(self):
        self.scanandaction.runthread.stopevent.set()

    def insert_scan(self):
        self.scanandaction.insert_scan()

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

    def load_scan(self,doreload = False):
        self.scanandaction.load_scan(doreload=doreload)

    def run_scan(self):
        self.scanandaction.run_scan()

    def stop(self):
        self.scanandaction.runthread.stopevent.set()

    def insert_scan(self):
        self.scanandaction.insert_scan()

class ScanThread(threading.Thread):
    """
    Class that takes in a runset and an actionlist of the form 
    [[action0,datarun0],[action1,datarun1],...],
    in which action and datarun are functions. For each element, the action is taken,
    and then the datarun is started, and inserted into the runset.
    """
    def __init__(self,runset,actionlist):
        threading.Thread.__init__(self)
        self.stopevent = threading.Event()
        self.runset = runset
        self.actionlist = actionlist
        self.current = None

    def run(self):
        for el in self.actionlist:
            if not self.stopevent.is_set():
                action, datarun = el
                print('moving before scan')
                action()
                dr = datarun()
                print('scan started')
                self.current = dr
                self.runset.insert(dr)
                self._wait_current_complete()
            else:
                self.current.stop()
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
            while not self.stopevent.is_set():
                cur_time = self.current.acquisition_time()
                if cur_time != prev_time:
                    prev_time = cur_time
                    self.stopevent.wait(1)
                else:
                    break    

class RunSequence:
    """
    Class that creates a sequence of DataRun instances, letting each data collection
    finish before beginning the next.
    """
    def __init__(self, runset, prefix = None, number_runs = 0,
            htime = None, dashinfilename=True, **kwargs):
        """
        prefix : str
            Datarun prefix prefix.
        number_runs : int
        htime : str
        """
        self.runset = runset
        if htime is None:
            raise ValueError("kwarg htime MUST be provided to instantiate RunSequence.")
        if prefix is None:
            raise ValueError("RunSequence must have a prefix to save files.")
        else:
            prefixes = [prefix + '_%d' % i for i in range(number_runs)]
        self.datarunfuncs = [lambda run_prefix=prefix: camcontrol.DataRun(run_prefix = run_prefix, htime = htime, **kwargs)
            for prefix in prefixes]
        self.actionlist = [[_do_nothing,dr] for dr in self.datarunfuncs]

    def run_scan(self):
        self.runthread = ScanThread(self.runset,self.actionlist)
        self.runthread.start()

    def stop(self):
        self.runthread.stopevent.set()

class ActionQueue(threading.Thread):
    """
    Class that takes in a list of dictionaries describing actions,
    and does them in order, waiting for the action to complete before
    moving on to the next action.
    """
    def __init__(self,name='actionqueuethread'):
        threading.Thread.__init__(self,name=name)
        self.stopevent = threading.Event()
        self.queue = []
        self.currentindex = 0
        self.runsetlist = []

    def run(self):
        while self.currentindex < len(self.queue):
            if not self.stopevent.is_set():
                actiondict = self.queue[self.currentindex]
                if actiondict['action'] == 'capture':
                    self.datarun_action(runset=actiondict['runset'],
                        run_prefix=actiondict['run_prefix'],htime=actiondict['htime'],
                        **actiondict['kwargs'])
                elif actiondict['action'] == 'move_sample':
                    self.move_sample_action(actiondict['degree'])
                elif actiondict['action'] == 'move_camera':
                    self.move_camera_action(actiondict['mm'])
                self.currentindex+=1


    def datarun_action(self,runset,run_prefix,htime,**kwargs):
        if runset is None:
            runset = camcontrol.RunSet()
            self.runsetlist.insert(runset)
        print("Starting datarun: "+run_prefix)
        dr = camcontrol.DataRun(run_prefix = run_prefix, htime = htime, **kwargs)
        self.current = dr
        runset.insert(dr)
        self._wait_current_complete()

    def move_sample_action(self,degree):
        print("Moving sample to position: "+str(degree)+'deg')
        ardstep.go_to_degree(degree)

    def move_camera_action(self,mm):
        print("Moving camera to position: "+str(mm)+'mm')
        ardstep.go_to_mm(mm)

    def insert_action(self,actiondict,index=None):
        if index is None:
            self.queue.append(actiondict)
        else:
            self.queue.insert(index,actiondict)

    def _wait_current_complete(self):
        """
        Wait until the current run has completed.
        """
        import time
        # Wait until current run is complete
        if self.current is not None:
            prev_time = self.current.acquisition_time()
            while not self.stopevent.is_set():
                cur_time = self.current.acquisition_time()
                if cur_time != prev_time:
                    prev_time = cur_time
                    self.stopevent.wait(1)
                else:
                    break 



