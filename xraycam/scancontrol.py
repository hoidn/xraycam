import threading, time
import numpy as np
from . import camcontrol
from arduinostepper import arduinostepper as ardstep
from .config import saveconfig
import os

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
                run_prefix=self.make_prefix(m), duration=self.duration,**self.kwargs)
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
            elif self.continuescan:
                raise IOError('Partial files found, continuescan set true, attempting to continue...')
            elif not self.continuescan:
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
                print('All files already loaded. Run load_scan with doreloafd=True to reload files.')

    def insert_scan(self,movevalue):
        self.movefunc(movevalue)
        self.runset.insert(camcontrol.DataRun(run_prefix=self.make_prefix(movevalue),duration=self.duration,
            runparam={self.movetype:movevalue},**self.kwargs))

class AngleScan:
    """Takes scans at a series of sample angle positions and stores the 
    dataruns in a runset object.

    Example:
        AngleScan('30s',[-15,15],5,'6.20.17.ZnS_ExampleAngleScan')
        Will run 30second scans at positions [-15,-10,-5,...,10,15]
        and store under runset prefix '6.20.17.ZnS_ExampleAngleScan'

    Args:
        duration (str): length of each scan as a string (e.g. '30s','2m')
        anglerange ([float,float]): range of angles to step through (e.g. [30,60])
        stepsize (float): step size between scans
        prefix (str): prefix of dataruns saved by the runset
    """

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
        else:#TODO: sort plot based on angles, in case run gets iserted with angle out of order
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

    def insert_scan(self,movevalue):
        self.anglelist.append(movevalue)
        self.scanandaction.insert_scan(movevalue)

    def plot_scans(self, **kwargs):
        [x.plot_lineout(show=False,**kwargs) for x in self.runset]
        camcontrol.plt.show()

    def check_scan_status(self):
        import humanfriendly
        currentscans = len(self.runset.dataruns)
        numscans = len(self.anglelist)
        timeleft = humanfriendly.parse_timespan(self.duration) - self.runset.dataruns[-1].acquisition_time()
        if currentscans < numscans:
            print("On scan ",str(currentscans)," of ",str(numscans),'.')
            print("{:.0f}".format(timeleft)," seconds left in current scan.")
        elif currentscans == numscans:
            if timeleft > 0:
                print("On scan ",str(currentscans)," of ",str(numscans),'.')
                print("{:.0f}".format(timeleft)," seconds left in current scan.")
            else:
                print("Scan complete!")

class CameraScan:
    """Takes scans at a series of camera  positions and stores the 
    dataruns in a runset object.

    Example:
        CameraScan('300s',[0.25,2.25],0.25,'6.20.17.ZnS_ExampleCameraScan')
        Will run 30second scans at positions [0.25,0.5,0.75,...,2,2.25]
        and store under runset prefix '6.20.17.ZnS_ExampleCameraScan'

    Args:
        duration (str): length of each scan as a string (e.g. '30s','2m')
        anglerange ([float,float]): range of camera positions to step through (e.g. [0,1.75])
        stepsize (float): step size between scans
        prefix (str): prefix of dataruns saved by the runset
    """

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

    def insert_scan(self,movevalue):
        self.distlist.append(movevalue)
        self.scanandaction.insert_scan(movevalue)

    def plot_scans(self, **kwargs):
        [x.plot_lineout(show=False,**kwargs) for x in self.runset]
        camcontrol.plt.show()

    def check_scan_status(self):
        import humanfriendly
        currentscans = len(self.runset.dataruns)
        numscans = len(self.distlist)
        timeleft = humanfriendly.parse_timespan(self.duration) - self.runset.dataruns[-1].acquisition_time()
        if currentscans < numscans:
            print("On scan ",str(currentscans)," of ",str(numscans),'.')
            print("{:.0f}".format(timeleft)," seconds left in current scan.")
        elif currentscans == numscans:
            if timeleft > 0:
                print("On scan ",str(currentscans)," of ",str(numscans),'.')
                print("{:.0f}".format(timeleft)," seconds left in current scan.")
            else:
                print("Scan complete!")

    def plot_fwhm_vs_pos(self,**kwargs):
        from xraycam.camalysis import fwhm_2d
        fwhmpos = np.array([[x.runparam['mm_camera'],fwhm_2d(x.get_lineout(**kwargs))] for x in self.runset])
        camcontrol.plt.plot(*fwhmpos.transpose(),label='FWHM vs Pos')
        camcontrol.plt.xlabel('mm_camera')
        camcontrol.plt.ylabel('FWHM (eV or bins)')
        camcontrol.plt.show()

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
                print('setting up scan (e.g. moving if motor scan)')
                action()
                dr = datarun()
                print('scan started')
                self.current = dr
                self.runset.dataruns.append(dr)#quick fix 8.16.17
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
            duration = None, dashinfilename=True, **kwargs):
        """
        prefix : str
            Datarun prefix prefix.
        number_runs : int
        duration : str
        """
        self.runset = runset
        if duration is None:
            raise ValueError("kwarg duration MUST be provided to instantiate RunSequence.")
        if prefix is None:
            raise ValueError("RunSequence must have a prefix to save files.")
        else:
            prefixes = [prefix + '_%d' % i for i in range(number_runs)]
        self.datarunfuncs = [lambda run_prefix=prefix: camcontrol.DataRun(run_prefix = run_prefix, duration = duration, **kwargs)
            for prefix in prefixes]
        self.actionlist = [[_do_nothing,dr] for dr in self.datarunfuncs]

    def run_scan(self):
        self.runthread = ScanThread(self.runset,self.actionlist)
        self.runthread.start()

    def stop(self):
        self.runthread.stopevent.set()

# class ActionQueue(threading.Thread):
#     """
#     Class that takes in a list of dictionaries describing actions,
#     and does them in order, waiting for the action to complete before
#     moving on to the next action.
#     """
#     def __init__(self,name='actionqueuethread'):
#         threading.Thread.__init__(self,name=name)
#         self.stopevent = threading.Event()
#         self.queue = []
#         self.currentindex = 0
#         self.runsetlist = []

#     def run(self):
#         while self.currentindex < len(self.queue):
#             if not self.stopevent.is_set():
#                 actiondict = self.queue[self.currentindex]
#                 if actiondict['action'] == 'capture':
#                     self.datarun_action(runset=actiondict['runset'],
#                         run_prefix=actiondict['run_prefix'],duration=actiondict['duration'],
#                         **actiondict['kwargs'])
#                 elif actiondict['action'] == 'move_sample':
#                     self.move_sample_action(actiondict['degree'])
#                 elif actiondict['action'] == 'move_camera':
#                     self.move_camera_action(actiondict['mm'])
#                 self.currentindex+=1


#     def datarun_action(self,runset,run_prefix,duration,**kwargs):
#         if runset is None:
#             runset = camcontrol.RunSet()
#             self.runsetlist.insert(runset)
#         print("Starting datarun: "+run_prefix)
#         dr = camcontrol.DataRun(run_prefix = run_prefix, duration = duration, **kwargs)
#         self.current = dr
#         runset.insert(dr)
#         self._wait_current_complete()

#     def move_sample_action(self,degree):
#         print("Moving sample to position: "+str(degree)+'deg')
#         ardstep.go_to_degree(degree)

#     def move_camera_action(self,mm):
#         print("Moving camera to position: "+str(mm)+'mm')
#         ardstep.go_to_mm(mm)

#     def insert_action(self,actiondict,index=None):
#         if index is None:
#             self.queue.append(actiondict)
#         else:
#             self.queue.insert(index,actiondict)

#     def _wait_current_complete(self):
#         """
#         Wait until the current run has completed.
#         """
#         import time
#         # Wait until current run is complete
#         if self.current is not None:
#             prev_time = self.current.acquisition_time()
#             while not self.stopevent.is_set():
#                 cur_time = self.current.acquisition_time()
#                 if cur_time != prev_time:
#                     prev_time = cur_time
#                     self.stopevent.wait(1)
#                 else:
#                     break 

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
        self.completed = []
        self.currentdatarun = None

    def run(self):
        while self.queue != []:
            if not self.stopevent.is_set()
            actionitem = self.queue.pop(0)
            self.parse_action(actionitem)
            self.completed.append(actionitem)

    def parse_action(self,actionitem):
        if actionitem['action'] == 'datarun':
            self.currentdatarun = camcontrol.DataRun(**actionitem['runkwargs'])
            rs = actionitem.get('runset',None)
            if rs is not None:
                rs.dataruns.append(self.currentdatarun)
            self.currentdatarun.zrun.block_until_complete()
        if actionitem['action'] == 'move_sample':
            ardstep.go_to_degree(actionitem['degree'])
        if actionitem['action'] == 'move_camera':
            ardstep.go_to_mm(actionitem['position'])

    def stop(self):
        self.stopevent.set()
        if self.currentdatarun is not None:
            self.currentdatarun.stop()

class MultipleRuns:

    def __init__(self, runset = None, prefix = None, number_runs = 0, rotate = False, 
        photon_value = 1, window_min = 0, window_max = 255, threshold = 0, 
        decluster = True, duration = None, loadonly = False):

    try:
        self.load()

    except FileNotFoundError:
        if runset is None:
            self.runset = camcontrol.RunSet()
        else:
            self.runset = runset
        self.prefix = prefix
        self.number_runs = number_runs
        self.runkwargs = dict(rotate =rotate, photon_value = photon_value, 
            window_min = window_min, window_max = window_max, threshold = threshold, 
            decluster = decluster, duration = duration, loadonly = loadonly)
        self.prefixlist = [self.prefix + '_{:d}'.format(i) for i in range(number_runs)]

    def load(self):
        def _exists(prefix):
            return prefix+'_array.npy' in os.listdir(xraycam.config.saveconfig['Directory'])

        if all([_exists(p) for p in self.prefixlist]):
            print('Loading all scans.')
            self.runset = camcontrol.RunSet()
            for p in self.prefixlist:
                self.runset.dataruns.append(camcontrol.DataRun(run_prefix = p))
        if any([_exists(p) for p in self.prefixlist]):
            print('Partially complete: {0} runs of {1} were found.'.format(
                sum([_exists(p) for p in self.prefixlist]),self.number_runs))
        else:
            raise FileNotFoundError

    def start(self):
        self.actionqueue = ActionQueue()
        for p in self.prefixlist:
            pkwargs = self.runkwargs.copy()
            pkwargs['run_prefix'] = p
            self.actionqueue.queue.append(
                {'action':'datarun',
                'runkwargs':pkwargs,
                'runset':self.runset
                })
        self.actionqueue.start()
