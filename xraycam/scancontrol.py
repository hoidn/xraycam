import threading
import time
import datetime
import re
import numpy as np
from . import camcontrol
from . import config
import os
from arduinostepper import arduinostepper as ardstep
import SpellmanUSB
import xraycam

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

samplemap = {
    'example':{
        'angle':132,
        'runset':None,
        'runsetindex':0,
        'duration':180
        }
}

def check_samplemap_status():
    done = []
    for sample in samplemap:
        try:
            done.append(samplemap[sample]['runset'].dataruns[-1].zrun._finished)
        except (IndexError, AttributeError) as e:
            pass
                     
    if all(done):
        print('Stalled or Finished')
    
    for sample in samplemap:
        try:
            dr = samplemap[sample]['runset'].dataruns[-1]
            print(dr.acquisition_time(),'/',dr.duration,', ',dr.name)
        except (IndexError, AttributeError) as e:
            pass

def _get_next_index(rootprefix):
    matches=[]
    rootre = re.compile(rootprefix+r'(\d*)_array')
    for f in os.listdir(config.saveconfig['Directory']):
        m = rootre.search(f)
        if m:
            matches.append(m.group(1))
    if matches == []:
        return 0
    else:
        return  max([int(s) for s in matches]) + 1

class ActionQueue(threading.Thread):
    """
    Class that takes in a list of dictionaries describing actions,
    and does them in order, waiting for the action to complete before
    moving on to the next action.
    """
    def __init__(self,name='actionqueuethread',continuous = False):
        threading.Thread.__init__(self,name=name)
        self.stopevent = threading.Event()
        self.queue = []
        self.completed = []
        self.current = None
        self._finished = False
        self.continuous = continuous


    def run(self):
        while True:
            while self.queue != []:
                if not self.stopevent.is_set():
                    actionitem = self.queue.pop(0)
                    self.parse_action(actionitem)
                    self.completed.append(actionitem)
                else:
                    print('Stop command received, stopping queue.')
                    self._finished = True
                    break
            if not self.continuous or self.stopevent.is_set():
                break
            else:
                time.sleep(1)
        self._finished = True
        print('ActionQueue completed.')

    def parse_action(self,actionitem):
        if actionitem['action'] == 'datarun':
            if actionitem['runkwargs']['run_prefix'] == 'next':
                root = actionitem['rootprefix']
                actionitem['runkwargs']['run_prefix'] = root + str(_get_next_index(root))
            self.current = camcontrol.DataRun(**actionitem['runkwargs'])
            rs = actionitem.get('runset',None)
            if rs is not None:
                rs.dataruns.append(self.current)
            self.current.zrun.block_until_complete()

        elif actionitem['action'] == 'move_sample':
            ardstep.go_to_degree(actionitem['degree'])

        elif actionitem['action'] == 'move_camera':
            ardstep.go_to_mm(actionitem['position'])

        elif actionitem['action'] == 'multiple_runs':
            self.current = MultipleRuns(
                prefix = actionitem['prefix'],
                number_runs = actionitem['number_runs'],
                duration = actionitem['duration'],
                **actionitem['runkwargs']
                )
            assert not self.current._loaded, \
            'MultipleRuns scan prefixes already exist. Use a new prefix.'

            self.current.start()
            self.current.block_until_complete()

        elif actionitem['action'] == 'disengage_high_voltage':
            SpellmanUSB.disengage_high_voltage()

        elif actionitem['action'] == 'engage_high_voltage':
            SpellmanUSB.engage_high_voltage()

        elif actionitem['action'] == 'spellman_clear_setpoints':
            SpellmanUSB.clear_setpoints()

        elif actionitem['action'] == 'xraycam_shutdown':
            xraycam.shutdown()

    def stop(self):
        self.stopevent.set()
        if self.current is not None:
            self.current.stop()

    def add_sample_scan(self, sample, totalcounts = None, index = None):
        
        #configure run_prefix
        rootprefix = '{0}.{1}runset{2}_'.format(
            datetime.date.today().strftime('%m.%d.%y'),
            sample,
            samplemap[sample]['runsetindex']
        )
        
        #configure duration based on count rate
        if totalcounts is not None:
            dur = int(totalcounts / samplemap[sample]['countspersec'])
        else:
            dur = samplemap[sample]['duration']
        
        #initialize runset if necessary
        if samplemap[sample]['runset'] is None:
            rs = camcontrol.RunSet()
            samplemap[sample]['runset'] = rs
            rs.load_runs(rootprefix[:-1])


        #insert into proper position
        if index is None:
            index = len(self.queue)
        self.queue.insert(index,{'action':'move_sample','degree':samplemap[sample]['angle']})
        self.queue.insert(index + 1,{
                'action':'datarun',
                'runkwargs':{
                    'run_prefix':'next',
                    'duration':dur},
                'runset':samplemap[sample]['runset'],
                'rootprefix':rootprefix
            })

    def time_left_in_queue(self):
        durations=[]
        for a in self.queue:
            if a['action'] == 'datarun':
                durations.append(a['runkwargs']['duration'])
        print('{} sec in queue'.format(np.sum(durations)))
        return np.sum(durations)

class MultipleRuns:
    """Class that takes multiple DataRuns with the same run parameters. Useful
    when taking long exposures to monitor for possible changes in signal, or to
    be able to recover some useful data in case the instrument crashes.
    """
    def __init__(self, prefix, number_runs, duration, runset = None, rotate = False, 
        photon_value = 1, window_min = 0, window_max = 255, threshold = 0, 
        decluster = True, loadonly = False):

        self.prefix = prefix
        self.prefixlist = [self.prefix + '_{:d}'.format(i) for i in range(number_runs)]
        self.number_runs = number_runs
        self._loaded = False

        try:
            self.load()
            self._loaded = True

        except FileNotFoundError:
            if runset is None:
                self.runset = camcontrol.RunSet()
            else:
                self.runset = runset
            self.runkwargs = dict(rotate =rotate, photon_value = photon_value, 
                window_min = window_min, window_max = window_max, threshold = threshold, 
                decluster = decluster, duration = duration, loadonly = loadonly)

    def load(self):
        def _exists(prefix):
            return prefix+'_array.npy' in os.listdir(config.saveconfig['Directory'])

        if all([_exists(p) for p in self.prefixlist]):
            print('Loading all scans.')
            self.runset = camcontrol.RunSet()
            for p in self.prefixlist:
                self.runset.dataruns.append(camcontrol.DataRun(run_prefix = p,
                                                            loadonly = True))

        elif any([_exists(p) for p in self.prefixlist]):
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

    def stop(self):
        return self.actionqueue.stop()

    def block_until_complete(self, waitperiod = 0.1):
        while not self.actionqueue._finished:
            time.sleep(waitperiod)

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

    def __init__(self, prefix, duration, anglerange, stepsize, **kwargs):
        self.duration = duration
        self.anglerange = anglerange
        self.stepsize = stepsize
        self.prefix = prefix
        self.runset = camcontrol.RunSet()
        self.kwargs = kwargs
        self.anglelist = np.arange(self.anglerange[0],self.anglerange[1]+self.stepsize,self.stepsize)
        self.prefixlist = [self.prefix + '_{:.2f}deg'.format(a) for a in self.anglelist]

    def make_queue_entry(self, angle, prefix):
        self.actionqueue.queue.append({
            'action':'move_sample',
            'degree':angle
            })
        pkwargs = self.kwargs.copy()
        pkwargs['run_prefix'] = prefix
        pkwargs['duration'] = self.duration
        pkwargs['angle'] = angle.item()#.item is to convert numpy float to float for JSON serializability
        self.actionqueue.queue.append({
            'action':'datarun',
            'runkwargs':pkwargs,
            'runset':self.runset
            })

    def load(self):
        #TODO: make this load a common function for all multiple-run-type classes
        def _exists(prefix):
            return prefix+'_array.npy' in os.listdir(config.saveconfig['Directory'])

        if all([_exists(p) for p in self.prefixlist]):
            print('Loading all scans.')
            self.runset = camcontrol.RunSet()
            for p in self.prefixlist:
                self.runset.dataruns.append(camcontrol.DataRun(run_prefix = p))
        elif any([_exists(p) for p in self.prefixlist]):
            print('Partially complete: {0} runs of {1} were found.'.format(
                sum([_exists(p) for p in self.prefixlist]),len(self.anglelist)))
        else:
            raise FileNotFoundError

    def run_scan(self):
        try:
            self.load()
        except FileNotFoundError:
            self.actionqueue = ActionQueue()
            for a, p in zip(self.anglelist, self.prefixlist):
                self.make_queue_entry(a,p)
            self.actionqueue.start()

    def stop(self):
        self.actionqueue.stop()

    def insert_scan(self,movevalue):
        self.anglelist.append(movevalue)
        p = self.prefix + '_{:.2f}deg'.format(movevalue)
        self.prefixlist.append(p)
        assert self.actionqueue is None or self.actionqueue._finished == True, \
        'Error, self.actionqueue is not empty or finished. Must not be in the\
         middle of a scan to insert another value.'
        self.actionqueue = ActionQueue()
        self.make_queue_entry(movevalue,p)
        self.actionqueue.start()

    def plot_scans(self, **kwargs):
        [x.plot_lineout(show=False,**kwargs) for x in self.runset]
        camcontrol.plt.show()

    def angle_plot(self,start=0,end=-1,show=True,**kwargs):
        if self.runset.dataruns is None:
            print('Error: datafiles not loaded.')
        else:#TODO: sort plot based on angles, in case run gets iserted with angle out of order
            angles = [x.zrun.angle for x in self.runset.dataruns]
            counts = [x.counts_per_second(start=start,end=end) for x in self.runset.dataruns]
            sortindices = np.argsort(angles)
            angles = np.array(angles)[sortindices]
            counts = np.array(counts)[sortindices]
            camcontrol.plt.plot(angles,counts,label='angle_scan',**kwargs)
            camcontrol.plt.xlabel('Angle (deg)')
            camcontrol.plt.ylabel('Counts/sec in region '+str(start)+':'+str(end))
            if show:
                camcontrol.plt.show()

    def check_scan_status(self):
        # import humanfriendly
        currentscans = len(self.runset.dataruns)
        numscans = len(self.anglelist)
        timeleft = self.duration - self.runset.dataruns[-1].acquisition_time()
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
        self.distlist = np.arange(self.distancerange[0],self.distancerange[1]+self.stepsize,self.stepsize)
        self.prefixlist = [self.prefix + '_{:.2f}mm'.format(d) for d in self.distlist]

    def make_queue_entry(self, dist, prefix):
        self.actionqueue.queue.append({
            'action':'move_camera',
            'position':dist
            })
        pkwargs = self.kwargs.copy()
        pkwargs['run_prefix'] = prefix
        pkwargs['duration'] = self.duration
        pkwargs['mm_camera'] = dist.item() #.item is to convert numpy float to float for JSON serializability
        self.actionqueue.queue.append({
            'action':'datarun',
            'runkwargs':pkwargs,
            'runset':self.runset
            })

    def load(self):
        #TODO: make this load a common function for all multiple-run-type classes
        def _exists(prefix):
            return prefix+'_array.npy' in os.listdir(config.saveconfig['Directory'])

        if all([_exists(p) for p in self.prefixlist]):
            print('Loading all scans.')
            self.runset = camcontrol.RunSet()
            for p in self.prefixlist:
                self.runset.dataruns.append(camcontrol.DataRun(run_prefix = p))
        elif any([_exists(p) for p in self.prefixlist]):
            print('Partially complete: {0} runs of {1} were found.'.format(
                sum([_exists(p) for p in self.prefixlist]),len(self.distlist)))
        else:
            raise FileNotFoundError

    def run_scan(self):
        try:
            self.load()
        except FileNotFoundError:
            self.actionqueue = ActionQueue()
            for d, p in zip(self.distlist, self.prefixlist):
                self.make_queue_entry(d,p)
            self.actionqueue.start()

    def stop(self):
        self.actionqueue.stop()

    def insert_scan(self,movevalue):
        self.distlist.append(movevalue)
        p = self.prefix + '_{:.2f}deg'.format(movevalue)
        self.prefixlist.append(p)
        assert self.actionqueue is None or self.actionqueue._finished == True, \
        'Error, self.actionqueue is not empty or finished. Must not be in the\
         middle of a scan to insert another value.'
        self.actionqueue = ActionQueue()
        self.make_queue_entry(movevalue,p)
        self.actionqueue.start()

    def plot_scans(self, **kwargs):
        [x.plot_lineout(show=False,**kwargs) for x in self.runset]
        camcontrol.plt.show()

    def check_scan_status(self):
        # import humanfriendly
        currentscans = len(self.runset.dataruns)
        numscans = len(self.distlist)
        timeleft = self.duration - self.runset.dataruns[-1].acquisition_time()
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
        from xraycam.camalysis import fwhm_lineout
        fwhmpos = np.array([[x.zrun.mm_camera,fwhm_lineout(x.get_lineout(**kwargs))] for x in self.runset])
        camcontrol.plt.plot(*fwhmpos.transpose(),label='FWHM vs Pos')
        camcontrol.plt.xlabel('mm_camera')
        camcontrol.plt.ylabel('FWHM (eV or bins)')
        camcontrol.plt.show()
