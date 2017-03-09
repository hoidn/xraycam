#There was an unfortunate error in the way data was collected at ALS. 
#This package is solely meant to allow importing of those datasets that were affected.

import xraycam
import numpy as np

from xraycam.camcontrol import _plot_lineout

class RunSequence:
    """
    Class that creates a sequence of DataRun instances, letting each data collection
    finish before beginning the next.
    """
    def __init__(self, prefix = None, number_runs = 0, total_runs=0, run_prefix = None,
            htime = None, dashinfilename=True, reverseimport=True,**kwargs):
        """
        prefix : str
            Datarun prefix prefix.
        number_runs : int
        htime : str
        """

        if reverseimport:
            importlist = np.arange(total_runs-number_runs,total_runs)
        else:
            importlist = range(number_runs)

        if htime is None:
            raise ValueError("kwarg htime MUST be provided to instantiate RunSequence.")
        if prefix is None:
            prefixes = [str(time.time()) for _ in importlist]
        else:
            if dashinfilename:
                prefixes = [prefix + '_%d' % i for i in importlist] #quick fix for filename problems..
            else:
                prefixes = [prefix + '%d' % i for i in importlist]
        self.funcalls = [lambda prefix=prefix: xraycam.camcontrol.DataRun(run_prefix = prefix, htime = htime, **kwargs)
            for prefix in prefixes]
        self.current = None
        self.reverseimport = reverseimport


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
            if self.reverseimport:
                run = self.funcalls.pop()()
            else:
                run = self.funcalls.pop(0)()
            self.current = run
        except IndexError:
            raise StopIteration
        return run


class RunSet:
    """
    Class containing a collection of DataRun instances.
    """
    def __init__(self, *args, **kwargs):
        """
        TODO docstring
        """
        self.dataruns = None

    def from_multiple_exposure(self,*args,**kwargs):
        from xraycam import async
        self.dataruns = async.IterThread(RunSequence(*args, **kwargs))

    def insert(self,  datarun = None, *args, **kwargs):
        """
        datarun : DataRun

        If datarun is provided, insert is into this RunSet. Otherwise pass
        args and kwargs to the constructor for DataRun and insert the resulting
        object into this RunSet.
        """
        if datarun is None:
            datarun = xraycam.camcontrol.DataRun(*args, **kwargs)
        if self.dataruns is None:
            self.dataruns = []
        self.dataruns.append(datarun)

    def get_dataruns(self):
        return self.dataruns

    def plot_lineouts(self, *args, show = True, **kwargs):
        def one_lineout(datarun):
            return datarun.plot_lineout(*args, show = False, **kwargs)
        lineouts = list(map(one_lineout, self.dataruns))
        if show:
            plt.show()
        return lineouts

    def filter_reduce_frames(self, **kwargs):
        """
        Do artifact filtering for frames of all component data runs and
        merge all the resulting Frames into a new Frame instance.

        kwargs are passed through to DataRun.filter
        """
        def make_frame(dr):
            old_frame = dr.get_frame()
            return Frame(old_frame.get_data(), name = self.name,
                time = old_frame.time, rotate = self.rotate)
        frames = list(map(make_frame, self.dataruns))
        def framefilter(frame):
            return frame.filter(**kwargs)
        return reduce(operator.add, list(map(framefilter, frames)))

    def get_total(self, normalize=False,energy=(None,None),**kwargs):
        # lineoutx = self.dataruns[0].get_frame().get_lineout(energy=(None,None),**kwargs)[0]
        lineouty = np.sum([x.get_frame().get_lineout(energy=(None,None),**kwargs)[1] for x in self.dataruns],axis=0)#hardcoded (none,none) to energy arg here
        # try:
        #     if not normalize[1]:
        #         lineouty = lineouty/np.sum(lineouty[normalize[1][0],normalize[1][1]])
        # except IndexError:
        try:
            if normalize == 'peak':
                lineouty = lineouty/max(lineouty)
            elif normalize[0] == 'integral':
                norm = np.sum(intensity[normalize[1][0]:normalize[1][1]])
            elif normalize == 'integral':
                lineouty = lineouty/np.sum(lineouty)
        except TypeError:
            pass
        lineoutx = np.arange(len(lineouty))
        lineout = np.array([lineoutx,lineouty])
        if energy != (None,None):
            from xraycam.camalysis import add_energy_scale
            lineout = add_energy_scale(lineouty,energy[0],known_bin=energy[1],rebinparam=kwargs.get('rebin',1),camerainvert=True,braggorder=1)
        return np.array(lineout)

    def plot_total_lineout(self, normalize=False,show=True,label=None,**kwargs):
        lineout = self.get_total(normalize=normalize,**kwargs)
        _plot_lineout(*lineout,show=show,label=label)
        return lineout