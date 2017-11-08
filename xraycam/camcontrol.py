import numpy as np
import pkg_resources
import copy
import pdb
import operator
import logging
import os
import time
import dill

from functools import reduce

from . import utils
from . import config
from . import zwo
from xraycam.mpl_plotly import plt

logging.basicConfig(filename='xraycam.log', level=logging.DEBUG)

# from https://gist.github.com/rossdylan/3287138
# TODO: how about this?:
# def compose(*funcs): return lambda x: reduce(lambda v, f: f(v), reversed(funcs), x)
from functools import partial
def _composed(f, g, *args, **kwargs):
    return f(g(*args, **kwargs))

def compose(*a):
    try:
        return partial(_composed, a[0], compose(*a[1:]))
    except:
        return a[0]

# from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_substring
def _longest_common_substring(s1, *rest):
    if not rest:
        return s1
    else:
        s2, newrest = rest[0], rest[1:]
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:   
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    merged = s1[x_longest - longest: x_longest]
    return _longest_common_substring(merged, *newrest)


@utils.conserve_type
def _rebin_spectrum(x, y, rebin = 5):
    """
    Rebin `x` and `y` into arrays of length `int(len(x)/rebin)`. The
    highest-x bin is dropped in case len(x) isn't a multiple of rebin.

    x is assumed to be evenly-spaced and in ascending order.

    Returns: x, y
    """
    def group(arr1d, op = np.mean):
        """
        op: a function to evaluate on each new bin that returns a numeric value.
        >>> rebin = 3
        >>> group(range(10))
        [1.0, 4.0, 7.0]
        """
        import itertools
        i = itertools.count()
        def key(dummy):
            xindx = i.__next__()
            return int(xindx/rebin)
        return [op(list(values)) for groupnum, values in itertools.groupby(arr1d, key = key)][:-1]#is this -1 redundant? this shortens array from 1096->1095
    return group(x, np.mean), group(y, np.sum)

def _get_poisson_uncertainties(intensities):
    """
    Return the array of Poisson standard deviations, based on photon
    counting statistics alone, for an array containing summed ADC
    values.
    """
    return np.sqrt(np.array(intensities))

def _plot_lineout(pixeli, intensity, show = False, label = '', error_bars = False, normalize=False):
    if normalize=='peak':
        norm = np.max(intensity)
    elif normalize == 'integral':
        norm = np.sum(intensity)
    else:
        norm = 1.
    if error_bars:
        bars = _get_poisson_uncertainties(intensity) / norm
        error_y = dict(
            type = 'data',
            array = bars,
            visible = True
        )
        plt.plot(pixeli, intensity/norm, label = label, error_y = error_y)
    else:
        plt.plot(pixeli, intensity/norm, label = label)
    if show:
        plt.show()
    return intensity/norm

def _plot_histogram(values,xmin=1,xmax=255,show=True,binsize=1.000001,calib=[None,None],label='histogram'):
    xvalues = copy.deepcopy(values)
    xvalues[xvalues < xmin] = 0
    xvalues[xvalues > xmax] = 0
    data = xvalues[xvalues>0]
    slope = 1
    if calib[0] is not None:
        if calib[1] is None:
            calib[1] = [0,0]
        slope = np.abs((calib[1][1]-calib[0][1])/(calib[1][0]-calib[0][0]))
        data = data*slope
    plt.hist(data,xbins=dict(start=xmin*slope,end=xmax*slope,size=binsize*slope),label=label)
    if show:
        plt.show()

def _load_detector_settings(d):
    defaults = dict(window_min = 0, window_max = 255, threshold = 0)
    if all([d[k]==defaults[k] for k in defaults.keys()]):
        if config.sensorsettings is not {}:
            print('Loading sensor settings: ',config.sensorsettings)
            for k,v in config.sensorsettings.items():
                d[k] = v


class DataRun:
    def __init__(self, run_prefix = '', rotate = False, photon_value = 1, 
        window_min = 0, window_max = 255, threshold = 0, decluster = True, 
        duration = None, loadonly = False, saveonstop = True, **kwargs):

        #which parameters to search for in kwargs that will get saved in
        #the 'datarun_parameters' file.
        paramsavefromkwargs = ('angle','mm_camera')

        self.name = run_prefix
        self.rotate = rotate
        self.window_min = window_min
        self.window_max = window_max
        self.threshold  = threshold
        self.decluster = decluster
        self.duration = duration
        self.loadonly = loadonly
        self.saveonstop = saveonstop
        
        paramsave = {}
        for k in paramsavefromkwargs:
            if k in kwargs:
                paramsave[k] = kwargs[k]

        _load_detector_settings(self.__dict__)

        try:
            self.photon_value = config.datasettings['photon_value']
            print('Using photon_value from config: ',self.photon_value)
        except KeyError:
            self.photon_value = photon_value
            print('Using photon_value from function call: ',self.photon_value)

        self.zrun = zwo.ZRun(run_prefix = self.name, window_min = self.window_min, 
            window_max = self.window_max, threshold = self.threshold, decluster = self.decluster, 
            duration = self.duration, loadonly = self.loadonly, saveonstop = self.saveonstop,
            photon_value = self.photon_value,**paramsave)
        if self.zrun._finished:
            for k, v in self.zrun.__dict__.items():
                self.__dict__[k] = v
        

    def acquisition_time(self):
        return self.zrun.acquisition_time()

    def get_array(self):
        return self.zrun.get_array()

    def get_histograms(self):
        return self.zrun.get_histograms()

    def stop(self):
        """
        Stop the acquisition.
        """
        self.zrun.stop()

    def get_frame(self):
        """
        Get the frame object with the current data of the DataRun.  Stores in temporary _frame variable.
        """
        time = self.acquisition_time()
        self._frame = Frame(array = self.get_array(), name = self.name,
            photon_value = self.photon_value, time = time, rotate= self.rotate)
        return self._frame

    def plot_histogram(self, **kwargs):
        """
        Plot histogram output from the camera acquisition.
        """
        return self.get_frame().plot_histogram(**kwargs)

    def get_lineout(self,**kwargs):
        """
        kwargs are passed to self.frame.get_lineout
        """
        return self.get_frame().get_lineout(**kwargs)

    def plot_lineout(self, **kwargs):
        """
        kwargs are passed to self.frame.plot_lineout
        """
        return self.get_frame().plot_lineout(**kwargs)

    def show(self, **kwargs):
        self.get_frame().show(**kwargs)

    def counts_per_second(self, **kwargs):
        return self.get_frame().counts_per_second(**kwargs)


class RunSet:
    """
    Class containing a collection of DataRun instances.
    """
    def __init__(self, *args, **kwargs):
        """
        TODO docstring
        """
        self.dataruns = []

    def __iter__(self):
        return self.dataruns.__iter__()

    def plot_lineouts(self, *args, show = True, **kwargs):
        def one_lineout(datarun):
            return datarun.plot_lineout(*args, show = False, **kwargs)
        lineouts = list(map(one_lineout, self.dataruns))
        if show:
            plt.show()
        return lineouts

    def get_total_frame(self, runindices = ...):
        import operator
        from functools import reduce
        return reduce(operator.add,[x.get_frame() for x in np.array(self.dataruns)[runindices]])

    def load_runs(self, prefix, specify = None):
        if type(specify) != np.ndarray:
            if specify == None:
                istart = 0
            elif type(specify) == int:
                istart = specify
            i = istart
            while True:
                try:
                    self.dataruns.append(DataRun(run_prefix='{}_{}'.format(prefix,i),loadonly=True))
                    i += 1
                except FileNotFoundError:
                    break
                    
            print('Runs {0} through {1} were loaded.'.format(istart,i-1))
        else:
            for i in specify:
                    self.dataruns.append(DataRun(run_prefix='{}_{}'.format(prefix,i),loadonly=True))    


class Frame:
    def __init__(self, array = None, time = 0., name = '', photon_value = 1.,
            rotate = False):
        if rotate:
            self.data = array
        else:
            self.data = np.rot90(array)
        self.time = time
        self.name = name
        self.photon_value = photon_value

    def show(self,width=10, vmax=None, **kwargs):
        """Show the frame. Kwargs are passed through to plt.imshow."""
        import matplotlib.pyplot as mplt
        from matplotlib.colors import LogNorm
        countdata = self.data/self.photon_value
        if vmax is None:
            vmax = np.max(countdata)*0.7
        fig, ax = mplt.subplots(figsize=(width,1936/1096*width))
        cax = ax.imshow(countdata,vmax=vmax,interpolation='none',norm=LogNorm(vmin=1, vmax=vmax))
        cbar = fig.colorbar(cax, 
            ticks=[int(x) for x in np.logspace(0.01,np.log10(vmax),num=8)],#np.insert(np.arange(0,int(vmax),vmax/10),0,1),
            format='$%d$',fraction=0.05, pad=0.04)
        mplt.show()

    def get_data(self, **kwargs):
        return self.data

    def __add__(self, other):
        #TODO, check if frames have same photon value
        if self.name and other.name:
            name =  _longest_common_substring(self.name, other.name)
        else:
            name = ''
        new = Frame(array = self.data, name = name, rotate=True, photon_value=self.photon_value)
        new.data = new.data + other.data
        new.time = self.time + other.time
        return new

    def remove_hot(self, darkrun = None, threshold = 0):
        """
        darkrun : DataRun
            A dark run to use as the hot pixel mask
        Returns a new Frame with hot pixels removed.
        """
        from . import camalysis
        new = copy.deepcopy(self)
        hot_indices = camalysis.get_hot_pixels(darkrun = darkrun, threshold = threshold)
        new.data[hot_indices] = 0
        return new

#note to self 3/6/17: changed the xrange and yrange below to None so that we don't lose the edge of the frame
    def _raw_lineout(self, xrange=(None,None), yrange=(None,None),**kwargs):
        return np.sum(self.data[yrange[0]:yrange[1],xrange[0]:xrange[1]], axis = 0) / self.photon_value

    def get_lineout(self, energy=(None,None) , rebin = 1, smooth = 0, **kwargs):
        """
        Return a smoothed and rebinned lineout of self.data.

        smooth : number of pixel columns by which to smooth rebin :
        number of pixel columns per bin

        Returns: bin values, intensities

        Optionally add energy scale with tuple energy=(known energy, known bin).
        If known_bin is None, max value of lineout is set to known_energy.
        Returns: energies, intensities
        """
        def apply_smooth(arr1d):
            from scipy.ndimage.filters import gaussian_filter as gf
            return gf(arr1d, smooth)
        def apply_rebin(arr1d):
            return _rebin_spectrum(np.array(range(len(arr1d))), arr1d, rebin = rebin)

        if (not isinstance(rebin, int)) or rebin < 1:
            raise ValueError("Rebin must be a positive integer")

        lineout =  compose(apply_rebin, apply_smooth)(self._raw_lineout(**kwargs))

        #Add energy scale to lineout
        if energy != (None,None):
            from xraycam.camalysis import add_energy_scale
            lineout_x, lineout_y = lineout
            lineout = add_energy_scale(lineout_y,energy[0],known_bin=energy[1],rebinparam=rebin,camerainvert=True,braggorder=1)

        return np.array(lineout)

    def plot_lineout(self, smooth = 0, error_bars = False, rebin = 1, label = '',
            show = True, normalize=False, **kwargs):
        if not label:
            label = self.name
        return _plot_lineout(*self.get_lineout(rebin = rebin, smooth = smooth,**kwargs),
            show = show, error_bars = error_bars, label = label, normalize=normalize)

    def plot_histogram(self, show = True, binsize=1.000001, xmin=5, xmax=255, calibrate = [None,None], **kwargs):
        """
        Plot and return a histogram of pixel values.

        kwargs are passed to plt.plot.
        """
        _plot_histogram(self.data.flatten(),xmin=xmin,xmax=xmax,binsize=binsize,calib=calibrate,show=show,**kwargs)

    def counts_per_second(self, elapsed = None, start = None, end = None, yrange = (None,None)):
        """
        Return average counts per second the exposures that constitute this Frame.
        """
        if elapsed is None:
            elapsed = self.time
        return np.sum(self._raw_lineout(yrange = yrange)[start:end]) / self.time

    def save_to_disk(self, filename, directory = 'cache'):
        with open(directory + '/' + filename + '.frame', 'wb') as f:
            dill.dump(self,f)

def load_Frame(filename, directory = 'cache'):
    with open(directory + '/' + filename + '.frame', 'rb') as f:
        frame = dill.load(f)
    return frame

def save_lineout_csv(datarun,filename,tosharedfolder=False,**kwargs):
    try:
        lineoutx, lineouty = datarun.get_frame().get_lineout(**kwargs)
        acquisitiontime = datarun.acquisition_time()
        countrate = datarun.counts_per_second()
    except AttributeError:
        lineoutx, lineouty = datarun
        acquisitiontime = 'unspecified'
        countrate = 'unspecified'
    savedata = np.array([lineoutx,lineouty])
    headerstr = 'Plot options: '+str(kwargs)+'\nCount rate: '+str(countrate)+'\nAcquisition time: '+str(acquisitiontime)+'\nEnergies(eV),Intensities'
    if tosharedfolder:
        filename = '/media/sf_VBoxShare/'+filename
    np.savetxt(filename,savedata,delimiter=',',header=headerstr)
    print('file saved as: ',filename)

# Below are a couple of scripts for importing legacy data, in which the data was saved as: [_initial_array,_final_array,_run_parameters,_time_start,_total_time]
def import_legacy(prefix,directory='/home/xrays/xraycam/examples/cache/',attributes=None,photon_value=None):
    if attributes is None:
        attributes = ['_time_start','_total_time','_run_parameters']
    params = {}
    for a in attributes:
        with open(directory+prefix+a,'rb') as f:
            params[a]=(dill.load(f))
    arr = np.load(directory+prefix+'_final_array')

    if '_run_parameters' in params:
        frame = Frame(arr,time=params['_total_time'],photon_value=params['_run_parameters']['photon_value'])
    else:
        frame = Frame(arr,time=params['_total_time'],photon_value=photon_value)
    frame.runparams = params
    return frame

def import_runset_legacy(prefix,runnumlist,directory='/home/xrays/xraycam/examples/cache/', dosum=True, attributes=None,photon_value=None):
    prefixes = [prefix+str(i) for i in runnumlist]
    
    if dosum:
        firstprefix = prefixes.pop(0)
        frametot = import_legacy(firstprefix,directory=directory,attributes=attributes,photon_value=photon_value)
        for p in prefixes:
            frametot += import_legacy(p,directory=directory,attributes=attributes,photon_value=photon_value)
        frame = frametot
    else:
        frames = []
        for p in prefixes:
            frames.append(import_legacy(p,directory=directory,attributes=attributes,photon_value=photon_value))
        frame = frames
    return frame