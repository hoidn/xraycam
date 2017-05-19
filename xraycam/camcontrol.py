import numpy as np
import pkg_resources
import copy
import pdb
import operator
import logging
import os
import time

from functools import reduce

from . import utils
from . import config
from . import zwo
from . import detconfig_zwo

if config.plotting_mode != 'minigui':
    if config.plotting_mode == 'notebook':
        from xraycam.mpl_plotly import plt
        #import sys
        #sys.stdout = open(config.logfile_path, 'w')
    else:
        import matplotlib.pyplot as plt

logging.basicConfig(filename='xraycam.log', level=logging.DEBUG)

# TODO: set up keypair authentication at first usage of the package, if necessary,
# or else simply switch to using password authentication throughout. 

PKG_NAME = __name__.split('.')[0]

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

def resource_f(fpath):
    from io import StringIO
    return StringIO(pkg_resources.resource_string(PKG_NAME, fpath))

def resource_path(fpath):
    return pkg_resources.resource_filename(PKG_NAME, fpath)

def adc_to_eV(adc_values):
    """Generate an energy scale"""
    calib_slope, calib_intercept = calib_params()
    return adc_values * calib_slope + calib_intercept

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

def _plot_lineout(pixeli, intensity, show = False, label = '', error_bars = False,
        peaknormalize = False, normalize=False):
    #keep peaknormalize for now for backwards compabitility
    # try:
    #     if not normalize[1]:
    #         lineouty = lineouty/np.sum(lineouty[normalize[1][0],normalize[1][1]])
    # except IndexError:
    try:
        if peaknormalize==True or normalize=='peak':
            norm = np.max(intensity)
        elif normalize[0] == 'integral':
            norm = np.sum(intensity[normalize[1][0]:normalize[1][1]])
        elif normalize == 'integral':
            norm = np.sum(intensity)
        else:
            norm = 1.
    except TypeError:
        norm = 1.
    if error_bars:
        if not (config.plotting_mode == 'notebook'):
            raise NotImplementedError("Error bars not supported in matplotlib mode")
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

def _plot_histogram(values,xmin=5,xmax=255,show=True,binsize=1.000001,calib=[None,None],label='histogram'):
    values[values < xmin] = 0
    values[values > xmax] = 0
    slope = 1
    if calib[0] is not None:
        if calib[1] is None:
            calib[1] = [0,0]
        slope = np.abs((calib[1][1]-calib[0][1])/(calib[1][0]-calib[0][0]))
        values = values*slope
    plt.hist(values,xbins=dict(start=xmin*slope,end=xmax*slope,size=binsize*slope),label=label)
    if show:
        plt.show()

def _load_detector_settings(kwargs):
    for k,v in detconfig_zwo.sensorsettings.items():
        if k not in kwargs:
            kwargs[k] = v


class DataRun:
    def __init__(self, run_prefix = '', rotate = False, runparam = {}, norunparam = False, photon_value = 1, *args, **kwargs):
        self.rotate = rotate
        _load_detector_settings(kwargs)
        try:
            self.photon_value = detconfig_zwo.datasettings['photon_value']
        except KeyError:
            self.photon_value = photon_value

        self.runparam = runparam
        for k in ('threshold','htime','window_min','window_max'):
            if k in kwargs:
                runparam[k]=kwargs[k]
        runparam['photon_value']=self.photon_value
        from . import zwo
        self.base = zwo.ZRun(run_prefix = run_prefix, runparam = self.runparam, norunparam = norunparam, *args, **kwargs)
        self.name = run_prefix      
        if not norunparam:
            self.runparam = self.base._run_parameters #override previous runparam with the actual values from ZRun (i.e. the cached files if loaded from cache)

    def acquisition_time(self):
        return self.base.acquisition_time()
    def get_array(self):
        return self.base.get_array()
    def get_histograms(self):
        return self.base.get_histograms()
    def stop(self):
        """
        Stop the acquisition.
        """
        self.base.stop()

    def get_frame(self):
        # TODO: do we need this attribute?
        time = self.acquisition_time()
        self._frame = Frame(array = self.get_array(), name = self.name,
            photon_value = self.photon_value, time = time, rotate= self.rotate)
        return self._frame

    @staticmethod
    def time_to_numexposures(timestring):
        """
        Convert a human-readable time string (e.g. '3m', '1h', etc.) to a number of exposures.
        """
        import humanfriendly
        def roundup(x):
            return int(np.ceil(x / 10.0)) * 10
        return roundup(config.frames_per_second * humanfriendly.parse_timespan(timestring))

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
        kwargs are passed to self.frame.filter.
        """
        return self.get_frame().plot_lineout(**kwargs)

    def show(self, **kwargs):
        self.get_frame().show(**kwargs)

    def counts_per_second(self, **kwargs):
        return self.get_frame().counts_per_second(elapsed = self.acquisition_time(), **kwargs)


class RunSet:
    """
    Class containing a collection of DataRun instances.
    """
    def __init__(self, *args, **kwargs):
        """
        TODO docstring
        """
        self.dataruns = None
        self.totallineout = np.array([])
        self.totallineoutdict = {}
        self.totaldata = None

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
            datarun = DataRun(*args, **kwargs)
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
        fdict={}
        fdict['normalize']=normalize
        fdict['energy']=energy
        for key,val in kwargs.items():
            fdict[key]=val

        if not self.totallineout.any() or fdict != self.totallineoutdict:
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
                lineout = np.array(add_energy_scale(lineouty,energy[0],known_bin=energy[1],rebinparam=kwargs.get('rebin',1),camerainvert=True,braggorder=1))
            self.totallineout = lineout
            self.totallineoutdict = fdict

        else:
            lineout = self.totallineout

        return np.array(lineout)

    def plot_total_lineout(self, normalize=False,show=True,label=None,**kwargs):
        lineout = self.get_total(normalize=normalize,**kwargs)
        _plot_lineout(*lineout,show=show,label=label)
        return lineout

    def get_total_frame(self):
        return np.sum([x.get_frame().data for x in self.dataruns],axis=0)


class Frame:
    def __init__(self, array = None, time = 0., name = '', photon_value = 45.,
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
        if self.name and other.name:
            name =  _longest_common_substring(self.name, other.name)
        else:
            name = ''
        new = Frame(array = self.data, name = name, rotate=True, photon_value=self.photon_value)
        new.data = new.data + other.data
        new.time = self.time + other.time
        return new
        
    def filter(self, threshold_min = 10, threshold_max = 150):
        """
        Returns a new Frame with pixels outside the specified range filtered out.
        """
        new = copy.deepcopy(self)
        data = new.data
        lineout = np.sum(data, axis = 1)
        mean, std = np.mean(lineout), np.std(lineout)
        row_outliers = (np.abs(lineout - mean)/std > 2)
        pixel_outliers = (data < threshold_min) | (data > threshold_max)
        data[row_outliers[:, np.newaxis] | pixel_outliers] = 0.
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
            show = True, peaknormalize = False, normalize=False, **kwargs):
        if not label:
            label = self.name
        return _plot_lineout(*self.get_lineout(rebin = rebin, smooth = smooth,**kwargs),
            show = show, error_bars = error_bars, label = label, peaknormalize = peaknormalize, normalize=normalize)

    def plot_histogram(self, show = True, binsize=1.000001, xmin=5, xmax=255, calibrate = [None,None], **kwargs):
        """
        Plot and return a histogram of pixel values.

        kwargs are passed to plt.plot.
        """
        _plot_histogram(self.data.flatten(),xmin=xmin,xmax=xmax,binsize=binsize,calib=calibrate,show=show,**kwargs)

    def counts_per_second(self, elapsed = None, start = 0, end = -1):
        """
        Return average counts per second the exposures that constitute this Frame.
        """
        if elapsed is None:
            elapsed = self.time
        return np.sum(self._raw_lineout()[start:end]) / self.time

# class Monitor:
#     def __init__(self, *args, transpose = True, vmax = 150, rebin = 1, **kwargs):
#         self.run = DataRun(*args, **kwargs)
#         self.vmax = vmax
#         self.rebin = rebin

#     def frame(self):
#         return self.run.get_frame()
    
#     def update(self):
#         self.run.show(vmax = self.vmax)
#         self.run.plot_lineout(rebin = self.rebin)
#         self.frame().plot_histogram(xmin = 0, xmax = self.vmax)
        
#     def stop(self):
#         self.run.stop()

def runlist(name,number,time=None,theta=None,z=None):
    monitorinstance = Monitor(threshold = 2, window_min = 120, window_max = 132, photon_value = 126,
            run_prefix = name+str(number), htime=time)
    monitorinstance.run.theta=theta
    monitorinstance.run.z=z
    return monitorinstance

def runlist_update(runlist,show=True,label='',**kwargs):
    print([x.run.counts_per_second() for x in runlist])
    print([x.run.acquisition_time() for x in runlist])
    
    plotdict=dict(energy=(None,None),yrange=[800,1600],rebin=3, peaknormalize=False)
    for key, value in kwargs.items():
        if key in plotdict:
            plotdict[key]=value
    
    [x.run.plot_lineout(**plotdict,show=False) for x in runlist]
                        #label=str(x.run.z)+'mm'+' - '+fwhm_ev(x.run.get_frame().get_lineout(**plotdict))+'eV') for x in runlist]
    if show:
        plt.show()

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
