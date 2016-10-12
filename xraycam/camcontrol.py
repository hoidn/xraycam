from __future__ import division
from __future__ import absolute_import
import numpy as np
import detconfig
import pkg_resources
import copy
import pdb
import operator
#import _thread
import logging
import os
import time

from . import utils
from . import config
from itertools import imap


# TODO: move this setting from config.py to detconfig.py
if config.plotting_mode == u'notebook':
    from xraycam.mpl_plotly import plt
    #import sys
    #sys.stdout = open(config.logfile_path, 'w')
else:
    import matplotlib.pyplot as plt

logging.basicConfig(filename=u'xraycam.log', level=logging.DEBUG)

# TODO: set up keypair authentication at first usage of the package, if necessary,
# or else simply switch to using password authentication throughout. 

PKG_NAME = __name__.split(u'.')[0]

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


def calib_params():
    u"""
    Returns (slope, intercept) for the energy calibration.
    """
    calib_slope =\
        (detconfig.point2.energy - detconfig.point1.energy)/\
        (detconfig.point2.ADC - detconfig.point1.ADC)
    calib_intercept = detconfig.point1.energy -\
        detconfig.point1.ADC * calib_slope
    return calib_slope, calib_intercept

def adc_to_eV(adc_values):
    u"""Generate an energy scale"""
    calib_slope, calib_intercept = calib_params()
    return adc_values * calib_slope + calib_intercept

# from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_substring
def _longest_common_substring(s1, *rest):
    if not rest:
        return s1
    else:
        s2, newrest = rest[0], rest[1:]
    m = [[0] * (1 + len(s2)) for i in xrange(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in xrange(1, 1 + len(s1)):
        for y in xrange(1, 1 + len(s2)):
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
    u"""
    Rebin `x` and `y` into arrays of length `int(len(x)/rebin)`. The
    highest-x bin is dropped in case len(x) isn't a multiple of rebin.

    x is assumed to be evenly-spaced and in ascending order.

    Returns: x, y
    """
    def group(arr1d, op = np.mean):
        u"""
        op: a function to evaluate on each new bin that returns a numeric value.
        >>> rebin = 3
        >>> group(range(10))
        [1.0, 4.0, 7.0]
        """
        import itertools
        i = itertools.count()
        def key(dummy):
            xindx = i.next()
            return int(xindx/rebin)
        return [op(list(values)) for groupnum, values in itertools.groupby(arr1d, key = key)][:-1]
    return group(x, np.mean), group(y, np.sum)

def _get_poisson_uncertainties(intensities):
    u"""
    Return the array of Poisson standard deviations, based on photon
    counting statistics alone, for an array containing summed ADC
    values.
    """
    return np.sqrt(np.array(intensities))

def _plot_lineout(pixeli, intensity, show = False, label = u'', error_bars = False,
        peaknormalize = False):
    if peaknormalize:
        norm = np.max(intensity)
    else:
        norm = 1.
    if error_bars:
        if not (config.plotting_mode == u'notebook'):
            raise NotImplementedError(u"Error bars not supported in matplotlib mode")
        bars = _get_poisson_uncertainties(intensity) / norm
        error_y = dict(
            type = u'data',
            array = bars,
            visible = True
        )
        plt.plot(pixeli, intensity/norm, label = label, error_y = error_y)
    else:
        plt.plot(pixeli, intensity/norm, label = label)
    if show:
        plt.show()
    return intensity/norm

def _plot_histogram(values, show = True, xmin = None, xmax = None,
        calibrate = False, label = u'', **kwargs):
#    if x is None:
#        x = np.array(list(range(len(values))))
    if calibrate:
        plt.xlabel(u'Energy (eV)')
        values = adc_to_eV(values)
        #x = adc_to_eV(x)
        #plt.plot(x, values, label = label, **kwargs)
    else:
        plt.xlabel(u'ADC value')
    if xmin is not None:
        values[values < xmin] = 0
    if xmax is not None:
        values[values > xmax] = 0
    # TODO: make mpl_plotly interpret the kwarg `bins` (instead of `nbinsx`)
    plt.hist(values, label = label, **kwargs)
    if show:
        plt.show()

# TODO: reimplement inheritance
class DataRun(object):
    def __init__(self, run_prefix = u'', rotate = False, photon_value = 45., *args, **kwargs):
        self.rotate = rotate
        self.photon_value = photon_value
        if detconfig.detector == u'zwo':
            from . import zwo
            self.base = zwo.ZRun(run_prefix = run_prefix, *args, **kwargs)
            #self.stop = base.stop
        elif detconfig.detector == u'beaglebone':
            from . import bbb
            self.base = bbb.DataRun(run_prefix = run_prefix, *args, **kwargs)
            self.check_complete = self.base.check_complete
        else:
            raise ValueError
        self.name = run_prefix


    def acquisition_time(self):
        return self.base.acquisition_time()
    def get_array(self):
        return self.base.get_array()
    def get_histograms(self):
        return self.base.get_histograms()
    def stop(self):
        u"""
        Stop the acquisition.
        """
        self.base.stop()

    def get_frame(self):
        # TODO: do we need this attribute?
        time = self.acquisition_time()
        self._frame = Frame(array = self.get_array(), name = self.name,
            photon_value = self.photon_value, time = time)
        return self._frame

    @staticmethod
    def time_to_numexposures(timestring):
        u"""
        Convert a human-readable time string (e.g. '3m', '1h', etc.) to a number of exposures.
        """
        import humanfriendly
        def roundup(x):
            return int(np.ceil(x / 10.0)) * 10
        return roundup(config.frames_per_second * humanfriendly.parse_timespan(timestring))

    def plot_histogram(self, cluster_rejection = True, show = True,
            calibrate = True, **kwargs):
        u"""
        Plot histogram output from the camera acquisition.
        """
        pixels, singles = self.get_histograms()
        if cluster_rejection:
            values = singles
        else:
            values = pixels
        _plot_histogram(values, show = show, calibrate = calibrate, **kwargs)

    def plot_lineout(self, **kwargs):
        u"""
        kwargs are passed to self.frame.filter.
        """
        return self.get_frame().plot_lineout(**kwargs)

    def show(self, **kwargs):
        self.get_frame().show(**kwargs)

    def counts_per_second(self, **kwargs):
        return self.get_frame().counts_per_second(elapsed = self.acquisition_time(), **kwargs)


#loc = locals()
def set_detector(detid):
    if detid == u'beaglebone' or detid == u'zwo':
        detconfig.detector = detid
        if detid == u'zwo':
            # Module import causes oaCapture to launch
            from . import zwo
    else:
        raise ValueError(u"detector identifier: must be 'beaglebone' or 'zwo'")

class RunSequence(object):
    u"""
    Class that creates a sequence of DataRun instances, letting each data collection
    finish before beginning the next.
    """
    def __init__(self, prefix = None, number_runs = 0, run_prefix = None,
            htime = None, **kwargs):
        u"""
        prefix : str
            Datarun prefix prefix.
        number_runs : int
        htime : str
        """
        if htime is None:
            raise ValueError(u"kwarg htime MUST be provided to instantiate RunSequence.")
        if prefix is None:
            prefixes = [unicode(time.time()) for _ in xrange(number_runs)]
        else:
            prefixes = [prefix + u'_%d' % i for i in xrange(number_runs)]
        self.funcalls = [lambda prefix=prefix: DataRun(run_prefix = prefix, htime = htime, **kwargs)
            for prefix in prefixes]
        self.current = None


    def __iter__(self):
        return self

    def _wait_current_complete(self):
        u"""
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

    def next(self):
        self._wait_current_complete()
        try:
            run = self.funcalls.pop()()
            self.current = run
        except IndexError:
            raise StopIteration
        return run


class RunSet(object):
    u"""
    Class containing a collection of DataRun instances.
    """
    def __init__(self, *args, **kwargs):
        u"""
        TODO docstring
        """
        from xraycam import async
        self.dataruns = async.IterThread(RunSequence(*args, **kwargs))

    def insert(self,  datarun = None, *args, **kwargs):
        u"""
        datarun : DataRun

        If datarun is provided, insert is into this RunSet. Otherwise pass
        args and kwargs to the constructor for DataRun and insert the resulting
        object into this RunSet.
        """
        if datarun is None:
            datarun = DataRun(*args, **kwargs)
        self.dataruns.append(datarun)

    def get_dataruns(self):
        return self.dataruns

    def plot_lineouts(self, *args, **kwargs):
        if 'show' in kwargs: show = kwargs['show']; del kwargs['show']
        else: show =  True
        def one_lineout(datarun):
            return datarun.plot_lineout(*args, show = False, **kwargs)
        lineouts = list(imap(one_lineout, self.dataruns))
        if show:
            plt.show()
        return lineouts

    def filter_reduce_frames(self, **kwargs):
        u"""
        Do artifact filtering for frames of all component data runs and
        merge all the resulting Frames into a new Frame instance.

        kwargs are passed through to DataRun.filter
        """
        def make_frame(dr):
            old_frame = dr.get_frame()
            return Frame(old_frame.get_data(), name = self.name,
                time = old_frame.time, rotate = self.rotate)
        frames = list(imap(make_frame, self.dataruns))
        def framefilter(frame):
            return frame.filter(**kwargs)
        return reduce(operator.add, list(imap(framefilter, frames)))

class Frame(object):
    def __init__(self, array = None, time = 0., name = u'', photon_value = 45.,
            rotate = False):
        if rotate:
            self.data = array
        else:
            self.data = np.rot90(array)
        self.time = time
        self.name = name
        self.photon_value = photon_value

    def show(self, **kwargs):
        u"""Show the frame. Kwargs are passed through to plt.imshow."""
        plt.imshow(self.data, **kwargs)
        plt.show()

    def get_data(self, **kwargs):
        return self.data

    def __add__(self, other):
        if self.name and other.name:
            name =  _longest_common_substring(self.name, other.name)
        else:
            name = u''
        new = Frame(array = self.data, name = name)
        new.data = new.data + other.data
        new.time = self.time + other.time
        return new
        
    def filter(self, threshold_min = 10, threshold_max = 150):
        u"""
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
        u"""
        darkrun : DataRun
            A dark run to use as the hot pixel mask
        Returns a new Frame with hot pixels removed.
        """
        from . import camalysis
        new = copy.deepcopy(self)
        hot_indices = camalysis.get_hot_pixels(darkrun = darkrun, threshold = threshold)
        new.data[hot_indices] = 0
        return new

    def _raw_lineout(self):
        return np.sum(self.data, axis = 0) / self.photon_value

    def get_lineout(self, rebin = 1, smooth = 0, xmin = 0, xmax = -1):
        u"""
        Return a smoothed and rebinned lineout of self.data.

        smooth : number of pixel columns by which to smooth rebin :
        number of pixel columns per bin

        Returns: bin values, intensities
        """
        def apply_smooth(arr1d):
            from scipy.ndimage.filters import gaussian_filter as gf
            return gf(arr1d, smooth)
        def apply_rebin(arr1d):
            return _rebin_spectrum(np.array(xrange(len(arr1d))), arr1d, rebin = rebin)

        if (not isinstance(rebin, int)) or rebin < 1:
            raise ValueError(u"Rebin must be a positive integer")

        return compose(apply_rebin, apply_smooth)(self._raw_lineout()[xmin: xmax])

    def plot_lineout(self, smooth = 0, error_bars = False, rebin = 1, label = u'',
            show = True, peaknormalize = False, xmin = 0, xmax = -1):
        if not label:
            label = self.name
        return _plot_lineout(*self.get_lineout(rebin = rebin, smooth = smooth,
                xmin = xmin, xmax = xmax),
            show = show, error_bars = error_bars, label = label, peaknormalize = peaknormalize)

    def plot_histogram(self, show = True, calibrate = False, **kwargs):
        u"""
        Plot and return a histogram of pixel values.

        kwargs are passed to plt.plot.
        """
        flat = self.data.flatten()
        nonzero_flat = flat[flat != 0]
        _plot_histogram(nonzero_flat, show = show,
                calibrate = calibrate, **kwargs)

    def counts_per_second(self, elapsed = None, start = 0, end = -1):
        u"""
        Return average counts per second the exposures that constitute this Frame.
        """
        if elapsed is None:
            elapsed = self.time
        return np.sum(self._raw_lineout()[start:end]) / self.time
