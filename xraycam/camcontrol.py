from paramiko import SSHClient
import paramiko
import numpy as np
import time
from . import detconfig
import argparse
import pkg_resources
import os
import copy
import pdb

from functools import reduce

from xraycam.mpl_plotly import plt
from xraycam import utils
from xraycam import config

PKG_NAME = __name__.split('.')[0]

# from https://gist.github.com/rossdylan/3287138
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

def get_file(source, dest = None):
    """
    source : file path on the beaglebone relative to detid.base_path
    dest : FULL path of the file to be copied (not just the target
    directory.
    """
    if dest is None:
        dest = source
    dirname, basename = os.path.split(dest)
    if not dirname:
        dirname = '.'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    dirname = dirname + '/'
    os.system('rsync -avz debian@' + detconfig.BBB_IP + ':' +\
        detconfig.base_path + source + ' ' +  dirname)

def adc_to_eV(adc_values):
    """Generate an energy scale"""
    calib_slope = detconfig.calib_slope
    calib_intercept = detconfig.calib_intercept
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
    Rebin `x` and `y` into arrays of length `int(len(x)/rebin)`. The highest-x
    bin is dropped in case len(x) isn't a multiple of rebin.

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
        return [op(list(values)) for groupnum, values in itertools.groupby(arr1d, key = key)][:-1]
    return group(x, np.mean), group(y, np.sum)

def _get_poisson_uncertainties(intensities):
    """
    Return the array of Poisson standard deviations, based on photon counting
    statistics alone, for an array containing summed ADC values.
    """
    return np.sqrt(np.array(intensities)*config.photon_ADC_value)

def _plot_lineout(pixeli, intensity, show = False, label = '', error_bars = True):
    if error_bars:
        if not (config.plotting_mode == 'notebook'):
            raise NotImplementedError("Error bars not supported in matplotlib mode")
        error_y = dict(
            type = 'data',
            array = _get_poisson_uncertainties(intensity),
            visible = True
        )
        plt.plot(pixeli, intensity, label = label, error_y = error_y)
    else:
        plt.plot(pixeli, intensity, label = label)
    if show:
        plt.show()
    return intensity

class DataRun:
    def __init__(self, run_prefix = 'data/' + 'exposure.' + str(time.time()),\
            run = False, numExposures = 40, threshold = 15, window_min = 0, window_max = 255,\
            gain = '0x7f', filter_sum = True):
        """
        Instantiate a DataRun and optionally run an exposure sequence
        on the BBB.

        run_prefix : str
            Directory containing data files (relative to detconfig.base_path on the
            BBB; relative to run directory locally)
        reload : bool
            If true, run an exposure sequence. (If false, we assume that output files
            with the name prefix: run_prefix + run_name already exist on the BBB). 
        TODO: etc..
        """
        self.gain = gain
        self.threshold = threshold
        self.numExposures = numExposures
        self.filter_sum = filter_sum
        self.prefix = run_prefix
        self.filter_sum = filter_sum

        if run:
            exposure_cmd = 'time sudo ./main_mt9m001 %d -o ' % threshold + run_prefix\
                + ' -n ' + str(numExposures) + ' -g ' + gain +\
                ' -r %d %d' % (window_min, window_max)
            if filter_sum:
                exposure_cmd += ' -p'

            #keypath = resource_path('data/id_rsa.pub')

            ssh = SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(detconfig.BBB_IP, 22, 'debian', password = 'bbb')

            #take an exposure
            (sshin2, sshout2, ssherr2) = ssh.exec_command('cd' + ' ' + detconfig.base_path\
                + '; ' + exposure_cmd)
            print(sshout2.read())
            ssh.close()

        self.frame = Frame(array = self.get_array(), name = self.prefix)

    def get_frame(self):
        return self.frame

    def get_array(self):
        """
        Get the sum of exposures for a dataset prefix as a numpy array
        """
        import time
        name= self.prefix + 'sum.dat'
        if not os.path.isfile(name):
            get_file(name, name)
        return np.reshape(np.fromfile(name, dtype = 'uint32'), (1024, 1280))

    def get_histograms(self):
        def plot_one(name):
            if not os.path.isfile(name):# or args.reload:
                get_file(name, name)
            return np.fromfile(name, dtype = 'uint32')
        pixels = plot_one(self.prefix + 'pixels.dat')
        singles = plot_one(self.prefix + 'singles.dat')
        return pixels, singles

    def plot_histograms(self, pixels_plot = True, singles_plot = True, show = True,
            block = False,  calibrate = True, **kwargs):
        pixels, singles = self.get_histograms()
        #plt.ylim((0, np.max(pixels[15:])))
        x = np.array(list(range(len(pixels))))
        if calibrate:
            plt.xlabel('Energy (eV)')
            x = adc_to_eV(x)
        else:
            plt.xlabel('ADC value')
        if pixels_plot:
            plt.plot(x, pixels, **kwargs)
        if singles_plot:
            plt.plot(x, singles, **kwargs)
        if show:
            plt.show(block = block)

    def plot_lineout(self, filter = False, show = False, smooth = 0, rebin = 1,
            error_bars = True, **kwargs):
        """
        kwargs are passed to self.frame.filter.
        """
        if filter:
            frame = self.frame.filter(**kwargs)
        else:
            frame = self.frame
        return frame.plot_lineout(rebin = rebin, smooth = smooth,
            label = self.prefix, error_bars = error_bars)


    def filter_frame(self, **kwargs):
        self.frame = self.frame.filter(**kwargs)

    def show(self, **kwargs):
        self.frame.show(**kwargs)

class RunSet:
    """
    Class containing a collection of DataRun instances.
    """
    def __init__(self, dataruns = None, prefixes = None, number_runs = 0, run_prefix = None, **kwargs):
        """
        dataruns : iterable containing `DataRun` instances.
        prefixes : list of str
            A list of dataset prefixes.
        number_runs : int
        run_prefix : str
        
        Instantiate a RunSet  using:
            (1) `dataruns`, if `dataruns is not None`, OR
            (2) prefixes, if provided, OR
            (3) `number_runs` and `run_prefix`
        """
        if dataruns is not None:
            self.dataruns = dataruns
        else:
            if prefixes is None:
                if run_prefix is None:
                    prefixes = [str(time.time()) for _ in range(number_runs)]
                else:
                    prefixes = [run_prefix + '_%d' % i for i in range(number_runs)]
            self.dataruns = [DataRun(run_prefix = prefix, **kwargs) for prefix in prefixes]
        if prefixes is not None:
            self.name = _longest_common_substring(*prefixes)
        else:
            self.name = ''

    def __add__(self, other):
        new = RunSet()
        new.dataruns = self.dataruns + other.dataruns
        return new

    def insert(self,  datarun = None, *args, **kwargs):
        """
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

    def plot_lineouts(self, *args, show = True, **kwargs):
        def one_lineout(datarun):
            return datarun.plot_lineout(*args, show = False, **kwargs)
        lineouts = list(map(one_lineout, self.dataruns))
        if show:
            plt.show()
        return lineouts

    def filter_reduce_frames(self, **kwargs):
        """
        Do artifact filtering for frames of all component data runs and return
        merge all the resulting Frames into a new Frame instance.

        kwargs are passed through to DataRun.filter
        """
        import operator
        datarun_arrays = [dr.get_frame().get_data() for dr in self.dataruns]
        frames = list(map(lambda arr: Frame(arr, name = self.name), datarun_arrays))
        def framefilter(frame):
            return frame.filter(**kwargs)
        return reduce(operator.add, list(map(framefilter, frames)))

class Frame:
    def __init__(self, array = None, numExposures = 0, name = ''):
        """Construct using either a DataRun instance, or another Frame."""
        self.data = array
        self.numExposures = numExposures
        self.name = name

    def show(self, **kwargs):
        """Show the frame. Kwargs are passed through to plt.imshow."""
        plt.imshow(self.data, **kwargs)
        plt.show()

    def get_data(self, **kwargs):
        return self.data

    def __add__(self, other):
        if self.name and other.name:
            name =  _longest_common_substring(self.name, other.name)
        else:
            name = ''
        new = Frame(array = self.data, name = name)
        new.data = new.data + other.data
        new.numExposures = self.numExposures + other.numExposures
        return new
        
    def filter(self, threshold_min = 10, threshold_max = 150):
        """
        Returns a new Frame with artifacts filtered out.
        """
        new = copy.deepcopy(self)
        data = new.data
        lineout = np.sum(data, axis = 1)
        mean, std = np.mean(lineout), np.std(lineout)
        row_outliers = (np.abs(lineout - mean)/std > 2)
        pixel_outliers = (data < threshold_min) | (data > threshold_max)
        data[row_outliers[:, np.newaxis] | pixel_outliers] = 0.
        return new

    def _raw_lineout(self):
        return np.sum(self.data, axis = 0)

    def get_lineout(self, rebin = 1, smooth = 0):
        """
        Return a smoothed and rebinned lineout of self.data.

        smooth : number of pixel columns by which to smooth
        rebin : number of pixel columns per bin
        """
        def apply_smooth(arr1d):
            from scipy.ndimage.filters import gaussian_filter as gf
            return gf(arr1d, smooth)
        def apply_rebin(arr1d):
            return _rebin_spectrum(list(range(len(arr1d))), arr1d, rebin = rebin)

        if (not isinstance(rebin, int)) or rebin < 1:
            raise ValueError("Rebin must be a positive integer")

        return compose(apply_rebin, apply_smooth)(self._raw_lineout())

    def plot_lineout(self, smooth = 0, error_bars = True, rebin = 1, label = None):
        if label is None:
            label = self.name
        return _plot_lineout(*self.get_lineout(rebin = rebin, smooth = smooth),
            error_bars = error_bars, label = label)

