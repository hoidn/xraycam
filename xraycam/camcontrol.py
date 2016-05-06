from paramiko import SSHClient
import paramiko
import numpy as np
import time
from . import detconfig
import argparse
import matplotlib.pyplot as plt
import pkg_resources
import os
import copy
import pdb
from functools import reduce

PKG_NAME = __name__.split('.')[0]

def resource_f(fpath):
    from io import StringIO
    return StringIO(pkg_resources.resource_string(PKG_NAME, fpath))

def resource_path(fpath):
    return pkg_resources.resource_filename(PKG_NAME, fpath)


def adc_to_eV(adc_values):
    """Generate an energy scale"""
    calib_slope = detconfig.calib_slope
    calib_intercept = detconfig.calib_intercept
    return adc_values * calib_slope + calib_intercept
    

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

        self.frame = Frame(array = self.get_array())

    def get_frame(self):
        return self.frame

    def get_array(self):
        """
        Get the sum of exposures for a dataset prefix as a numpy array
        """
        name= self.prefix + 'sum.dat'
        if not os.path.isfile(name):
            os.system('rsync -avz debian@' + detconfig.BBB_IP + ':' +\
                detconfig.base_path + name + ' ' + name)
        return np.reshape(np.fromfile(name, dtype = 'uint32'), (1024, 1280))

    def get_histograms(self):
        def plot_one(name):
            if not os.path.isfile(name):# or args.reload:
                get_ipython().system('scp -r debian@' + detconfig.BBB_IP +\
                    ':' + detconfig.base_path + '/' + name + ' ' + name)
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

    def plot_lineout(self, show = False, filter = False, smooth = 0, **kwargs):
        if filter:
            lineout = self.frame.filter(**kwargs).lineout(smooth = smooth)
        else:
            lineout = self.frame.lineout(smooth = smooth)
        plt.plot(lineout)
        if show:
            plt.show()
        return lineout

    def filter_frame(self, **kwargs):
        self.frame = self.frame.filter(**kwargs)

    def show(self, **kwargs):
        self.frame.show(**kwargs)

class RunSet:
    """
    Class containing a collection of DataRun instances.
    """
    def __init__(self, prefixes = None, number_runs = 0, run_prefix = None, **kwargs):
        """
        prefixes : list of str
            A list of dataset prefixes.
        number_runs : int
        run_prefix : str
        
        Call using either a list of run prefixes
        OR number_runs and run_prefix. 
        """
        if prefixes is None:
            if run_prefix is None:
                prefixes = [str(time.time()) for _ in range(number_runs)]
            else:
                prefixes = [run_prefix + '_%d' % i for i in range(number_runs)]
        self.dataruns = [DataRun(run_prefix = prefix, **kwargs) for prefix in prefixes]

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
        frames = list(map(Frame, datarun_arrays))
        def framefilter(frame):
            return frame.filter(**kwargs)
        return reduce(operator.add, list(map(framefilter, frames)))

class Frame:
    def __init__(self, array = None, numExposures = 0):
        """Construct using either a DataRun instance, or another Frame."""
        self.data = array
        self.numExposures = numExposures

    def show(self, **kwargs):
        """Show the frame. Kwargs are passed through to plt.imshow."""
        plt.imshow(self.data, **kwargs)
        plt.show()

    def get_data(self, **kwargs):
        return self.data

    def __add__(self, other):
        new = Frame(array = self.data)
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

    def lineout(self, smooth = 0):
        from scipy.ndimage.filters import gaussian_filter as gf
        return gf(np.sum(self.data, axis = 0), smooth)


