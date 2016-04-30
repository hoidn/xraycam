from paramiko import SSHClient
import paramiko
import numpy as np
import time
import detconfig
import argparse
import matplotlib.pyplot as plt
import pkg_resources
import os

PKG_NAME = __name__.split('.')[0]

def resource_f(fpath):
    from StringIO import StringIO
    return StringIO(pkg_resources.resource_string(PKG_NAME, fpath))

def resource_path(fpath):
    return pkg_resources.resource_filename(PKG_NAME, fpath)


def adc_to_eV(adc_values):
    """Generate an energy scale"""
    calib_slope = detconfig.calib_slope
    calib_intercept = detconfig.calib_intercept
    return adc_values * calib_slope + calib_intercept
    

class DataRun:
    def __init__(self, run_prefix = 'data/', run_name = 'exposure.' + str(time.time()),\
            numExposures = 40, threshold = 15, window_min = 0, window_max = 255,\
            gain = '0x7f', filter_sum = True):
        """
        Runs a series of exposures on the camera.
        """
        exposure_cmd = 'time sudo ./main_mt9m001 %d -o ' % threshold + run_prefix\
            + run_name + ' -n ' + str(numExposures) + ' -g ' + gain +\
            ' -r %d %d' % (window_min, window_max)
        self.gain = gain
        self.threshold = threshold
        self.numExposures = numExposures
        self.filter_sum = filter_sum
        self.path = run_prefix + run_name
        self.filter_sum = filter_sum
        if filter_sum:
            exposure_cmd += ' -p'

        keypath = resource_path('data/id_rsa.pub')
        #keypath = "/home/oliver/.ssh/id_rsa.pub"

        ssh = SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(detconfig.BBB_IP, 22, 'debian', key_filename=keypath)

        #take an exposure
        (sshin2, sshout2, ssherr2) = ssh.exec_command('cd' + ' ' + detconfig.base_path\
            + '; ' + exposure_cmd)
        print sshout2.read()
        ssh.close()

    def get_frame(self):
        """
        Get the sum of exposures for a dataset prefix as a numpy array
        """
        name= self.path + 'sum.dat'
        if not os.path.isfile(name):
            os.system('rsync -avz debian@' + detconfig.BBB_IP + ':' +\
                detconfig.base_path + name + ' ' + name)
        return np.reshape(np.fromfile(name, dtype = 'uint32'), (1024, 1280))

    def get_histograms(self):
        def plot_one(name):
            if not os.path.isfile(name):# or args.reload:
                get_ipython().system(u'scp -r debian@' + detconfig.BBB_IP +\
                    ':' + detconfig.base_path + '/' + name + ' ' + name)
            return np.fromfile(name, dtype = 'uint32')
        pixels = plot_one(self.path + 'pixels.dat')
        singles = plot_one(self.path + 'singles.dat')
        return pixels, singles

    def plot_histograms(self, pixels_plot = True, singles_plot = True, show = False,
            block = False,  calibrate = True, **kwargs):
        pixels, singles = self.get_histograms()
        #plt.ylim((0, np.max(pixels[15:])))
        x = np.array(range(len(pixels)))
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

    def show(self):
        plt.show()

class Frame:
    def __init__(self, datarun = None, frame = None):
        """Construct using either a DataRun instance, or another Frame."""
        if datarun is not None:
            self.data = datarun.get_frame()
        elif frame is not None:
            self.data = frame.data

    def show(self, **kwargs):
        """Show the frame. Kwargs are passed through to plt.imshow."""
        plt.imshow(self.data, **kwargs)
        plt.show()

    def get_data(self, **kwargs):
        return self.data

    def __add__(self, other):
        new = Frame(frame = self)
        new.data = new.data + other.data
        return new
        


#def runsequence(exposures_per_run = 40, n_runs = 2, **kwargs):
#    frames = [autorun_and_get_prefix(numExposures = exposures_per_run, **kwargs) for _ in range(n_runs)]
#    return frames

