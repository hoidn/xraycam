import time
import numpy as np
import os
from paramiko import SSHClient
import paramiko
import subprocess, pipes

from . import config
from . import utils
from .camalysis import _plot_histogram
import detconfig

def exists_remote(host, path):
    """Test if a file exists at path on a host accessible with SSH."""
    status = subprocess.call(
        ['ssh', host, 'test -f {}'.format(pipes.quote(path))])
    if status == 0:
        return True
    if status == 1:
        return False
    raise Exception('SSH failed')

def _get_destination_dirname(source, dest = None):
    """
    source : file path on the beaglebone relative to detid.base_path
    dest : FULL path of the file to be copied (not just the target
    directory.

    Return the full data target path on the local machine.
    """
    if dest is None:
        dest = source
    dirname, basename = os.path.split(dest)
    if not dirname:
        dirname = '.'
    return dirname + '/'

def _copy_file(source, dest = None, partial_suffix = '.part'):
    """
    source : str
        -File path on the beaglebone relative to detid.base_path. If no match is
        found this function will look for the file path `source + partial_suffix`, designating
        a file that is still being written to.
    dest : str
        FULL path of the file to be copied (not just the target
    directory.

    If a file matching `source` is found, return the string 'complete'.
    If a file matching `source + partial_prefix` is found, return the string 'partial'.
    """
    dirname = _get_destination_dirname(source, dest)
    host = detconfig.host
    path = detconfig.base_path + source

    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if exists_remote(host, path):
        os.system('rsync -avz ' + host + ':' +\
            path + ' ' +  dirname)
        return 'complete'
    else: # try to get the .part file
        print("{0} not found. \nSearching for {0}{1}".format(path, partial_suffix))
        os.system('rsync -avz ' + host + ':' +\
            path + partial_suffix + ' ' +  dirname)
        return 'partial'

class DataRun:
    def __init__(self, run_prefix = 'data/' + 'exposure.' + str(time.time()),
            run = False, numExposures = 40, htime = None, threshold = 15,
            window_min = 0, window_max = 255, gain = '0x7f', filter_sum = True,
            update_interval = 80, block = False, photon_value = 45.):
        """
        Instantiate a DataRun and optionally run an exposure sequence
        on the BBB.

        run_prefix : str
            Directory containing data files (relative to detconfig.base_path on the
            BBB; relative to run directory locally)
        reload : bool
            If true, run an exposure sequence. (If false, we assume that output files
            with the name prefix: run_prefix + run_name already exist on the BBB). 
        htime : str
            Time duration of the data acquisition if `run == True`, in a humanfriendly
            format, for example '3m' or '1h'. Overrides numExposures if provided.
        numExposures : int
            Number of exposures in the data acquisition if `run == True`.

        Raises ValueError if data corresponding to the run_prefix parameter already
        exists locally.
        TODO: etc..
        """
        self.run = run
        self.gain = gain
        self.threshold = threshold
        self.filter_sum = filter_sum
        self.prefix = run_prefix
        self.name = run_prefix
        self.filter_sum = filter_sum
        self.photon_value = photon_value

        self._time_start = time.time()
        self.arrayname= self.prefix + 'sum.dat'
        self._partial_suffix = '.part'
        if not htime:
            self.numExposures = numExposures
        else:
            self.numExposures = self.time_to_numexposures(htime)

        # TODO: tidy this up
        if run:
            if os.path.exists(self.arrayname):
                raise ValueError("Data file: %s already exists. Please choose a different run prefix." % self.arrayname)
            exposure_cmd = "screen -d -m bash -c 'time sudo ./main_mt9m001 %d -o " % threshold + run_prefix\
                + " -n " + str(self.numExposures) + " -g " + gain +\
                " -w %d -r %d %d" % (update_interval, window_min, window_max)
            if filter_sum:
                exposure_cmd += " -p"
            exposure_cmd += " >> xraycam.log '"
            print(exposure_cmd)

            ssh = SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(detconfig.BBB_IP, 22, detconfig.user, password = detconfig.password)

            #take an exposure
            (sshin2, sshout2, ssherr2) = ssh.exec_command('cd' + ' ' + detconfig.base_path\
                + '; ' + exposure_cmd)
            if block:
                time.sleep(float(numExposures) / config.frames_per_second + 3)
            else:
                time.sleep(float(update_interval) / config.frames_per_second + 3)
            ssh.close()
        

        if block:
            self._set_complete_state(True)
        else:
            self._set_complete_state(False)

        self._frame = None


    def acquisistion_time(self):
        default = float(self.numExposures) / config.frames_per_second
        if self.run == False:
            return default
        else:
            return min(time.time() - self._time_start, default)
                

    def _set_complete_state(self, state):
        self._complete = state

    def check_complete(self):
        """
        Return True if this datarun is complete, i.e. if its
        corresponding file exists locally or on the Beaglebone Black.
        """
        if not self._complete:
            path = detconfig.base_path + self.arrayname
            if os.path.isfile(self.arrayname) or exists_remote(detconfig.host, path):
                self._complete = True
        return self._complete


    @utils.memoize_timeout(timeout = 10)
    def get_array(self):
        """
        Get the sum of exposures for a dataset prefix as a numpy array
        """
        def get_and_process(suffix = ''):
            return np.reshape(np.fromfile(self.arrayname + suffix, dtype = 'uint32'), (1024, 1280))

        if not os.path.isfile(self.arrayname):
            if _copy_file(self.arrayname, self.arrayname) == 'complete':
                self._set_complete_state(True)
                return get_and_process(suffix = '')
            else:
                return get_and_process(suffix = self._partial_suffix)
        else:
            return get_and_process(suffix = '')

    def get_histograms(self):
        if not self.check_complete():
            raise Exception("Histogram data not available during readout.")
        def plot_one(name):
            if not os.path.isfile(name):# or args.reload:
                _copy_file(name, name)
            return np.fromfile(name, dtype = 'uint32')
        pixels = plot_one(self.prefix + 'pixels.dat')
        singles = plot_one(self.prefix + 'singles.dat')
        return pixels, singles

