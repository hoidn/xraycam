# Initialize stuff
import matplotlib
matplotlib.use('nbagg')

import numpy as np
from functools import partial
import time

from xraycam import camcontrol
from xraycam.camcontrol import plt
import xraycam.config as config


import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

def runset_and_merge(run_prefix, numExposures = 1000, run = False, update_interval = 1000, window_min = 31, window_max =  55, threshold_min = 31, threshold_max =  55, **kwargs):
    """Returns a Frame"""
    datarun = camcontrol.DataRun(run_prefix= run_prefix,
       run = run, numExposures = numExposures, update_interval = update_interval,
        window_min = window_min, window_max = window_max, **kwargs)
    runset = camcontrol.RunSet([datarun])
    return runset.filter_reduce_frames(threshold_min = threshold_min, threshold_max = threshold_max)

def datarun_and_frame(run_prefix, numExposures = 1000, run = False, update_interval = 100,
        window_min = 31, window_max =  55, photon_value = 45., lineout = True, rebin = 10,
        smooth = 1, error_bars = True, post_filter = True, **kwargs):
    datarun = camcontrol.DataRun(run_prefix= run_prefix,
       run = run, numExposures = numExposures, update_interval = update_interval,
        window_min = window_min, window_max = window_max, photon_value = photon_value,
        **kwargs)
    
    time.sleep(float(update_interval) / config.frames_per_second + 1)
    if post_filter:
        frame = datarun.get_frame().filter(threshold_min = window_min, threshold_max = window_max)
    else:
        frame = datarun.get_frame()
    if lineout:
        frame.plot_lineout(rebin = rebin, smooth = smooth, error_bars = error_bars)
    return datarun, frame

def runset_merge_plot(*args, rebin = 10, smooth = 1, error_bars = True, **kwargs):
    frame = runset_and_merge(*args, **kwargs)
    return frame, frame.plot_lineout(rebin = rebin, smooth = smooth, error_bars = error_bars)

def lineout_subregion(frame, cutoff, rebin = 4, error_bars = True, **kwargs):
    """Plot a lineout using the lower `cutoff` rows of the frame"""
    import copy
    nframe = copy.deepcopy(frame)
    nframe.data = nframe.data[len(frame.data) - cutoff:, :]
    return nframe.plot_lineout(rebin = rebin, error_bars = error_bars)
