# Initialize stuff
import matplotlib
matplotlib.use('nbagg')

import numpy as np
from functools import partial

from . import camcontrol
from .camcontrol import plt

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

def runset_and_merge(run_prefix, number_runs, run = False, threshold_min = 31, threshold_max =  55, **kwargs):
    """Returns a Frame"""
    runset = camcontrol.RunSet(run_prefix= run_prefix,
        run = run, number_runs = number_runs, **kwargs)
    return runset.filter_reduce_frames(threshold_min = threshold_min, threshold_max = threshold_max)

def runset_merge_plot(*args, rebin = 10, smooth = 1, error_bars = True, **kwargs):
    frame = runset_and_merge(*args, **kwargs)
    return frame, frame.plot_lineout(rebin = rebin, smooth = smooth, error_bars = error_bars)

def lineout_subregion(frame, cutoff, rebin = 4, error_bars = True, **kwargs):
    """Plot a lineout using the lower `cutoff` rows of the frame"""
    import copy
    nframe = copy.deepcopy(frame)
    nframe.data = nframe.data[len(frame.data) - cutoff:, :]
    return nframe.plot_lineout(rebin = rebin, error_bars = error_bars)
