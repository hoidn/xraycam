# Initialize stuff
from __future__ import absolute_import
import matplotlib
matplotlib.use(u'nbagg')

import numpy as np
from functools import partial

from xraycam import camcontrol
from xraycam.camcontrol import plt

import warnings
warnings.filterwarnings(u"ignore",category=DeprecationWarning)

def runset_and_merge(run_prefix, number_runs, run = False, threshold_min = 31, threshold_max =  55, **kwargs):
    u"""Returns a Frame"""
    runset = camcontrol.RunSet(run_prefix= run_prefix,
        run = run, number_runs = number_runs, **kwargs)
    return runset.filter_reduce_frames(threshold_min = threshold_min, threshold_max = threshold_max)

def runset_merge_plot(*args, **kwargs):
    if 'rebin' in kwargs: rebin = kwargs['rebin']; del kwargs['rebin']
    else: rebin =  10
    frame = runset_and_merge(*args, **kwargs)
    return frame, frame.plot_lineout(rebin = rebin)

def lineout_subregion(frame, cutoff, rebin = 4, error_bars = True, **kwargs):
    u"""Plot a lineout using the lower `cutoff` rows of the frame"""
    import copy
    nframe = copy.deepcopy(frame)
    nframe.data = nframe.data[len(frame.data) - cutoff:, :]
    return nframe.plot_lineout(rebin = rebin, error_bars = error_bars)
