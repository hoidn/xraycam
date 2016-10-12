# Initialize stuff
from __future__ import division
from __future__ import absolute_import
import matplotlib
from itertools import imap
from itertools import izip
matplotlib.use(u'nbagg')

import numpy as np
from functools import partial
import time

from xraycam import camcontrol
from xraycam.camcontrol import plt
import xraycam.config as config


import warnings
warnings.filterwarnings(u"ignore",category=DeprecationWarning)

def runset_and_merge(run_prefix, numExposures = 1000, run = False, update_interval = 1000, window_min = 31, window_max =  55, threshold_min = 31, threshold_max =  55, **kwargs):
    u"""Returns a Frame"""
    datarun = camcontrol.DataRun(run_prefix= run_prefix,
       run = run, numExposures = numExposures, update_interval = update_interval,
        window_min = window_min, window_max = window_max, **kwargs)
    runset = camcontrol.RunSet([datarun])
    return runset.filter_reduce_frames(threshold_min = threshold_min, threshold_max = threshold_max)

def datarun_and_frame(run_prefix, numExposures = 1000, run = False, update_interval = 100,
        window_min = 31, window_max =  55, photon_value = 45., lineout = False, rebin = 10,
        smooth = 1, error_bars = True, post_filter = True, **kwargs):
    datarun = camcontrol.DataRun(run_prefix= run_prefix,
       run = run, numExposures = numExposures, update_interval = update_interval,
        window_min = window_min, window_max = window_max, photon_value = photon_value,
        **kwargs)
    
    if post_filter:
        frame = datarun.get_frame().filter(threshold_min = window_min, threshold_max = window_max)
    else:
        frame = datarun.get_frame()
    if lineout:
        frame.plot_lineout(rebin = rebin, smooth = smooth, error_bars = error_bars)
    return datarun, frame

def runset_merge_plot(*args, **kwargs):
    if 'error_bars' in kwargs: error_bars = kwargs['error_bars']; del kwargs['error_bars']
    else: error_bars =  True
    if 'smooth' in kwargs: smooth = kwargs['smooth']; del kwargs['smooth']
    else: smooth =  1
    if 'rebin' in kwargs: rebin = kwargs['rebin']; del kwargs['rebin']
    else: rebin =  10
    frame = runset_and_merge(*args, **kwargs)
    return frame, frame.plot_lineout(rebin = rebin, smooth = smooth, error_bars = error_bars)

def lineout_subregion(frame, cutoff, rebin = 4, error_bars = True, **kwargs):
    u"""Plot a lineout using the lower `cutoff` rows of the frame"""
    import copy
    nframe = copy.deepcopy(frame)
    nframe.data = nframe.data[len(frame.data) - cutoff:, :]
    return nframe.plot_lineout(rebin = rebin, error_bars = error_bars)

def plot_width_progression(dataruns, bragg_degrees, energy, deltatheta = 5.2e-4/10,
        xmin = 0, xmax = 1000, smooth = 5, rebin = 1):
    u"""
    dataruns : list of DataRun instances
    bragg_degrees : float
        Bragg angle in degrees.
    energy : float
        Photon energy in eV.
    deltatheta : float
        Angular size of a pixel.
    xmax : int
        Upper index at which to truncate the lineout.
    xmin : int
        Lower index at which to truncate the lineout.
    """
    bragg = np.deg2rad(bragg_degrees)
    deltaE = energy * deltatheta * np.cos(bragg)/np.sin(bragg)
    from xraycam.camalysis import fwhm
    lineouts = [dr.get_frame().get_lineout(smooth = 3, xmin = xmin, xmax = xmax)[1] for dr in dataruns]
    fwhmlist = list(imap(lambda l: u"FWHM: {:.4f} eV".format(deltaE * fwhm(l)), lineouts))

    salpha_width_study_lineouts =\
        [dr.plot_lineout(peaknormalize = True, xmin = xmin,
            xmax = xmax, smooth = smooth, rebin = rebin, show = False, label = l)
        for dr, l
        in izip(dataruns, fwhmlist)]
    plt.show()
    return salpha_width_study_lineouts
