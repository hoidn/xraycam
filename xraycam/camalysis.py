from . import detconfig
from . import camcontrol
from . import utils
import numpy as np
from scipy.interpolate import UnivariateSpline

@utils.memoize(timeout = None)
def get_hot_pixels(threshold = 0):
    """
    Return tuple (x, y) of indices of pixels above threshold in the sensor-specific dark run.
    """
    darkrun = camcontrol.DataRun(run_prefix = detconfig.darkrun_prefix_map[detconfig.sensor_id])
    array = darkrun.get_array()
    return np.where(array > threshold)

def fwhm(arr1d):
    """
    Given an array containing a peak, return its FWHM based on a spline interpolation.
    """
    x = np.arange(len(arr1d))
    spline = UnivariateSpline(x, arr1d - np.max(arr1d)/2, s = 0)
    r1, r2 = spline.roots()
    return r2 - r1
