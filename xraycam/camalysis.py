from . import detconfig
from . import camcontrol
from . import utils
import numpy as np
from scipy.interpolate import UnivariateSpline

@utils.memoize(timeout = None)
def get_hot_pixels(darkrun = None, threshold = 0):
    """
    darkrun : camcontrol.DataRun
        A dark run 
    threshold : int
        The threshold value for hot pixels Return tuple (x, y) of
        indices of pixels above threshold in the provided dark run. Reverts to
        the sensor-specific dark run `detconfig.sensor_id` if `darkrun == None`.
    """
    if darkrun is None:
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

def _plot_histogram(values, show = True,
        calibrate = False, label = '', **kwargs):
#    if x is None:
#        x = np.array(list(range(len(values))))
    if calibrate:
        plt.xlabel('Energy (eV)')
        values = adc_to_eV(values)
        #x = adc_to_eV(x)
        #plt.plot(x, values, label = label, **kwargs)
    else:
        plt.xlabel('ADC value')
    # TODO: make mpl_plotly interpret the kwarg `bins` (instead of `nbinsx`)
    plt.hist(values, label = label, **kwargs)
    if show:
        plt.show()
