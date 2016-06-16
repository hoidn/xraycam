from . import detconfig
from . import camcontrol
from . import utils
import numpy as np

@utils.memoize(timeout = None)
def get_hot_pixels(threshold = 0):
    """
    Return tuple (x, y) of indices of pixels above threshold in the sensor-specific dark run.
    """
    darkrun = camcontrol.DataRun(run_prefix = detconfig.darkrun_prefix_map[detconfig.sensor_id])
    array = darkrun.get_array()
    return np.where(array > threshold)
