import numpy as np
import time

from xraycam import camcontrol
from xraycam import config
from xraycam.nbinit import *


    
#def test_data_transfer():
#    def get_datarun(run_prefix, run = False):
#        return camcontrol.DataRun(run_prefix = run_prefix, run = run, gain = '0x3f')
#    import os
#    import time
#    prefix = 'data/' + str(time.time())
#    # Should run exposure series on BBB
#    get_datarun(prefix, run = True)
#    os.system('rm -r data')
#    # Should copy data from BBB
#    get_datarun(prefix, run = False)
#    # Should load existing data
#    get_datarun(prefix, run = False)
import numpy as np

def npcomp(a, b):
    """
    Compare two iterables, returning true if all elements are
    the same and false otherwise. Nesting is supported.
    """
    from functools import reduce
    fand = lambda x, y: x & y
    if isinstance(a, tuple) or isinstance(a, list):
        return reduce(fand, map(npcomp, a, b))
    elif isinstance(a, str):
        return a == b
    else:
        return np.all(np.isclose(a, b))

def test_rebin():
    c, d = camcontrol._rebin_spectrum(list(range(20)), np.array(list(range(20))))
    assert np.all(
            d == np.array([10, 35, 60])
            )
    return 0

def test_compose():
    from xraycam.camcontrol import compose
    def f(*args, **kwargs):
        return args[0]
    def g(*args, **kwargs):
        return args[0]
    assert compose(f, g)(5) == 5

#def test_get_lineout():
#    assert npcomp(frame.get_lineout(rebin = 500), [np.array([ 249.5,  749.5]), np.array([ 75983, 711786], dtype='uint64')])

def test_runset():
    frame = runset_and_merge('data/5.6.real1__1', gain = '0x3f', run = False,
        window_min = 31, window_max = 55, threshold_min = 31, threshold_max = 55, numExposures = 1000)
    assert frame.numExposures > 0

def test_exisiting_data():
    import time
    import pytest
    tname = str(time.time())
    test_frame = runset_and_merge(tname, gain = '0x3f', run = True, window_min = 31,
        window_max = 55, numExposures = 40,  threshold_min = 31, threshold_max = 55,
        update_interval = 40, block = True)
    test_run = camcontrol.DataRun(tname, run = False)
    assert test_run.check_complete()
    with pytest.raises(Exception) as e_info:
        test_frame = runset_and_merge(tname, gain = '0x3f', run = True, window_min = 31,
            window_max = 55, numExposures = 40,  threshold_min = 31, threshold_max = 55,
            update_interval = 40, block = True)

def test_number_exposures():
    run = camcontrol.DataRun('foobar', numExposures = 100)
    assert run.numExposures == 100
    run2 = camcontrol.DataRun('foobar2', htime = '3h')
    assert run2.numExposures == 10800 * int(config.frames_per_second)

def test_acquisition_time():
    run = camcontrol.DataRun('foobar3', htime = '5s')
    time.sleep(2)
    assert run._acquisistion_time() >= 2.
    time.sleep(4)
    assert run._acquisistion_time() == 5.

def test_countrate():
    run = camcontrol.DataRun('foobar3', htime = '5s')
    def flattime():
        return 1.
    frame = camcontrol.Frame(np.ones((1024, 1280)))
    def flatarr():
        return frame
    run.get_frame = flatarr
    run._acquisistion_time = flattime
    i1 =  run.counts_per_second(start = 10, end = 20)
    i2 =  run.counts_per_second(start = 10, end = 30)
    assert np.isclose(i2/i1, 2.)
    
