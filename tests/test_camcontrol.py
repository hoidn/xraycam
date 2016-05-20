import numpy as np
import time

from xraycam import camcontrol


    
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

def test_get_lineout():
    assert npcomp(frame.get_lineout(rebin = 500), [np.array([ 249.5,  749.5]), np.array([ 75983, 711786], dtype='uint64')])
