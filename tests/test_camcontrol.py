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
