import numpy as np
from xraycam.camalysis import *

def test_fwhm():
    def make_norm_dist(x, mean, sd):
        return 1.0/(sd*np.sqrt(2*np.pi))*np.exp(-(x - mean)**2/(2*sd**2))
    x = np.arange(1000)
    spec = make_norm_dist(x, 500, 50)
    assert np.isclose(fwhm(spec), 2*np.sqrt(2*np.log(2)) * 50)

#def test_get_hot_pixels(dark
