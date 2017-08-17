import numpy as np
from ctypes import c_int
import numpy.ctypeslib as npct
import os

def do_decluster(arr2d, threshold, dtype = np.uint8):
    # TODO: does ascontiguousarray slow things down if arr2d is already
    # contiguous?
    # TODO: add support for uint8 arrays in the c module
    arr1d = np.ascontiguousarray(arr2d.ravel(), dtype = np.uint8)
    declustered = np.zeros_like(arr1d)
    arr_uint = npct.ndpointer(dtype = np.uint8, ndim = 1, flags = 'C_CONTIGUOUS')
    # load the c extension
    libcd = npct.load_library("libclusters", os.path.dirname(__file__)+'/../lib/')

    libcd.searchFrame_array_8.restype = None
    libcd.searchFrame_array_8.argtypes = [arr_uint, arr_uint, c_int, c_int, c_int]

    dimx, dimy = np.shape(arr2d)
    libcd.searchFrame_array_8(declustered, arr1d, dimx, dimy, threshold)
    return declustered.reshape(np.shape(arr2d))
