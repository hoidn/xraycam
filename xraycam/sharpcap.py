"""
Module for offline processing of .ser video captures from Sharpcap.
"""

import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int

from functools import reduce
from operator import add
from . import utils
from PIL import Image

PKG_NAME = __name__.split(u'.')[0]


def decluster(arr2d, threshold, nloop = 1):
    # TODO: does ascontiguousarray slow things down if arr2d is already
    # contiguous?
    arr1d = np.ascontiguousarray(arr2d.ravel(), dtype = np.uint16)
    declustered = np.zeros_like(arr1d)
    arr_uint = npct.ndpointer(dtype = np.uint16, ndim = 1, flags = 'C_CONTIGUOUS')
    # load the c extension
    libcd = npct.load_library("libclusters", utils.resource_path("../lib/",  PKG_NAME))

    libcd.searchFrame_array.restype = None
    libcd.searchFrame_array.argtypes = [arr_uint, arr_uint, c_int, c_int, c_int]

    dimx, dimy = np.shape(arr2d)
    for i in range(nloop):
        libcd.searchFrame_array(declustered, arr1d, dimx, dimy, threshold)
    return declustered.reshape(np.shape(arr2d))


def process_snapshot_path(path):
    i = Image.open(path)
    return process_snapshot_arr(np.asarray(i))

def process_snapshot_arr(arr):
    arr2d = np.sum(arr, axis=2)
    dec2 = decluster(arr2d, 50)
    #h = np.histogram(dec2, bins = 1000)[0][1:]
    return dec2

def filt_frame(imarr, lower = 350, upper = 450):
    imarr = imarr.copy()
    imarr[imarr < lower] = 0.
    imarr[imarr > upper] = 0.
    return imarr

def load_images(ser_name, dtype = u'uint16'):
    # handle gzip compression
    if ser_name[-3:] == '.gz':
        import gzip
        with gzip.open(ser_name, 'rb') as g:
            return np.fromstring(g.read(), dtype = dtype)
    else:
        return np.fromfile(ser_name, dtype = dtype)


def partition_frames(arr2d_from_ser, nframes = 10, dimx = 1080,
        dimy = 1920):
    arr = arr2d_from_ser[89:-40].reshape(dimx * nframes, dimy)
    return np.split(arr, nframes)

#from utils.utils import memoize
#@memoize()
def get_raw(path, dtype = np.uint16):
    return load_images(path, dtype = dtype)
    
def ser_extract(path, dimx = 1080, dimy = 1920, nframes = 100, offset = 400, dtype = np.uint16):
    raw = get_raw(path, dtype = dtype)
    excess = raw.shape[0] - dimx * dimy * nframes
    partitioned = np.array(np.split(raw[excess - offset:-offset].reshape(dimx * nframes, dimy), nframes))
    return partitioned
    #summed = reduce(add, partitioned[:1])

def extract_one(arr, threshold = 50, lower = 0, upper = 1000):
        declustered = decluster(arr.astype(np.uint16), threshold)
        return filt_frame(declustered, lower = lower, upper = upper)

def ser_decluster(path, threshold = 50, lower = 0, upper = 1000, **kwargs):
    def _extract_one(arr):
        return extract_one(arr, threshold = threshold, lower = lower, upper = upper)
        #declustered = decluster(arr.astype(np.uint16), threshold)
        #return filt_frame(declustered, lower = lower, upper = upper)
    partitioned = ser_extract(path, **kwargs)
    return np.array(list(map(_extract_one, partitioned)))

def upsample_and_sum(arr):
    new = np.zeros_like(arr[0]).astype(np.uint32)
    return reduce(add, arr, new)


def hist_oneframe(arr, threshold = 4, lower = 0, upper = 1000, decluster = True):
    if decluster:
        flat = extract_one(arr, threshold = threshold, lower = lower, upper = upper).ravel()
    else:
        flat = arr.ravel()
    #flat = declustered.ravel()
    y, x = np.histogram(flat[flat < 255], bins = 254, range = (0, 254))
    y[0] = 0
    return x[:-1], y


    
def full_process2(path, threshold=40, window_low = 0, window_high = 255, ylim = 10000,
                 export_path = None):
    arr = ser_extract(path, dimx = 1080, dimy = 1920, nframes=100, dtype = np.uint8, offset=800)
    
    histograms =  [hist_oneframe(a, threshold = threshold) for a in arr]
    histo_x, histo_y = histograms[0][0], np.sum([h[1] for h in histograms], axis = 0)
    
    if export_path is not None:
        np.savetxt(export_path, np.array([histo_x, histo_y]).T,
               header = 'ADC value\t number of counts')
    print (np.sum(histo_y[np.logical_and(histo_x > window_low,  histo_x < window_high)]))
    return histo_x, histo_y
    
