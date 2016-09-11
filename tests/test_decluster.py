from xraycam.zwo import do_decluster as decluster
import numpy as np

def test_decluster1():
    arr = np.zeros((10, 20), dtype = np.uint8)
    arr[1,1] = 1.
    arr[1,2] = 3.

    arr = decluster(arr, 0)
    
    assert np.sum(arr) == 4
    return arr

def test_decluster2():
    arr = np.zeros((10, 20), dtype = np.uint8)
    arr[2,:] = 1.
    arr[3,:] = 2.
    arr[4,:] = 3.

    arr = decluster(arr, 2)
    
    assert np.sum(arr) == 60
    return arr

def test_decluster3():
    arr = np.zeros((10, 20), dtype = np.uint8)
    arr[4,:] = 3.
    arr[:,4] = 3.

    arr = decluster(arr, 2)
    
    assert np.sum(arr) == 87
    return arr

def test_decluster_cm():
    arr = np.ones((5, 5), dtype = np.uint8)
    arr = decluster(arr, 0)
    
    assert np.sum(arr) == 25
    assert arr[2, 2] == 25

    arr2 = np.ones((5, 5), dtype = np.uint8)
    arr2[0, 0] = 26
    arr2 = decluster(arr2, 0)
    assert np.sum(arr2) == 50
    assert arr2[1, 1] == 50
    return arr, arr2

def test_decluster_5():
    dimx, dimy = 1079, 1919
    arr = np.ones((dimx, dimy), dtype = np.uint8)
    #arr = np.ones((1080, 1920), dtype = np.uint8)
    arr = decluster(arr, 0)
    
    assert np.sum(arr) == (dimx * dimy)
    assert arr[(dimx - 1)/2, (dimy - 1)/2] == dimx * dimy

    return arr 

if __name__ == '__main__':
    test_decluster_5()
