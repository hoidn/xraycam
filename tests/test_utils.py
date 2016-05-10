from xraycam import utils
from xraycam import camcontrol
import numpy as np

def test_conserve_type():
    @utils.conserve_type
    def foo(lst, arr):
        return list(lst), list(arr)

    foo_result = foo(list(range(10)), np.array(list(range(10))))
    assert foo_result[0] == list(range(10))
    assert np.all(foo_result[1] == np.array(list(range(10))))



