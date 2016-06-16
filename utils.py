import numpy as np
import pdb

def conserve_type(func):
    """
    Decorator: for a function whose positional arguments are all either
    lists or np.ndarrays and returns the same number of iterables, convert
    each returned object to the same type (i.e. list or np.ndarray) as the
    corresponding input.
    """
    import collections
    def newfunc(*args, **kwargs):
        def eval_if_iterator(obj):
            if isinstance(obj, collections.Iterator):
                return list(obj)
            else:
                return obj

        def convert(arg, output):
            arg = eval_if_iterator(arg)
            if type(arg) == list:
                return list(output)
            elif type(arg) == np.ndarray:
                return np.array(output)
            elif type(arg) == tuple:
                return tuple(output)
            else:
                return output

        result = list(func(*args, **kwargs))
        converted = list(map(convert, args, result))
        return convert(result, converted)
    return newfunc
