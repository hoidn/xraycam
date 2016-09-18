import async

class SlowCount:
    """
    Class that outputs a list range, slowly.
    """
    def __init__(self, n, sleep = 1):
        self.high = n
        self.cur = 0
        self.sleep = sleep

    def __iter__(self):
        return self

    # Python 2 compatibility
    def next(self):
        return self.__next__()

    def __next__(self):
        import time
        if self.cur < self.high:
            time.sleep(self.sleep)
            self.cur += 1
            return self.cur - 1
        else:
            raise StopIteration

def test_iterthread():
    import time
    it = async.IterThread(SlowCount(5))
    start = time.time()
    time.sleep(1.5) # Wait for the first element to evaluate
    assert sum(it) == 0
    assert time.time() - start < 2 # Shouldn't have blocked
    time.sleep(4)
    assert sum(it) == 10 # Should be complete now
