import time
from threading import Thread



class mc:
    def __init__(self):
        def timer():
            print("in timer")
            time.sleep(3)
            print("exiting timer")
        mt = Thread(target = timer, args = ())
        mt.start()
