# TODO: turn all of this into explicit tests

from xraycam import camcontrol
from test_zmq_comm import *
from multiprocess import Process
Process(target = start_ventilator, args = ()).start()
camcontrol.detconfig.detector = 'zwo'

rs = camcontrol.RunSequence(prefix = 'foobar9', htime = '5s', number_runs = 5,
    decluster = False)

dr = rs.funcalls[0]()

l = list(rs)

[ll.name for ll in l]

ds = camcontrol.RunSet(prefix = 'foobar15', htime = '20s', number_runs = 2,
    decluster = False)

list(ds.dataruns)
ds.dataruns[2].get_array()
