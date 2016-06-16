from collections import namedtuple

# IP of the Beaglebone currently in use
BBB_IP = '172.28.188.208'

# Directory on the BBB in which to run data collection
base_path = '/home/debian/am335x_pru_package-master/pru_sw/example_apps/BBBcam/mt9001_driver/'

#darknames =\
#    ['data/autorun3.31_0',
#     'data/autorun3.31_1',
#     'data/autorun3.31_2',
#     'data/autorun3.31_3',
#     'data/autorun3.31_4',
#     'data/autorun3.31_5',
#     'data/autorun3.31_6',
#     'data/autorun3.31_7',
#     'data/autorun3.31_8', 
#     'data/autorun3.31_9',
#     'data/autorun3.31_10',
#     'data/autorun3.31_11',
#     'data/autorun3.31_12',
#     'data/autorun3.31_13',
#     'data/autorun3.31_14',
#     'data/autorun3.31_15',
#     'data/autorun3.31_16',
#     'data/autorun3.31_17',
#     'data/autorun3.31_18',
#     'data/autorun3.31_19',
#     'data/autorun3.31_20']

# Energy calibration from sensor two with gain 0x3f, taken 4/29/2016
Calibpoint = namedtuple('Calibpoint', ['energy', 'ADC'])
point1 = Calibpoint(5414., 127.5)
point2 = Calibpoint(6403., 150)
calib_slope = (point2.energy - point1.energy)/(point2.ADC - point1.ADC)
calib_intercept = point1.energy - point1.ADC * calib_slope
