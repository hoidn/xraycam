from collections import namedtuple

# IP of the Beaglebone currently in use
BBB_IP = '172.28.188.208'
user = 'debian'
password = 'bbb'
host = user + '@' + BBB_IP

# Directory on the BBB in which to run data collection
base_path = '/home/debian/am335x_pru_package-master/pru_sw/example_apps/BBBcam/mt9001_driver/'

# Energy calibration from sensor two with gain 0x3f, taken 4/29/2016
Calibpoint = namedtuple('Calibpoint', ['energy', 'ADC'])
point1 = Calibpoint(5414., 127.5)
point2 = Calibpoint(6403., 150)
calib_slope = (point2.energy - point1.energy)/(point2.ADC - point1.ADC)
calib_intercept = point1.energy - point1.ADC * calib_slope
