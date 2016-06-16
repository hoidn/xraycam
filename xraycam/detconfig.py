from collections import namedtuple

# TODO: figure out a system for storing sensor-specific configuration parameters.
# It should be possible to switch between sensors by simply changing the value of
# sensor_id

# IP of the Beaglebone currently in use
BBB_IP = '172.28.188.208'
user = 'debian'
password = 'bbb'
host = user + '@' + BBB_IP

# Directory on the BBB in which to run data collection
base_path = '/home/debian/am335x_pru_package-master/pru_sw/example_apps/BBBcam/mt9001_driver/'

# Sensor-specific information
sensor_id = 0

# Energy calibration from sensor two with gain 0x3f, taken 4/29/2016
Calibpoint = namedtuple('Calibpoint', ['energy', 'ADC'])
point1 = Calibpoint(5414., 127.5)
point2 = Calibpoint(6403., 150)
calib_slope = (point2.energy - point1.energy)/(point2.ADC - point1.ADC)
calib_intercept = point1.energy - point1.ADC * calib_slope

darkrun_prefix_map = {0: 'data/6.16.dark4'}

