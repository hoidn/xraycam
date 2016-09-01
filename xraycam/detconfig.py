from collections import namedtuple

# TODO: figure out a system for storing sensor-specific configuration
# parameters. It should be possible to switch between sensors by simply
# changing the value of sensor_id

# Default camera; either 'zwo' or 'beaglebone'
detector = 'beaglebone'

# ____  ____  ____                      
#| __ )| __ )| __ )  ___ __ _ _ __ ___  
#|  _ \|  _ \|  _ \ / __/ _` | '_ ` _ \ 
#| |_) | |_) | |_) | (_| (_| | | | | | |
#|____/|____/|____/ \___\__,_|_| |_| |_|
#
# IP of the Beaglebone currently in use
user = 'debian'
password = 'bbb'
BBB_IP = '10.155.95.216'
host = user + '@' + BBB_IP

# Directory on the BBB in which to run data collection
base_path = '/home/debian/am335x_pru_package-master/pru_sw/example_apps/BBBcam/mt9001_driver/'

# Current Beaglebone camera sensor ID. All information below it is
# sensor-specific.
sensor_id = 0

# Energy calibration from sensor two with gain 0x3f, taken 4/29/2016
Calibpoint = namedtuple('Calibpoint', ['energy', 'ADC'])
point1 = Calibpoint(5414., 127.5)
point2 = Calibpoint(6403., 150)

# Map Beaglebone camera sensor IDs to dark run prefixes.
darkrun_prefix_map = {0: 'data/6.16.dark4'}

