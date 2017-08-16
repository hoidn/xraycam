import os
from zwoasi import ASI_GAIN, ASI_EXPOSURE, ASI_HIGH_SPEED_MODE

#Directory for camera drivers
ZWO_LIB_DIRECTORY = '/home/xrays/zwodriver/lib/x64/libASICamera2.so'

#Log path
logfile_path = 'camcontrol.log'

saveconfig = {
	'Directory':os.path.expanduser('~/xraycam/examples/cache/')
}

cameraconfig = {
    'Gain':{
        'controlname':ASI_GAIN,
        'set_value':213
    },
    'Exposure':{
        'controlname':ASI_EXPOSURE,
        'set_value':50000
    },
    'HighSpeedMode':{
        'controlname':ASI_HIGH_SPEED_MODE,
        'set_value':0
    },
    'ImageType':'RAW8'
}

sensorsettings = {}
datasettings = {}

def set_detector_settings(emissionline):
    sensorsettings.clear()
    datasettings.clear()
    if emissionline == 'skalpha':
        sensorsettings['threshold'] = 0
        sensorsettings['window_min'] = 120
        sensorsettings['window_max'] = 132
        datasettings['photon_value'] = 126
        datasettings['avg_energy'] = 2307
        datasettings['emissionline'] = emissionline
    if emissionline == 'pkalpha':
        sensorsettings['threshold'] = 0
        sensorsettings['window_min'] = 104
        sensorsettings['window_max'] = 114
        datasettings['photon_value'] = 110
        datasettings['avg_energy'] = 2014
        datasettings['emissionline'] = emissionline