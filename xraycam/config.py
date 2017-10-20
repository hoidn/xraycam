import os
from zwoasi import ASI_GAIN, ASI_EXPOSURE, ASI_HIGH_SPEED_MODE

#Directory for camera drivers
ZWO_LIB_DIRECTORY = '/home/xrays/zwodriver/lib/x64/libASICamera2.so'

#Log path
logfile_path = 'camcontrol.log'

saveconfig = {
	'Directory':os.path.expanduser('~/xraycam/examples/cache/')
}

# Config for camera settings on bootup
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


# Config for crystal
crystalconfig = {
    '2d':crystal2d['si111'],
    'order':1
}

def set_crystal_config(crystal='si111', braggorder=1):
    global crystalconfig
    crystalconfig = {'2d':crystal2d[crystal],'order':braggorder}

#From x-ray data booklet
crystal2d = {
    'si111':6.2712,
    'si220':3.8403117,
    'ge220':4.00
}


# Config for sensor and data energy scaling
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