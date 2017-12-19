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
# 2d spacings from x-ray data booklet
crystal2d = {
    'si111':6.2712,
    'si220':3.8403117,
    'ge220':4.00
}

crystalconfig = {
    '2d':crystal2d['si111'],
    'order':1
}

def set_crystal_config(crystal='si111', braggorder=1):
    global crystalconfig
    crystalconfig = {'2d':crystal2d[crystal],'order':braggorder}


# Config for sensor and data energy scaling
sensorsettings = {}
datasettings = {}

detectordictionary = {
    'skalpha':{
        'sensorsettings':{
            'threshold':0,
            'window_min':120,
            'window_max':132
        },
        'detectorsettings':{
            'photon_value':126,
            'avg_energy' = 2307
        }
    }
    'pkalpha':{
        'sensorsettings':{
            'threshold':0,
            'window_min':104,
            'window_max':114
        },
        'detectorsettings':{
            'photon_value':110,
            'avg_energy' = 2014
        }
    }
}

def set_detector_settings(emissionline):
    '''Set based on tests with gain:213'''
    sensorsettings.clear()
    datasettings.clear()
    sensorsettings = detectordictionary[emissionline]['sensorsettings']
    datasettings = detectordictionary[emissionline]['detectorsettings']
    # if emissionline == 'skalpha':
    #     sensorsettings['threshold'] = 0
    #     sensorsettings['window_min'] = 120
    #     sensorsettings['window_max'] = 132
    #     datasettings['photon_value'] = 126
    #     datasettings['avg_energy'] = 2307
    #     datasettings['emissionline'] = emissionline
    # if emissionline == 'pkalpha':
    #     sensorsettings['threshold'] = 0
    #     sensorsettings['window_min'] = 104
    #     sensorsettings['window_max'] = 114
    #     datasettings['photon_value'] = 110
    #     datasettings['avg_energy'] = 2014
    #     datasettings['emissionline'] = emissionline
    # if emissionline == 'skbeta':
    #     sensorsettings['threshold']=0
    #     sensorsettings['window_min']=130
    #     sensorsettings['window_max']=142
    #     datasettings['photon_value']=136
    #     datasettings['avg_energy']=2464
    #     datasettings['emissionline'] = emissionline
    # if emissionline == 'tclalpha':
    #     sensorsettings['threshold']=0
    #     sensorsettings['window_min']=129
    #     sensorsettings['window_max']=137
    #     datasettings['photon_value']=133
    #     datasettings['avg_energy']=2424
    #     datasettings['emissionline'] = emissionline