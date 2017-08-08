from zwoasi import ASI_GAIN, ASI_EXPOSURE, ASI_HIGH_SPEED_MODE

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