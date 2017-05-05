sensorsettings = {}
datasettings = {}

def set_detector_settings(emissionline):
    if emissionline == 'skalpha':
        sensorsettings['threshold'] = 2
        sensorsettings['window_min'] = 120
        sensorsettings['window_max'] = 132
        sensorsettings['photon_value'] = 126
        datasettings['avg_energy'] = 2307
    if emissionline == 'pkalpha':
        sensorsettings['threshold'] = 2
        sensorsettings['window_min'] = 104
        sensorsettings['window_max'] = 114
        sensorsettings['photon_value'] = 110
        datasettings['avg_energy'] = 2014