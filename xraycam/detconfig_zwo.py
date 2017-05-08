sensorsettings = {}
datasettings = {}

def set_detector_settings(emissionline):
    sensorsettings.clear()
    datasettings.clear()
    if emissionline == 'skalpha':
        sensorsettings['threshold'] = 2
        sensorsettings['window_min'] = 120
        sensorsettings['window_max'] = 132
        datasettings['photon_value'] = 126
        datasettings['avg_energy'] = 2307
        datasettings['emissionline'] = emissionline
    if emissionline == 'pkalpha':
        sensorsettings['threshold'] = 2
        sensorsettings['window_min'] = 104
        sensorsettings['window_max'] = 114
        datasettings['photon_value'] = 110
        datasettings['avg_energy'] = 2014
        datasettings['emissionline'] = emissionline