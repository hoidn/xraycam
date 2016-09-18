from . import detconfig
from . import camcontrol
from . import utils
import numpy as np
from scipy.interpolate import UnivariateSpline
from xraycam.camcontrol import _rebin_spectrum
from scipy.ndimage.filters import gaussian_filter as gfilt

@utils.memoize(timeout = None)
def get_hot_pixels(darkrun = None, threshold = 0):
    """
    darkrun : camcontrol.DataRun
        A dark run 
    threshold : int
        The threshold value for hot pixels Return tuple (x, y) of
        indices of pixels above threshold in the provided dark run. Reverts to
        the sensor-specific dark run `detconfig.sensor_id` if `darkrun == None`.
    """
    if darkrun is None:
        darkrun = camcontrol.DataRun(run_prefix = detconfig.darkrun_prefix_map[detconfig.sensor_id])
    array = darkrun.get_array()
    return np.where(array > threshold)

def fwhm(arr1d):
    """
    Given an array containing a peak, return its FWHM based on a spline interpolation.
    """
    x = np.arange(len(arr1d))
    spline = UnivariateSpline(x, arr1d - np.max(arr1d)/2, s = 0)
    r1, r2 = spline.roots()
    return r2 - r1

def calc_bragg_angle(energy,braggorder=1):
    """
    calculates bragg angle from energy given in eV.  Currently specific to si111 2d spacing.
    """
    si111_2dspacing=6.27118
    return 180*np.arcsin(12398.4*braggorder/(si111_2dspacing*energy))/np.pi
    
def energy_from_x_position(bragg,xpx,rebinparam=1,braggorder=1):
    """
    This function takes a bragg angle 'bragg', which is the bragg angle for a known
    energy on the camera, and takes an x position which is left (negative) or right (positive)
     of the central energy, in pixels 'xpx',
    and returns the energy of the x-ray which will be refocused to that position in the Rowland geometry.
    
    It is specific to Rowland diameter = 10cm, pixel size = 5.2 microns, and camera tangent to the circle.
    """
    pizel_size=2.9e-3 # NOTE: changed for new camera
    xpos=xpx*pizel_size*rebinparam
    return braggorder*1000*1.97705*np.sqrt(1+(xpos*np.cos(np.pi*bragg/90)+50*np.sin(np.pi*bragg/90))**2/(50-50*np.cos(np.pi*bragg/90)+xpos*np.sin(np.pi*bragg/90))**2)

def add_energy_scale(lineout,known_energy,known_bin=None,rebinparam=1,camerainvert=True,braggorder=1,**kwargs):
    """
    Returns an np array of [energies,lineout], by either applying a known energy to the max of the dataset, or to a specified bin.
    """
    if known_bin == None:
        centerindex=np.argmax(gfilt(lineout,3)) # if known_bin not provided, set energy to max of lineout
        # note to self, I was worried that gfilt might change the length of the list, but it doesn't.
    else:
        centerindex=round(known_bin/rebinparam) # else set energy to be at known bin position
    indexfromcenter=np.array(range(len(lineout)))-centerindex
    if camerainvert == True:
            indexfromcenter=-indexfromcenter # if camera gets flipped upside down, just reverse the indices
    return (energy_from_x_position(calc_bragg_angle(known_energy,braggorder),indexfromcenter,rebinparam,braggorder),lineout)
    
def fwhm_ev(arr2d,fwhm_smooth=2):
    """
    Given a 2d-array of [energies(eV),lineout], calculate fwhm of peak in the lineout.
    """
    x, y = arr2d
    y = gfilt(y,fwhm_smooth)
    spline = UnivariateSpline(x, y - np.max(y)/2, s = 0)
    r1, r2 = spline.roots()
    return format(r2 - r1, '.3f')
    
def plot_with_energy_scale(datarun,known_energy,yrange=[0,-1],xrange=[0,-1],rebin=1,show=True,peaknormalize=False, label=None,calcfwhm=False,**kwargs):
    lineout = np.sum(datarun.get_array()[yrange[0]:yrange[1],xrange[0]:xrange[1]],axis=1)/datarun.photon_value
    if rebin != 1: #rebin using oliver's rebin_spectrum function
        lineout = _rebin_spectrum(np.array(range(len(lineout))),lineout,rebin)[1]
    if peaknormalize == True:
        lineout = lineout / max(lineout)
    lineout_energyscale=add_energy_scale(lineout,known_energy,rebinparam=rebin,**kwargs)
    if label == None and calcfwhm == False:
        label=datarun.prefix
    elif label == None and calcfwhm == True:
        s=' - '
        label=s.join((str(datarun.prefix),str(fwhm_ev(lineout_energyscale,3))))
    elif label != None and calcfwhm == True:
        s=' - '
        label=s.join((label,str(fwhm_ev(lineout_energyscale))))
    camcontrol.plt.plot(*lineout_energyscale,label=label)
    if show == True:
        camcontrol.plt.show()
        
def fwhm_datarun(datarun,known_energy,yrange=[0,-1],xrange=[0,-1],rebin=1,fwhm_smooth=2,**kwargs):
    """
    Given a 2d-array of [energies(eV),lineout], calculate fwhm of peak in the lineout.
    """
    lineout = np.sum(datarun.get_array()[yrange[0]:yrange[1],xrange[0]:xrange[1]],axis=1)/datarun.photon_value
    if rebin != 1: #rebin using oliver's rebin_spectrum function
        lineout = _rebin_spectrum(np.array(range(len(lineout))),lineout,rebin)[1]
    lineout_energyscale=add_energy_scale(lineout,known_energy,rebinparam=rebin,**kwargs)
    x, y = lineout_energyscale
    y = gfilt(y,fwhm_smooth)
    spline = UnivariateSpline(x, y - np.max(y)/2, s = 0)
    r1, r2 = spline.roots()
    return format(r2 - r1, '.3f')

def focus_ZvsFWHM_plot(dataruntuple,known_energy,**kwargs):
    camcontrol.plt.plot(*list(zip(*[(x.run.z,fwhm_datarun(x.run,known_energy,**kwargs)) for x in dataruntuple])),label='fwhm v z')
    camcontrol.plt.show()
