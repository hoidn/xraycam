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

def add_energy_scale(lineout,known_energy,known_bin='peak',rebinparam=1,camerainvert=True,braggorder=1,**kwargs):
    """
    Returns an np array of [energies,lineout], by either applying a known energy to the max of the dataset, or to a specified bin.
    """
    if known_bin == None:
        centerindex=np.argmax(gfilt(lineout,3)) # if known_bin not provided, set energy to max of lineout
        # note to self, I was worried that gfilt might change the length of the list, but it doesn't.
    elif known_bin == 'peak':
        centerindex = get_peaks(np.array([list(range(len(lineout))),lineout]))[0]
    else:
        #centerindex=round(known_bin/rebinparam) # else set energy to be at known bin position
        centerindex=known_bin/rebinparam #try without rounding 2.9.17
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

# def plot_with_energy_scale(datarun,known_energy,yrange=[0,-1],xrange=[0,-1],rebin=1,show=True,peaknormalize=False, label=None,calcfwhm=False,parabolic=False,**kwargs):
#     if parabolic == False:
#         lineout = np.sum(_reorient_array(datarun.get_array())[yrange[0]:yrange[1],xrange[0]:xrange[1]],axis=0)/datarun.photon_value
#     else:
#         lineout = get_parabolic_lineout(_reorient_array(datarun.get_array()),yrange=yrange)[xrange[0]:xrange[1]] 
#     if rebin != 1: #rebin using oliver's rebin_spectrum function
#         lineout = _rebin_spectrum(np.array(range(len(lineout))),lineout,rebin)[1]
#     if peaknormalize:
#         lineout = lineout / max(lineout)
#     lineout_energyscale=add_energy_scale(lineout,known_energy,rebinparam=rebin,**kwargs)
#     if label == None and calcfwhm == False:
#         label=datarun.prefix
#     elif label == None and calcfwhm == True:
#         s=' - '
#         label=s.join((str(datarun.prefix),str(fwhm_ev(lineout_energyscale,3))))
#     elif label != None and calcfwhm == True:
#         s=' - '
#         label=s.join((label,str(fwhm_ev(lineout_energyscale))))
#     camcontrol.plt.plot(*lineout_energyscale,label=label)
#     if show == True:
#         camcontrol.plt.show()

def focus_ZvsFWHM_plot(dataruntuple,known_energy,**kwargs):
    camcontrol.plt.plot(*list(zip(*[(x.run.z,fwhm_datarun(x.run,known_energy,**kwargs)) for x in dataruntuple])),label='fwhm v z')
    camcontrol.plt.show()

def center_of_masses(arr2d):
    def _cm(arr1d):
        return np.dot(arr1d, np.arange(len(arr1d)))/np.sum(arr1d)
    return np.array(list(map(_cm, arr2d)))

def cmplot(datarun, smooth=0,show=True):
    arr2d = np.rot90(datarun.get_array()) # trying rot90 instead of transpose

    y = center_of_masses(arr2d)
    x = np.arange(len(y))
    if smooth != 0:
        y = gfilt(y,smooth)
    camcontrol.plt.plot(x, y, label = 'CM lineout')
    if show == True:
        camcontrol.plt.show()

    return np.array([x,y])

def fwhm_vs_row_plot(datarun,step=100,**kwargs):
    camcontrol.plt.plot(*list(zip(*[(i+step/2,fwhm_ev(datarun.get_frame().get_lineout(yrange=[i,i+step],**kwargs))) for i in range(0,2000,step)])),label='fwhm v row')
    camcontrol.plt.show()

# def fwhm_vs_row_plot(datarun,step=100):
#     camcontrol.plt.plot(*list(zip(*[(i+step/2,fwhm_datarun(datarun.run,2300,xrange=[i,i+step],rebin=2)) for i in range(0,2000,step)])),label='fwhm v row')
#     camcontrol.plt.show()
    
# def focus_ThetavsFWHM_plot(dataruntuple,known_energy,**kwargs):
#     camcontrol.plt.plot(*list(zip(*[(x.run.theta,fwhm_datarun(x.run,known_energy,**kwargs)) for x in 
#                      dataruntuple])),label='fwhm v z')
#     camcontrol.plt.show()

def cropping_tool(datarun,step,known_energy=2014,calcfwhm=True,**kwargs):
    [plot_with_energy_scale(datarun,known_energy,label='['+','.join((str(i),str(i+step)))+']',yrange=[i,i+step],
                            show=False,calcfwhm=calcfwhm,peaknormalize=True,**kwargs) for i in range(0,2000,step)]
    camcontrol.plt.show()
# Below are functions which support the parabolic fitting.
def _reorient_array(arr2d):
    """Take output from the get_array() method for dataruns from the new camera,
    and reorient them to match what our usual analysis code expects."""
    return np.transpose(arr2d[::,::-1])


def parabolic_sort(a, b, shape = (1024, 1280)):
    """
    Returns: z, (rowsort, colsort)
    
    z : 2d numpy array of shape `shape` with values x**2 + b * x + c,
    where x is row index.
    rowsort : sequence of row indices that sort z
    colsort : sequence of column indices that sort z
    """
    x, y = np.indices(shape, dtype = 'int')
    z = ((a * (x**2)) + (b * x) + y)
    return z, np.unravel_index(
                np.argsort(z.ravel()), z.shape)

def quadfit(arr2d, smooth = 5):
    """
    Return the second- and first-order coefficients for a parabolic
    fit to array of center of mass values of the rows of arr2d.
    """
    y = gfilt(center_of_masses(arr2d), smooth)# - np.percentile(filtered, 1)) *Note: changed gfilt to act on y instead of on 2d array.  Seems to produce better parabolas.
    x = np.arange(len(y))
    good = np.where(np.isfinite(y))[0] #note to self: is there assumption here that cutting out non-finite elements won't appreciably change the curvature?
    a, b, c, = np.polyfit(x[good], y[good], 2)
    # For some reason a factor of -1 is needed
    return a, b, c

def get_parabolic_lineout(arr2d, nbins = None, fitregionmode = 'cm' , fitregionx = [0,-1], fitregiony = [0,-1],yrange=[0,-1],**kwargs):
    """Return lineout taken using parabolic bins"""
    # Fit only to specific region
    if fitregionmode != 'cm':
        a, b, _ = quadfit(arr2d[fitregiony[0]:fitregiony[1],fitregionx[0]:fitregionx[1]])
    else: 
        # Fit around center of mass to get better parabolas
        cm = np.mean(center_of_masses(arr2d))
        a, b, _ = quadfit(arr2d[fitregiony[0]:fitregiony[1],int(cm-150):int(cm+150)])
    if yrange != [0,-1]: 
        # crop the region in the lineout, but with parabolic parameters from the (possibly) different fitregion
        arr2d = arr2d[yrange[0]:yrange[1],:]
    num_rows, num_cols = arr2d.shape
    if nbins is None:
        nbins = num_cols
    def chunks():
        """Return array values grouped into chunks, one per bin"""
        increment = int(num_rows * (num_cols/nbins))
        _, sort_indices = parabolic_sort(a, b, arr2d.shape)
        sort_data = arr2d[sort_indices].ravel()
        return [sort_data[i:i + increment] for i in range(0, len(sort_data), increment)]
    return np.array(list(map(np.sum, chunks())))

def get_peaks(lineout,interp=True,**kwargs):
    """Get location of peaks using PeakUtils package.
    Format should be lineout=[energies,intensities].
    Returns [peaks_x,peaks_y].
    Peaks_x location is optionally improved using interpolation.
    Interpolation is hard-coded for Gaussian.  Can be modified for centroid, others."""
    
    #Import peak-detection package if not already loaded.
    import peakutils
    
    #Get thres and min_dist for peak-detection, set default if not provided
    thres = kwargs.get('thres',0.8)
    min_dist = kwargs.get('min_dist',30)
    width = kwargs.get('width',10)
    
    #Find peak indices
    x, y = lineout
    indexes = peakutils.indexes(y,thres=thres,min_dist=min_dist)
    peaks_x, peaks_y = lineout[0][indexes], lineout[1][indexes]
    
    #Improve peak location by fitting function locally around detected peaks in the data.  Default is gaussian.
    if interp:
        peaks_x=[]
        peaks_y=[]
        for i in indexes:
            a,b,c = peakutils.peak.gaussian_fit(x[int(i-width):int(i+width)],y[int(i-width):int(i+width)],center_only=False)
            peaks_x.append(b)
            peaks_y.append(a)

    return np.array([peaks_x,peaks_y])

def anglecounts(runlist):
    thetalist = [x.theta for x in runlist]
    countlist = [x.counts_per_second() for x in runlist]
    from xraycam.camcontrol import plt
    plt.plot(*[[thetalist[i] for i in np.argsort(thetalist)],[countlist[i] for i in np.argsort(thetalist)]], label='counts vs angle')
    plt.xlabel('theta (deg)')
    plt.ylabel('counts/sec')
    plt.show()

# def peaklocation_vs_theta(datarunlist,dosort=True,show=True,usetheta=True,useenergy=True,interp=True):
#     if usetheta:
#         thetalist = [x.run.theta for x in datarunlist]
#     else:
#         thetalist = np.arange(len(datarunlist))
#     if useenergy:
#         energy = (2307,400)
#     else:
#         energy = (None, None)
#     peaklist = [get_peaks(x.run.get_frame().get_lineout(energy=energy),interp=interp)[0,0] for x in datarunlist]
#     xylist = np.array([thetalist,peaklist])
#     if dosort:
#         xylist = xylist.T[np.argsort(thetalist)].T
#     plt.plot(*xylist,label='')
#     if usetheta:
#         plt.xlabel('theta (deg)')
#     else:
#         plt.xlabel('run order')
#     plt.ylabel('peaklocation (eV)')
#     if show:
#         plt.show()
#     return xylist