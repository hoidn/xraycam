from . import camcontrol
from . import utils
import numpy as np
from scipy.interpolate import UnivariateSpline
from xraycam.camcontrol import _rebin_spectrum
from scipy.ndimage.filters import gaussian_filter as gfilt
from xraycam.camcontrol import plt
import dill
from lmfit import models

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
    array = darkrun.get_frame().data
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
    
def fwhm_lineout(lineout,fwhm_smooth=2):
    """
    Given a lineout of [x's,y's] calculate fwhm of peak.
    """
    x, y = lineout
    y = gfilt(y,fwhm_smooth)
    spline = UnivariateSpline(x, y - np.max(y)/2, s = 0)
    r1, r2 = spline.roots()
    return r2 - r1

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

def makerange(w,s):
    return [[i,i+w] for i in np.arange(0,1936-w,s)]

class CalibPeakDrift:

    def __init__(self,dataruns,runoninit=True,**kwargs):
        self.dataruns = dataruns
        self.kwargs = kwargs

    def make_peakdrift_list(self,plot=False,show=False):
        peakdriftlist = []
        for dr in self.dataruns:
            peakdriftlist.append([dr.zrun._time_start,get_peaks(dr.get_lineout(**self.kwargs))[0,0]])
        peakdriftlist = np.transpose(peakdriftlist)
        times, peaks = peakdriftlist
        times = (times-times[0])/60
        peakdriftlist = np.array([times,peaks])
        if plot:
            plt.plot(*peakdriftlist,label='drift')
            if show:
                plt.show()
        self.peakdriftlist = peakdriftlist
        return peakdriftlist

    def exponential_fit(self,initialparams,show=True):
        import lmfit
        self.make_peakdrift_list(**self.kwargs)
        self.expmodel = lmfit.Model(exponential_func)
        self.expparams = self.expmodel.make_params()
        for param in self.expparams:
            if param in initialparams:
                self.expparams[param].set(value=initialparams[param])
        self.expfit = self.expmodel.fit(self.peakdriftlist[1],params=self.expparams,t=self.peakdriftlist[0])
        if show:
            plt.plot(*self.peakdriftlist,label='data')
            plt.plot(self.peakdriftlist[0],self.expfit.best_fit, label='fit')
            plt.title(' ; '.join([str(i)+':'+str(v) for i,v in self.expfit.best_values.items()]))
            plt.xlabel('time (min)')
            plt.ylabel('peak location (bin)')
            plt.show()


def exponential_func(t,tau,A,offset):
        return A*(1-np.exp(-t/tau))+offset

def find_files(prefixguess,directory='cache',postfix = '_array.npy'):
    import re, os
    import pandas as pd
    
    reg = re.compile(r'(.*'+prefixguess+r'.*)'+postfix)
    matches=[]
    for file in os.listdir(directory):
        m = reg.match(file)
        if m:
            if m.group(1) not in matches:
                matches.append(m.group(1))
    
    prefixes = []
    runsetreg = re.compile(r'(.*)_\d+')
    for match in matches:
        m = runsetreg.match(match)
        if m:
            mentry = {'prefix':m.group(1),'type':'runset'}
            if mentry not in prefixes:
                prefixes.append(mentry)
        else:
            mentry = {'prefix':match,'type':'single-run'}
            if mentry not in prefixes:
                prefixes.append(mentry)
                
    def count_entries(prefix,entries):
        i=0
        for e in entries:
            if prefix in e:
                i+=1
        return i
                
    for d in prefixes:
        d['numruns']=count_entries(d['prefix'],matches)
        
    return pd.DataFrame(prefixes)

def save_frame_object(frame,filename,directory='cache'):
    with open(directory+'/'+filename,'wb') as f:
        dill.dump(frame,f)

def load_frame_object(filename,directory='cache'):
    with open(directory+'/'+filename,'rb') as f:
        frame = dill.load(f)
    return frame

def process_final_frame(prefix,number_runs,norunparam=True,dashinfilename=True,**kwargs):
    finalrunset = camcontrol.RunSet()
    for i in range(number_runs):
        if dashinfilename:
            formatstr = '_{}'
        else:
            formatstr = '{}'
        finalrunset.insert(camcontrol.DataRun(run_prefix=prefix+formatstr.format(i),norunparam=norunparam,**kwargs))
    return finalrunset.get_total_frame()

def _take_lineout_erange(lineout,erange):
    energies, intensities = lineout
    indices = (energies > erange[0]) & (energies < erange[1])
    return np.array([energies[indices],intensities[indices]])

def _linear_background_subtraction(lineout,excluderegions,show=False):
    #remove exlusion regions
    bglineoutx,bglineouty = np.copy(lineout)
    for exclude in excluderegions:
        bgreg=(bglineoutx<exclude[0])|(bglineoutx>exclude[1])
        bglineoutx = bglineoutx[bgreg]
        bglineouty = bglineouty[bgreg]
    
    #linear fit and subtract
    lm = models.LinearModel()
    pars = lm.guess(bglineouty,x=bglineoutx)
    out = lm.fit(bglineouty,x=bglineoutx)
    lineoutx = lineout[0]
    lineouty = lineout[1]-out.eval(x=lineout[0])

    
    if show:
        #gather subtracted regions
        exclusions=[]
        for exclude in excluderegions:
            exclusions.append(_take_lineout_erange(lineout,exclude))
        plt.plot(*lineout,label='orig')
        for e in exclusions:
            plt.plot(*e,label='excluded-regions')
        plt.plot(bglineoutx,out.best_fit,label='linear-bg')
        plt.plot(lineoutx,lineouty,label='bg-sub')
        plt.figures[0].traces[-1]['visible']='legendonly'
        plt.show()
    
    return np.array([lineoutx,lineouty])

def explore_best_region(data,step=100,width=200, energy =(None,None), normalize = None):
    
    try:
        frame = data.get_frame()
    except AttributeError:
        frame = data
        assert type(frame) == camcontrol.Frame, "Input must be DataRun or Frame."
    
    #Plot for different ranges
    fwhmlist = []
    for r in [[i,i+width] for i in np.arange(0,frame.data.shape[0]-width,step)]:
        try:
            fwhm = fwhm_lineout(frame.get_lineout(yrange = r, 
                           normalize = normalize, energy = energy))
            lab = str(r)+'-fwhm {:.3f}'.format(fwhm)
        except (RuntimeError, ValueError):
            fwhm = 'CalcErr'
            lab = str(r)+'-fwhm '+fwhm
        fwhmlist.append([r,fwhm])
        frame.plot_lineout(yrange = r, label = lab, show = False, 
                           normalize = normalize, energy = energy)
    
    #Cleanup and label plot
    for tr in plt.figures[0].traces:
        tr['visible'] = 'legendonly'
    if energy == (None,None):
        plt.xlabel('Bin (px)')
    else:
        plt.xlabel('Energy (eV)')
    if normalize is None:
        plt.ylabel('Counts')
    else:
        plt.ylabel('Normalized Counts')
    plt.title('Subregion lineouts')    
    plt.show()
    
    #Plot Fwhm vs row if fwhm calculated successfully
    if 'CalcErr' not in [el[1] for el in fwhmlist]:
        fwhmplotlist = np.array([[np.mean(el[0]),el[1]] for el in fwhmlist])
        plt.plot(*np.transpose(fwhmplotlist), label = 'fwhm\'s')
        plt.xlabel('Center of Subregion')
        if energy == (None,None):
            plt.ylabel('fwhm (px)')
        else:
            plt.ylabel('fwhm (eV)')
        plt.title('Fwhm vs Subregion')
        plt.show()
        
    #Plot Center of mass vs row of frame
    cm = center_of_masses(frame.data)
    plt.plot(np.arange(len(cm)),cm,label='center-of-masses')
    plt.xlabel('Row of camera sensor')
    plt.ylabel('Center of mass of signal')
    plt.title('Center of mass vs. row of frame')
    plt.show()
    
                        
    

    