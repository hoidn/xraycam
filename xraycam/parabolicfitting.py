import numpy as np
from . import camcontrol
from .camalysis import get_peaks, center_of_masses
from scipy.ndimage.filters import gaussian_filter as gfilt


def integral_normalize(lineout):
    lineoutx, lineouty = lineout
    return np.array([lineoutx,lineouty/np.sum(lineouty)])

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

class ParabolicFit:
    
    def __init__(self,datarun,**kwargs):
        try:
            self.datarun = datarun.run
        except AttributeError:
            self.datarun = datarun
            
        self.raw_array = self.datarun.get_frame().data
        self.raw_lineout = self.datarun.get_frame().get_lineout(**kwargs)
        self.peaks = get_peaks(self.raw_lineout)
        #TODO: change fwhm_ev function to put out float instead of str
        # self.fwhm = float(fwhm_ev(self.raw_lineout))
        self.erange = self.raw_lineout[0][-1]-self.raw_lineout[0][0]
        self.fitparameters = quadfit(self.raw_array)
        self.fitregion = {'xrange':[0,-1], 'yrange':[0,-1]}
        self.set_fitregion()
        # self.update_quadfit()
        
    def cm_plot(self,smooth=5,show=True):
        xrange, yrange = self.fitregion['xrange'], self.fitregion['yrange']
        y = gfilt(center_of_masses(self.raw_array[yrange[0]:yrange[1],xrange[0]:xrange[1]]),smooth)
        x = np.arange(len(y))
        camcontrol.plt.plot(x+yrange[0],y+xrange[0],label="CM")
        if yrange == [0,-1]: 
            length = 1936 
        else: 
            length = yrange[1]-yrange[0]
        #length=1936
        
        camcontrol.plt.plot(*self.generate_parabola(),label="parab")
        if show:
            camcontrol.plt.show()
        return np.array([x,y])
            
    def calc_vertex(self):
        a, b, c = self.fitparameters
        vertexx = -b/(2*a)
        vertexy = a*vertexx**2+b*vertexx+c
        return (vertexx,vertexy)
    
    def generate_parabola(self,length=1936):
        a, b, c = self.fitparameters
        x = np.arange(length)
        y = a*x**2+b*x+c
        #if self.fitregion['xrange'] != [0,1]:
        #    y+=self.fitregion['xrange'][0]
        return np.array([x,y])
    
    def update_quadfit(self):
        xrange, yrange = self.fitregion['xrange'], self.fitregion['yrange']
        a, b, c = quadfit(self.raw_array[yrange[0]:yrange[1],xrange[0]:xrange[1]])
        #b = b-2*a*yrange[0]
        #c = a*yrange[0]**2-b*yrange[0]+c+xrange[0]
        vx = -b/(2*a)
        vy = (4*a*c-b**2)/(4*a)
        vx+=yrange[0]
        vy+=xrange[0]
        b = -2*a*vx
        c = vy+a*vx**2
        self.fitparameters = (a, b, c)
        
    def set_fitregion(self,width=0.2,region=None):
        if region:
            xrange=region
        else:
            xupper = np.searchsorted(self.raw_lineout[0],self.peaks[0,0]+self.erange*width/2)
            xlower = np.searchsorted(self.raw_lineout[0],self.peaks[0,0]-self.erange*width/2)
            self.fitregion['xrange'] = [xlower,xupper]
    
    def parabolic_sort(self,shape = (1936, 1096),xrange=[0,-1],yrange=[0,-1]):
        """
        Returns: z, (rowsort, colsort)

        z : 2d numpy array of shape `shape` with values x**2 + b * x + c,
        where x is row index.
        rowsort : sequence of row indices that sort z
        colsort : sequence of column indices that sort z
        """
        a, b, c = self.fitparameters
        x, y = np.indices(shape, dtype = 'int')
        z = -((a * (x**2)) + (b * x) - y)
        z = z[yrange[0]:yrange[1],xrange[0]:xrange[1]]
        return z, np.unravel_index(
                    np.argsort(z.ravel()), z.shape)
            
    def parabolic_lineout(self, energy=(None,None), nbins = None, xrange=[0,-1],yrange=[0,-1]):
        """Return lineout taken using parabolic bins"""
        arr2d = self.raw_array[yrange[0]:yrange[1],xrange[0]:xrange[1]]
        num_rows, num_cols = arr2d.shape
        if nbins is None:
            nbins = num_cols
        def chunks():
            """Return array values grouped into chunks, one per bin"""
            increment = int(num_rows * (num_cols/nbins))
            _, sort_indices = self.parabolic_sort(xrange=xrange,yrange=yrange)
            sort_data = arr2d[sort_indices].ravel()
            return [sort_data[i:i + increment] for i in range(0, len(sort_data), increment)]
        lineouty = np.array(list(map(np.sum, chunks())))/self.datarun.photon_value
        if energy == (None,None):
            lineoutx = np.arange(len(lineouty))
            lineout = np.array([lineoutx,lineouty])
        else:
            lineout = add_energy_scale(lineouty,energy[0],known_bin=energy[1],rebinparam=int(1096/nbins))
        return lineout
    
    def check_fit(self,xrange=[0,-1],yrange=[0,-1],vmax=500,size=8,parabolawidth=3):
        from scipy import stats
        import matplotlib.pyplot as mplt
        arr2d = self.raw_array[yrange[0]:yrange[1],xrange[0]:xrange[1]]
        z, _ = self.parabolic_sort(xrange=xrange,yrange=yrange)
        
        parr = np.sum([stats.threshold(z,threshmin=i,threshmax=i+parabolawidth,newval=0) for i in range(100,1000,40)],axis=0)
        parr[parr>0]=500
        mplt.figure(figsize=(size,1936/1096*size))
        mplt.imshow(parr+arr2d,vmax=vmax,interpolation='none')
        mplt.show()
        
    def compare_parabolic_lineout(self,energy=(None,None),show=True,**kwargs):
        camcontrol.plt.plot(*integral_normalize(self.raw_lineout),label='column')
        camcontrol.plt.plot(*integral_normalize(self.parabolic_lineout(yrange=self.fitregion['yrange'],energy=energy,**kwargs)),label='parabolic')
        if show:
            camcontrol.plt.show()
