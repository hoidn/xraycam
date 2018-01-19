#This is just a collection of scraps developed in Jupyter notebooks while running the instrument.
#The hope is to avoid rewriting bits of code because I can't figure out which nb they were in.

from . import camalysis
import numpy as np
from .camcontrol import plt

def plot_counts_vs_time(runset,show=True, label='counts vs. time', xrange=None):
    times=[]
    counts=[]
    for r in runset:
        times.append(datetime.datetime.fromtimestamp(r.zrun._time_start)-datetime.timedelta(hours=3))
        counts.append(r.counts_per_second(start=xrange[0],end=xrange[1]))
    plt.plot(times,counts,label=label)
    if show:
        plt.show()

def resample_array(arr,pxsize=100):
    pixels = 2.9 #microns
    factor = int(np.round(pxsize/pixels))
    print('resample will give pixels of size {:.2f} microns'.format(factor*pixels))
    row, cols = arr.shape
    grouped = arr[:row//factor*factor,:cols//factor*factor].reshape(row//factor, factor, cols//factor, factor)
    return grouped.sum(axis=3).sum(axis=1)    

def show(countdata,width=10, vmax=None, **kwargs):
    """Show the frame. Kwargs are passed through to plt.imshow."""
    import matplotlib.pyplot as mplt
    from matplotlib.colors import LogNorm
#     countdata = self.data/self.photon_value
    if vmax is None:
        vmax = np.max(countdata)*0.7
    fig, ax = mplt.subplots(figsize=(width,1936/1096*width))
    cax = ax.imshow(countdata,vmax=vmax,interpolation='none',norm=LogNorm(vmin=1, vmax=vmax))
    cbar = fig.colorbar(cax, 
        ticks=[int(x) for x in np.logspace(0.01,np.log10(vmax),num=8)],#np.insert(np.arange(0,int(vmax),vmax/10),0,1),
        format='$%d$',fraction=0.05, pad=0.04)
    mplt.show()
    
def show_resampled(data,pxsize = 100, **kwargs):
    show(resample_array(data,pxsize=pxsize),**kwargs)

def counts_bg_ratio(dr,signalregion=[400,600]):
    counts = dr.counts_per_second(start=signalregion[0],end=signalregion[1])
    bgleft = dr.counts_per_second(start=0,end=signalregion[0])
    bgright = dr.counts_per_second(start=signalregion[0],end=1935)
    return counts/(bgleft+bgright)

def counts_squred_bg_ratio(dr,signalregion=[400,600]):
    counts = dr.counts_per_second(start=signalregion[0],end=signalregion[1])
    bgleft = dr.counts_per_second(start=0,end=signalregion[0])
    bgright = dr.counts_per_second(start=signalregion[0],end=1935)
    return counts**2/(bgleft+bgright)

def queue_time_end(actionqueue):
    endtime = datetime.datetime.now()+datetime.timedelta(seconds=int(actionqueue.time_left_in_queue()))-datetime.timedelta(hours=3)
    return endtime.strftime('%H:%M:%S')

def shutdown_script(actionqueue,cameraalso=False):
    actionqueue.queue.append({'action':'disengage_high_voltage'})
#     actionqueue.queue.append({'action':'spellman_clear_setpoints'})
    if cameraalso:
        actionqueue.queue.append({'action':'xraycam_shutdown'})

def integralnorm(lineout):
    x, y = lineout
    return np.array([x,y/np.sum(y)])

from functools import reduce

from operator import add

import lmfit
def LalphaFit(lineout, plot=True, linbg=False, omitnans=False, weighted=False):
    mod = reduce(add, [lmfit.models.VoigtModel(prefix=s) for s in ('La1_','La2_')])
    if linbg:
        mod += lmfit.models.LinearModel(prefix='linbg_')
    pars = mod.make_params()
    
    pars['La1_gamma'].set(vary=True)
    pars['La2_gamma'].set(vary=True)
    peaksdict = dict(La1_gamma = 0.3,
                 La1_sigma = 0.3,
                 La2_gamma = 0.3,
                 La2_sigma = 0.3,
                 La1_center = 2424,
                 La2_center = 2420,
                 La1_amplitude = 350,
                 La2_amplitude = 35,
                )
    for k,v in peaksdict.items():
        pars[k].set(value=v)
        
    #Try adding constraints:
    for k in ('La1_gamma','La1_sigma','La2_gamma','La2_sigma'):
        pars[k].set(min=0)
    pars['La2_center'].set(max=2422)
        
    if linbg:
        pars['linbg_slope'].set(value=0)
        pars['linbg_intercept'].set(value=50)
    fit_kws={}
    if omitnans:
        fit_kws['nan_policy']='omit'
    if not weighted:
        fit = mod.fit(lineout[1],params=pars,x=lineout[0],fit_kws=fit_kws)
    else:
        fit = mod.fit(lineout[1],params=pars,x=lineout[0],fit_kws=fit_kws,weights=1/np.sqrt(lineout[1]))
    
    if plot:
        xs = lineout[0]
        comps = fit.eval_components(x=xs)
        plt.plot(*lineout,label='data')
        plt.plot(xs,fit.best_fit,label='fit')
        plt.figures[0].traces[0]['marker']=dict(size=5)
        plt.figures[0].traces[0]['mode']='markers'
        for k, v in comps.items():
            plt.plot(xs,v,label=k)
        plt.show()
        
    return fit

def quick_plot_fit(fit, show = True):
        xs = fit.userkws['x']
        lineout = np.array([xs,fit.data])
        comps = fit.eval_components(x=xs)
        plt.plot(*lineout,label='data')
        plt.plot(xs,fit.best_fit,label='fit')
        plt.figures[0].traces[0]['marker']=dict(size=5)
        plt.figures[0].traces[0]['mode']='markers'
        for k, v in comps.items():
            plt.plot(xs,v,label=k)
        if show:
        	plt.show()

def lalpha_fit_to_region(dr,params = None, region = [2410,2436], omitnans=False, plot=False, weighted=True):
    if params is None:
        params = dr.best_parameters
    lo = dr.get_lineout(**params)
    locrop = camalysis._take_lineout_erange(lo,region)
    fit = LalphaFit(locrop,linbg=True,omitnans=omitnans,plot=plot,weighted=weighted)
    print('Voigt FWHM:',voigt_fwhm(fit))
    return fit, voigt_fwhm(fit)

def voigt_fwhm(lmfitoutput,prefix = 'La1_'):
    """estimation from https://en.wikipedia.org/wiki/Voigt_profile"""
    fwhmL = lmfitoutput.best_values[prefix+'gamma']*2
    fwhmG = lmfitoutput.best_values[prefix+'sigma']*2*np.sqrt(2*np.log(2))
    phi = fwhmL/fwhmG
    c0, c1 = 2.0056, 1.0593
    return fwhmG*(1-c0*c1+np.sqrt(phi**2+2*c1*phi+c0**2*c1**2))

def pretty_matplotlib_kalpha_lcf(lcffit, plotxrange=[2009,2018], plotcomponents=['data','fit','components'], show=True, legend = True, title=None, interiorlabel = True, plotyrange = None):
    fit = lcffit.out
    xs = fit.userkws['x']
    lineout = np.array([lcffit.lineoutx,lcffit.lineouty])
    fxs,_ = camalysis._take_lineout_erange(lineout,lcffit.fitrange)

    comps = fit.eval_components(x=xs)
    
    if 'data' in plotcomponents:
        mplt.plot(*lineout,'o',markersize=3,label=lcffit.sample+' data')
    if 'fit' in plotcomponents:
        mplt.plot(xs,fit.eval(x=xs),'k',label=lcffit.sample+' fit')
    
    if 'components' in plotcomponents:
        for k, v in lcffit.complist.items():
            mplt.plot(*v,'--',label=lcffit.sample+' '+k)

    mplt.xlim(*plotxrange)
    if plotyrange is not None:
        mplt.ylim(*plotyrange)
    mplt.ylabel('Intensity')
    mplt.xlabel('Energy (eV)')
    ax=mplt.gca()
    ax.get_yaxis().set_ticks([])
    if title:
        ax.text(0.02,0.95,
                r'$\mathregular{'+lcffit.sample+r'}$'+' '+r'$\mathregular{K\alpha}$ XES',
               verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12)
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    if legend:
        mplt.legend(fontsize=9)
    if interiorlabel:
        lcffit.calc_contributions()
        ax.text(0.02,0.8,
        ('Reduced ({:.2f}%) '.format(100*lcffit.components['reduced'])+
        '\n  '
         r'$\mathregular{K\alpha_1}$:'+'{:.2f}'.format(lcffit.out.best_values['reduced_1_center'])+'\n\n'
         'Oxidized ({:.2f}%) '.format(100*lcffit.components['oxidized'])+
        '\n  '
         r'$\mathregular{K\alpha_1}$:'+'{:.2f}'.format(lcffit.out.best_values['oxidized_1_center'])
         ),
       verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=10)
    if show:
        mplt.show()

def check_fit_progression(runset, parameters, partialfit, of='reduced', plot=True):
    shift=[]
    for r in runset:
        try:
            fit = partialfit(r.get_lineout(**parameters))
            fit.do_fit(fit_kws={'nan_policy':'omit'})
            fit.calc_contributions()
            shift.append([datetime.datetime.fromtimestamp(r.zrun._time_start)-datetime.timedelta(hours=3), 
                          fit.components[of]])
        except ValueError:
            print('error, skipped run ',r.name)
    shift = np.array(shift)
    if plot:
        plt.plot(*shift.transpose(), label='peak pos vs. time')
        if show:
            plt.show()
    return shift

##### Below are tools developed in the efforts of analyzing the biochar data.
##### The functions allow for taking subgroups of a large runset, and fitting
##### to each subset.  Also tools for quickly parsing fit outputs for organizing
##### results over testing different fitting techiniques/parameters (e.g. peak ratios, etc.)

import datetime
def make_frame_list(rs,step = 5):
    ranges=[]
    for i in np.arange(0,len(rs.dataruns),step):
        ranges.append(np.arange(i,i+step))
    ranges[-1]=np.arange(i,len(rs.dataruns)) 
    
    frames=[]
    for r in ranges:
        fr = rs.get_total_frame(r)
        fr.best_parameters = rs.best_parameters
        fr.sample=rs.sample
        fr._time_start=datetime.datetime.fromtimestamp(rs.dataruns[r[0]]._time_start)-datetime.timedelta(hours=3)
        frames.append(fr)
    return frames

def fix_energies(name):
    fixedkws = {}
    fixedkws['oxidized_1_center']=alldf[alldf['name']==name]['oxid Ka1'].values[0]
    fixedkws['reduced_1_center']=alldf[alldf['name']==name]['red Ka1'].values[0]
    return fixedkws

def process_groupings(data, fixedkws=None, weighted=True, nan_policy=None, getstarttimes=True):
    fits = []
    for d in data:
        f = xfit.KalphaLinearCombinationFit(d.get_lineout(**d.best_parameters),sample=d.sample,
                                     refpeakshapes = znsrefpeakshape, runoninit=False,weighted=weighted,
                                    fitrange = (2302,2314))
        if fixedkws is not None:
            for k,v in fixedkws.items():
                f.pars[k].set(value=v,vary=False)
        if nan_policy is None:
            f.do_fit()
        elif nan_policy == 'omit':
            f.do_fit(fit_kws={'nan_policy':'omit'})
        if getstarttimes:
            f._time_start = d._time_start
        fits.append(f)
    df =  pd.DataFrame([parse_fit_output(x) for x in fits])
    print(df)
    return fits, df

def parse_fit_output(fit):
	#Might have issue with _time_start
    fit.calc_contributions()
    fitout = fit.out
    red, oxid = fit.components.values()
    oka1 = fit.out.best_values['oxidized_1_center']
    rka1 = fit.out.best_values['reduced_1_center']
    splitting = oka1-rka1
    redchi = fitout.redchi
    return pd.Series({
                     'reduced':red,
                     'oxidized':oxid,
                     'red Ka1':rka1,
                     'oxid Ka1':oka1,
                     'splitting':splitting,
                     'redchi':redchi,
                     'time_start':fit._time_start,
                     'name':fit.sample
        })
#                     name = fit.sample)

def process(data, weighted=True, nan_policy=None, getstarttimes=True):
    fits = []
    for d in data:
        f = xfit.KalphaLinearCombinationFit(d.get_lineout(**d.best_parameters),sample=d.sample,
                                     refpeakshapes = znsrefpeakshape, runoninit=False,weighted=weighted,
                                    fitrange = (2302,2314))
        if nan_policy is None:
            f.do_fit()
        elif nan_policy == 'omit':
            f.do_fit(fit_kws={'nan_policy':'omit'})
        if getstarttimes:
            f._time_start = d._time_start
        fits.append(f)
    df =  pd.DataFrame([parse_fit_output(x) for x in fits])
    print(df)
    return fits, df



### Matplotlib updating plots in Jupyter notebooks
import threading
import numpy as np
import time
import matplotlib
# matplotlib.use('nbagg') Import order is important here.
import matplotlib.pyplot as plt


def update_plot(fig, ax, line, get_data_func, stopevent, refreshrate=1):
    while not stopevent.is_set():
        x, y = get_data_func()
        line.set_xdata(x)
        line.set_ydata(y)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        time.sleep(1/refreshrate)

def make_and_update(get_data_func, stopevent, refreshrate=1):
    fig,ax = plt.subplots()
    x, y = get_data_func()
    lines, = ax.plot(x,y)
    plt.show()
    updatethread = threading.Thread(target=update_plot, args=(fig,ax,lines,dummy_get_data,stopupdates))
    updatethread.start()
    return updatethread