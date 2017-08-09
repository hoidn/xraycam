from lmfit import minimize, Parameters, models
from plotly import tools
import plotly.offline as offline
import plotly.graph_objs as go
import numpy as np
from xraycam.camcontrol import plt
from xraycam.camalysis import _take_lineout_erange

def norm(y,mode='peak'):
    if mode=='peak':
        res = y/max(y)
    elif mode == 'integral':
        res = y/np.sum(y)
    return res

def fit_resid_plotly(lmfitoutput,xvalues,xrange=None,poisson=False,comptraces=[],complabels=[],save=False,joined=False):
    data=[]

    if poisson:
        poissontrace1=go.Scatter(x=xvalues,y=np.sqrt(lmfitoutput.data),xaxis='x',yaxis='y',
                                 name='poisson+',mode='lines',fill='tozeroy',line=dict(
                                color='rgba(31,120,180,.5)'),fillcolor='rgba(31,120,180,0.25)')
        poissontrace2=go.Scatter(x=xvalues,y=-np.sqrt(lmfitoutput.data),xaxis='x',yaxis='y',
                                 name='poisson-',mode='lines',fill='tozeroy',line=dict(
                                color='rgba(31,120,180,.5)'),fillcolor='rgba(31,120,180,0.25)')
        data.append(poissontrace2)
        data.append(poissontrace1)

    if poisson == 2:
        poissontrace3=go.Scatter(x=xvalues,y=2*np.sqrt(lmfitoutput.data),xaxis='x',yaxis='y',
                                 name='poisson2+',mode='lines',fill='tonexty',line=dict(
                                color='rgba(227,26,28,.5)'),fillcolor='rgba(227,26,28,0.25)')
        poissontrace4=go.Scatter(x=xvalues,y=-2*np.sqrt(lmfitoutput.data),xaxis='x',yaxis='y',
                                 name='poisson2-',mode='lines',fill='tonexty',line=dict(
                                color='rgba(227,26,28,.5)'),fillcolor='rgba(227,26,28,0.25)')
        data.insert(1,poissontrace4)
        data.append(poissontrace3)
    
    residtrace = go.Scatter(x=lmfitoutput.data,y=lmfitoutput.residual,xaxis='x',yaxis='y',mode='lines+markers',name='residuals',
                           marker=dict(size=5),line=dict(width=1,
                                color='rgb(49,54,149)'))
    fittrace = go.Scatter(x=xvalues,y=lmfitoutput.out.eval(x=xvalues),xaxis='x',yaxis='y2',name='fit',
                         line=dict(color='rgba(0,0,0,0.7)'))

    if joined:
        datamode = 'lines+markers'
    else:
        datamode = 'markers'
    datatrace = go.Scatter(x=xvalues,y=lmfitoutput.data,xaxis='x',yaxis='y2',name='data',mode=datamode,
                          marker = dict(size=5,color='rgba(200,0,0,0.8)'))

    for trace in[residtrace,datatrace,fittrace]:
        data.append(trace)
    
    if comptraces is not []:
        i=0
        if not complabels:
            complabels=['comp%d'% j for j in range(len(comptraces))]
        #for tr in reversed(comptraces):
        for tr in comptraces:
            data.append(go.Scatter(x=tr[0],y=tr[1],xaxis='x',yaxis='y2',name=complabels[i]))#,visible='legendonly'))
            i+=1
    
    layout=go.Layout(
        xaxis=dict(domain=[0,1],anchor='y2',title='Energy(ev)',range=xrange),
        yaxis=dict(domain=[0.7,1],title='residuals'),
        yaxis2=dict(domain=[0,0.68],title='intensity'),
        height=600)

    if poisson == 1:
        data=data[::-1]

    fig=go.Figure(data=data,layout=layout)
    if save:
        imagestr='svg'
    else:
        imagestr=None
    offline.iplot(fig,image=imagestr)

def splitting(lmfitout,v1str,v2str):
    v1center = lmfitout.best_values[v1str+'_center']
    v2center = lmfitout.best_values[v2str+'_center']
    return v2center-v1center

pkalpha2peakfitprofile = dict(v1_gamma=0.3,v1_center=2012.7,
    v2_center=2013.5,bg_intercept=-2013)

skalpha2peakfitprofile = dict(v1_gamma=0.3,v1_center=2306.9,
    v2_center=2307.7,bg_intercept=-2307)

class do_peak_fit:
    
    def __init__(self,lineout, numpeaks=2,sample='sample',initialprofile=pkalpha2peakfitprofile,runoninit=True):
        self.sample=sample
        self.lineoutx=lineout[0]
        self.lineouty=lineout[1]
        self.voigt1 = models.VoigtModel(prefix='v1_')
        self.voigt2 = models.VoigtModel(prefix='v2_')
        #if numpeaks == 4:
        #    self.voigt3 = models.VoigtModel(prefix='v3_')
        #    self.voigt4 = models.VoigtModel(prefix='v4_')
        self.linbg = models.LinearModel(prefix='bg_')
        #if numpeaks == 2:
        self.model = self.voigt1+self.voigt2+self.linbg
        #elif numpeaks == 4:
        #    self.model = self.voigt1+self.voigt2+self.voigt3+self.voigt4+self.linbg
        self.pars=self.voigt1.make_params()
        self.pars.update(self.voigt2.make_params())
        self.pars.update(self.linbg.make_params())
        self.initialprofile=initialprofile
        self.initialize_pars()
        self.complist=[]
        if runoninit:
            self.run_fit()
            self.print_summary()
    
    def reset_pars(self):
        self.pars=self.voigt1.make_params()
        self.pars.update(self.voigt2.make_params())
        self.pars.update(self.linbg.make_params())
    
    def initialize_pars(self):
        [self.pars[x].set(value=0.22) for x in [('v%d_sigma')%i for i in (1,2)]]
        #self.pars['v1_amplitude'].set(expr='v2_amplitude*0.5')
        self.pars['v2_sigma'].set(expr='v1_sigma')
        self.pars['v1_gamma'].set(vary=True)
        self.pars['v2_gamma'].set(expr='v1_gamma')
        #self.pars['v1_gamma'].set(value=0.3)
        #self.pars['v1_center'].set(value=2013)
        #self.pars['v2_center'].set(value=2013.8)
        self.pars['v1_amplitude'].set(value=max(self.lineouty)/2)
        self.pars['v2_amplitude'].set(value=max(self.lineouty))
        #self.pars['bg_intercept'].set(value=-2014)
        for k,v in self.initialprofile.items():
            self.pars[k].set(value=v)
        
    def run_fit(self):
        self.out = self.model.fit(self.lineouty,self.pars,x=self.lineoutx)
        self.complist = np.array([[self.lineoutx,self.out.eval_components()[i]] for i in ('v1_','v2_','bg_')])
    
    def plot_summary(self):
        plt.plot(self.lineoutx,self.lineouty,label='data')
        plt.plot(self.lineoutx,self.model.eval(self.pars,x=self.lineoutx),label='initial')
        try:
            plt.plot(self.lineoutx,self.out.best_fit,label='fit')
        except:
            plt.show()
        else:
            plt.show()

    def plot_fit(self, mode=u'markers', show=True, normalize='integral', data=True):
        lineouty=self.lineouty
        if normalize=='integral':
            lineouty=norm(lineouty,mode=normalize)
        elif normalize == 'peak':
            lineouty=norm(lineouty,mode=normalize)

        fitlabel = self.sample+' fit'
        if data:
            plt.plot(self.lineoutx,lineouty,label=self.sample+' data',mode=mode)
            fitcolor = 'rgba(0,0,0,0.7)'
            # fitlabel = None
        else:
            fitcolor = None
            # fitlabel = self.sample+' fit' // this line and line above commented out on 3.9, delete later if unneeded

        try:
            bestfit = self.out.best_fit
            if normalize=='integral':
                bestfit = bestfit/np.sum(self.lineouty)
            elif normalize=='peak':
                bestfit = bestfit/np.max(self.lineouty)
            plt.plot(self.lineoutx,bestfit,label=fitlabel,color=fitcolor)
        except:
            if show:
                plt.show()
        else:
            if show:
                plt.show()
        
    def residual_plot(self,save=False,**kwargs):
        fit_resid_plotly(self.out,self.lineoutx,poisson=1,comptraces=self.complist,save=save,**kwargs)
        
    def spin_splitting(self):
        split = self.out.best_values['v2_center']-self.out.best_values['v1_center']
        print('Ka1/Ka2 spin-split is : ',split)

    def print_summary(self,verbose=False):
        fitbestvalues = self.out.best_values
        if verbose:
            for p in ('v1_center','v2_center','v1_gamma','v2_gamma','v1_sigma','v2_sigma'):
                print(p+':\t'+str(fitbestvalues[p]))
        split = fitbestvalues['v2_center']-fitbestvalues['v1_center']
        ratio = fitbestvalues['v2_amplitude']/fitbestvalues['v1_amplitude']
        print('for sample: '+self.sample)
        print('\tsplitting:\t'+'{: 10.3f}'.format(split))
        print('\tratio:\t\t'+'{: 10.3f}'.format(ratio))
        print('\tka1:\t\t'+'{: 10.3f}'.format(fitbestvalues['v2_center']))
        print('\tgamma1:\t\t'+'{: 10.3f}'.format(fitbestvalues['v2_gamma']))
        print('\tsigma1:\t\t'+'{: 10.3f}'.format(fitbestvalues['v2_sigma']))
        print('\tredchi:\t\t'+'{: 10.3f}'.format(self.out.redchi))

pkalpha4peakfitprofile = dict(v1_gamma=0.3,v1_center=2013.5-0.8,v2_center=2013.5,
    v3_center=2014.5-0.8,v4_center=2014.5,bg_intercept=-2014)
        
class do_four_peak_fit:
    
    def __init__(self,lineout,linear=True,sample='sample',initialprofile=pkalpha4peakfitprofile):
        self.sample=sample
        self.lineoutx=lineout[0]
        self.lineouty=lineout[1]
        self.voigt1 = models.VoigtModel(prefix='v1_')
        self.voigt2 = models.VoigtModel(prefix='v2_')
        self.voigt3 = models.VoigtModel(prefix='v3_')
        self.voigt4 = models.VoigtModel(prefix='v4_')
        self.linbg = models.LinearModel(prefix='bg_')
        self.initialprofile=initialprofile
        if linear:
            self.model = self.voigt1+self.voigt2+self.voigt3+self.voigt4+self.linbg
            self.linterm = True
        else:
            self.model = self.voigt1+self.voigt2+self.voigt3+self.voigt4
            self.linterm = False
        self.initialize_pars()
        self.reset_pars()
        self.complist=[]
    
    def initialize_pars(self):
        self.pars=self.voigt1.make_params()
        self.pars.update(self.voigt2.make_params())
        self.pars.update(self.voigt3.make_params())
        self.pars.update(self.voigt4.make_params())
        if self.linterm:
            self.pars.update(self.linbg.make_params())
    
    def reset_pars(self):
        [self.pars[x].set(value=0.22) for x in [('v%d_sigma')%i for i in (1,2,3,4)]]
        #self.pars['v1_amplitude'].set(expr='v2_amplitude*0.5')
        self.pars['v2_sigma'].set(expr='v1_sigma')
        self.pars['v3_sigma'].set(expr='v1_sigma')
        self.pars['v4_sigma'].set(expr='v1_sigma')
        self.pars['v1_gamma'].set(vary=True)
        self.pars['v2_gamma'].set(expr='v1_gamma')
        self.pars['v3_gamma'].set(expr='v1_gamma')
        self.pars['v4_gamma'].set(expr='v1_gamma')
        #self.pars['v1_gamma'].set(value=0.3)
        #self.pars['v1_center'].set(value=2013)
        #self.pars['v2_center'].set(value=2013.8)
        #self.pars['v3_center'].set(value=2014.5-.8)
        #self.pars['v4_center'].set(value=2014.5)
        for k,v in self.initialprofile.items():
            self.pars[k].set(value=v)
        self.pars['v1_amplitude'].set(value=max(self.lineouty)/3)
        self.pars['v2_amplitude'].set(value=max(self.lineouty)/1.5)
        self.pars['v3_amplitude'].set(value=max(self.lineouty)/3)
        self.pars['v4_amplitude'].set(value=max(self.lineouty)/1.5)
        if self.linterm:
            self.pars['bg_intercept'].set(value=-2014)
        
    def run_fit(self):
        self.out = self.model.fit(self.lineouty,self.pars,x=self.lineoutx)
        if self.linterm:
            prefixlist = ('v1_','v2_','v3_','v4_','bg_')
        else:
            prefixlist = ('v1_','v2_','v3_','v4_')
        self.complist = np.array([[self.lineoutx,self.out.eval_components()[i]] for i in prefixlist])
    
    def plot_summary(self):
        plt.plot(self.lineoutx,self.lineouty,label='data')
        plt.plot(self.lineoutx,self.model.eval(self.pars,x=self.lineoutx),label='initial')
        try:
            plt.plot(self.lineoutx,self.out.best_fit,label='fit')
        except:
            plt.show()
        else:
            plt.show()

    def plot_fit(self, mode=u'markers', show=True, peaknormalize=True, data=True):
        lineouty=self.lineouty
        if peaknormalize:
            lineouty=norm(lineouty)
        if data:
            plt.plot(self.lineoutx,lineouty,label=self.sample+' data',mode=mode)
            fitcolor = 'rgba(0,0,0,0.7)'
            fitlabel = None
        else:
            fitcolor = None
            fitlabel = self.sample+' fit'
        try:
            bestfit = self.out.best_fit
            if peaknormalize:
                bestfit = bestfit/max(self.lineouty)
            plt.plot(self.lineoutx,bestfit,label=fitlabel,color=fitcolor)
        except:
            if show:
                plt.show()
        else:
            if show:
                plt.show()
        
    def residual_plot(self,save=False,**kwargs):
        fit_resid_plotly(self.out,self.lineoutx,poisson=1,comptraces=self.complist,save=save,**kwargs)
        
    def spin_splitting(self):
        split12 = self.out.best_values['v2_center']-self.out.best_values['v1_center']
        split34 = self.out.best_values['v4_center']-self.out.best_values['v3_center']
        print('V1/V2 spin-split is: ',split12)
        print('V3/V4 spin-split is: ',split34)
        
    def oxid_split(self):
        split = self.out.best_values['v4_center']-self.out.best_values['v2_center']
        print('split between reduced/oxidized component is: ',split)
        
    def oxid_red_complist(self):
        if self.linterm:
            prefixlist = ('v1_','v2_','v3_','v4_','bg_')
        else:
            prefixlist = ('v1_','v2_','v3_','v4_')
        allcomps = np.array([[self.lineoutx,self.out.eval_components()[i]] for i in prefixlist])
        comp1 = np.array([allcomps[0][0],allcomps[0][1]+allcomps[1][1]])
        comp2 = np.array([allcomps[2][0],allcomps[2][1]+allcomps[3][1]])
        if self.linterm:
            self.complist=[comp1,comp2,allcomps[-1]]
        else:
            self.complist=[comp1,comp2]

    def save_csv(self,filename):
        savedata = np.array([self.lineoutx,self.lineouty,self.out.best_fit,self.complist[0][1],self.complist[1][1]])#,self.complist[2][1]])
        np.savetxt(filename,savedata,delimiter=',',header='xvalues,rawdata,fit,comp0,comp1,comp2')
        print('file saved as:',filename)

def kalpha_doublet_model(mainpeakpos=2307.35,gamma=0.4475,sigma=0.1496,splitting=1.17,ratio=1/0.546,prefix1='a1_',prefix2='a2_'):
    """Makes lmfit model of kalpha doublet shape.
    
    Model is composed of two voigts with fixed Lorentzian
     and Gaussian widths and fixed splitting and peak ratios.
    Default values are a fit to ZnS doublet shape.

    Args:
        mainpeakpos: Energy (eV) position of kalpha1 peak.
        gamma: Lorentzian width (in voigt model).
        sigma: Gaussian width (in voigt model).
        splitting: Energy gap (eV) between ka1,ka2
        ratio: Ratio of peak heights ka1/ka2

    Returns:
        lmfit model, lmfit pars

    Raises:
        Nothing.
    """
    mod = models.VoigtModel(prefix=prefix1)+models.VoigtModel(prefix=prefix2)
    pars = mod.make_params()
    parsdict = dict([[prefix2+'amplitude',{'expr': prefix1+'amplitude*'+str(1/ratio), 'value': 1/ratio, 'vary': False}],
                     [prefix2+'gamma', {'expr': prefix1+'gamma', 'value': gamma, 'vary': False}],
                     [prefix2+'center', {'expr': prefix1+'center-'+str(splitting), 'value': mainpeakpos-splitting, 'vary': False}],
                     [prefix1+'sigma', {'expr': '', 'value': sigma, 'vary': False}],
                     [prefix1+'gamma', {'expr': '', 'value': gamma, 'vary': False}],
                     [prefix1+'center', {'expr': '', 'value': mainpeakpos, 'vary': True}],
                     [prefix2+'sigma', {'expr': prefix1+'sigma', 'value': sigma, 'vary': False}]])
    for k,v in parsdict.items():
        pars[k].set(vary=v['vary'],expr=v['expr'])
    for k,v in parsdict.items():
        pars[k].value=v['value']
    return mod, pars

import collections
examplerefpeakshapes=collections.OrderedDict([
    ('reduced',
        collections.OrderedDict([('mainpeakpos',2307.69),('gamma',0.4475),('sigma',0.1496),('splitting',1.17),('ratio',1/0.546)])),
    ('oxidized',
        collections.OrderedDict([('mainpeakpos',2309.15),('gamma',0.4475),('sigma',0.1496),('splitting',1.17),('ratio',1/0.546)]))])

class kalpha_linear_combination_fit:

    def __init__(self,lineout,sample='sample',refpeakshapes=None,linbg=True,runoninit=False, fitregion=None):
        self.sample=sample
        self.lineoutx=lineout[0]
        self.lineouty=lineout[1]
        if not fitregion:
            self.flineoutx, self.flineouty = self.lineoutx, self.lineouty
        else:
            self.flineoutx, self.flineouty =_take_lineout_erange(lineout,fitregion)
        self.refpeakshapes = refpeakshapes
        self.linbg = linbg
        self.make_model()
        if runoninit:
            self.do_fit()

    def make_model(self):
        self.modellist = []
        self.parslist = []

        for k,v in self.refpeakshapes.items():
            kdm, kdmpars = kalpha_doublet_model(prefix1=k+'_1_',prefix2=k+'_2_',**v)
            self.modellist.append(kdm)
            self.parslist.append(kdmpars)
        if self.linbg:
            self.modellist.append(models.LinearModel(prefix='linbg_'))
            self.parslist.append(self.modellist[-1].make_params())

        self.model = self.modellist[0]
        for mod in self.modellist[1:]:
            self.model = self.model+mod
        self.pars = self.parslist[0]
        for par in self.parslist[1:]:
            self.pars = self.pars+par

        for k in self.pars:
            if '1_amplitude' in k:
                self.pars[k].set(value=max(self.lineouty))

        if self.linbg:
            self.pars['linbg_intercept'].set(value=-list(self.refpeakshapes.items())[0][1]['mainpeakpos'])

    def do_fit(self,bgcomp=True):
        self.out = self.model.fit(self.flineouty,self.pars,x=self.flineoutx)
        self.complist = collections.OrderedDict()
        components = self.out.eval_components()
        for k in self.refpeakshapes:
            self.complist[k]=[self.lineoutx,components[k+'_1_']+components[k+'_2_']]
            #self.complist.append([self.lineoutx,components[k+'_1_']+components[k+'_2_']])

        if bgcomp:
            self.complist['linbg']=[self.lineoutx,components['linbg_']]
            #self.complist.append([self.lineoutx,components['linbg_']])

    def calc_contributions(self,printenergy=False):
        sumdict = collections.OrderedDict()
        for k,v in self.complist.items():
            if 'linbg' not in k:
                sumdict[k]=np.sum(v[1])
        tot = np.sum(list(sumdict.values()))

        self.components=collections.OrderedDict()
        for k,v in sumdict.items():
            self.components[k]=v/tot

        print('for sample '+self.sample)
        for k,v in self.components.items():
            print(k+': '+str(round(v*100,3))+'%')
        if printenergy:
            for k,v in self.out.best_values.items():
                if '1_center' in k:
                    print(k,' at ','{:6.2f}'.format(v))

    def residuals_plot(self,poisson=True,save=False,**kwargs):
        fit_resid_plotly(
            self.out,xvalues=self.lineoutx,
            comptraces=list(self.complist.values()),poisson=poisson,
            save=save,complabels=list(self.complist.keys()),**kwargs)

    def plot_summary(self):
        plt.plot(self.lineoutx,self.lineouty,label='data')
        plt.plot(self.lineoutx,self.model.eval(self.pars,x=self.lineoutx),label='initial')
        try:
            plt.plot(self.lineoutx,self.out.best_fit,label='fit')
        except:
            plt.show()
        else:
            plt.show()

    def save_fit_csv(self,filename):
        savelist = [self.lineoutx,self.lineouty,self.out.best_fit]
        _ = [savelist.append(c[1]) for c in self.complist.values()]
        savedata = np.array(savelist)
        headerlist = ['Energy(eV)','Rawdata (counts)','Fit']
        _ = [headerlist.append(k+'comp') for k in self.complist.keys()]
        np.savetxt(filename,np.transpose(savedata),delimiter=',',header=','.join(headerlist))
        print('file saved as:',filename)


    def pretty_plot_fit(self,xrange=None,save=False,joined=False):
        data=[]

        fittrace = go.Scatter(x=self.lineoutx,y=self.out.best_fit,xaxis='x',yaxis='y',name='fit',
                             line=dict(color='rgba(0,0,0,0.7)'))

        if joined:
            datamode = 'lines+markers'
        else:
            datamode = 'markers'
        datatrace = go.Scatter(x=self.lineoutx,y=self.lineouty,xaxis='x',yaxis='y',name='data',mode=datamode,
                              marker = dict(size=5,color='rgba(200,0,0,0.8)'))

        for trace in[datatrace,fittrace]:
            data.append(trace)
        
        #colorlist = [
    #     [0, 'rgb(150,0,90)'], [0.125, 'rgb(0,0,200)'],
    #     [0.25, 'rgb(0,25,255)'], [0.375, 'rgb(0,152,255)'],
    #     [0.5, 'rgb(44,255,150)'], [0.625, 'rgb(151,255,0)'],
    #     [0.75, 'rgb(255,234,0)'], [0.875, 'rgb(255,111,0)'],
    #     [1, 'rgb(255,0,0)']
    # ]
        complabels = list(self.complist.keys())
        i=0
        for tr in self.complist.values():
            #data.append(go.Scatter(x=tr[0],y=tr[1],xaxis='x',yaxis='y',name=complabels[i],line=dict(color=colorlist[i][1])))#,visible='legendonly'))
            data.append(go.Scatter(x=tr[0],y=tr[1],xaxis='x',yaxis='y',name=complabels[i]))#,visible='legendonly'))
            i+=1
        
        layout=go.Layout(
            xaxis=dict(domain=[0,1],anchor='y',title='Energy(ev)',range=xrange),
            yaxis=dict(domain=[0,1],title='intensity'),
            height=600)

        data = data[::-1]
        fig=go.Figure(data=data,layout=layout)
        if save:
            imagestr='svg'
        else:
            imagestr=None
        offline.iplot(fig,image=imagestr)


        # if self.linbg:
        #     rindex = -1
        # else:
        #     rindex = None
        # for c in self.complist[:rindex]:
        #     tot=tot+np.sum(c[1])

    def print_fit_summary(self):
        self.calc_contributions()
        fitbestvalues = self.out.best_values
        print('Oxidized Kalpha1: '+str(fitbestvalues['oxidized_1_center'])+' eV')
        print('Reduced Kalpha1: '+str(fitbestvalues['reduced_1_center'])+' eV')






# Old version
# class do_four_peak_fit:
    
#     def __init__(self,lineout):
#         self.lineoutx=lineout[0]
#         self.lineouty=lineout[1]
#         self.voigt1 = models.VoigtModel(prefix='v1_')
#         self.voigt2 = models.VoigtModel(prefix='v2_')
#         self.voigt3 = models.VoigtModel(prefix='v3_')
#         self.voigt4 = models.VoigtModel(prefix='v4_')
#         self.linbg = models.LinearModel(prefix='bg_')
#         self.model = self.voigt1+self.voigt2+self.voigt3+self.voigt4+self.linbg
#         self.initialize_pars()
#         self.reset_pars()
#         self.complist=[]
    
#     def initialize_pars(self):
#         self.pars=self.voigt1.make_params()
#         self.pars.update(self.voigt2.make_params())
#         self.pars.update(self.voigt3.make_params())
#         self.pars.update(self.voigt4.make_params())
#         self.pars.update(self.linbg.make_params())
    
#     def reset_pars(self):
#         [self.pars[x].set(value=0.22) for x in [('v%d_sigma')%i for i in (1,2,3,4)]]
#         #self.pars['v1_amplitude'].set(expr='v2_amplitude*0.5')
#         self.pars['v2_sigma'].set(expr='v1_sigma')
#         self.pars['v3_sigma'].set(expr='v1_sigma')
#         self.pars['v4_sigma'].set(expr='v1_sigma')
#         self.pars['v1_gamma'].set(vary=True)
#         self.pars['v2_gamma'].set(expr='v1_gamma')
#         self.pars['v3_gamma'].set(expr='v1_gamma')
#         self.pars['v4_gamma'].set(expr='v1_gamma')
#         self.pars['v1_gamma'].set(value=0.3)
#         self.pars['v1_center'].set(value=2013)
#         self.pars['v2_center'].set(value=2013.8)
#         self.pars['v3_center'].set(value=2014.5-.8)
#         self.pars['v4_center'].set(value=2014.5)
#         self.pars['v1_amplitude'].set(value=max(self.lineouty)/4)
#         self.pars['v2_amplitude'].set(value=max(self.lineouty)/2)
#         self.pars['v3_amplitude'].set(value=max(self.lineouty)/4)
#         self.pars['v4_amplitude'].set(value=max(self.lineouty)/2)
#         self.pars['bg_intercept'].set(value=-2014)
        
#     def run_fit(self):
#         self.out = self.model.fit(self.lineouty,self.pars,x=self.lineoutx)
#         self.complist = np.array([[self.lineoutx,self.out.eval_components()[i]] for i in ('v1_','v2_','v3_','v4_','bg_')])
    
#     def plot_summary(self):
#         plt.plot(self.lineoutx,self.lineouty,label='raw data')
#         plt.plot(self.lineoutx,self.model.eval(self.pars,x=self.lineoutx),label='initial')
#         try:
#             plt.plot(self.lineoutx,self.out.best_fit,label='fit')
#         except:
#             plt.show()
#         else:
#             plt.show()
        
#     def residual_plot(self,save=False,**kwargs):
#         fit_resid_plotly(self.out,self.lineoutx,poisson=1,comptraces=self.complist,save=save,**kwargs)
        
#     def spin_splitting(self):
#         split12 = self.out.best_values['v2_center']-self.out.best_values['v1_center']
#         split34 = self.out.best_values['v4_center']-self.out.best_values['v3_center']
#         print('V1/V2 spin-split is: ',split12)
#         print('V3/V4 spin-split is: ',split34)
        
#     def oxid_split(self):
#         split = self.out.best_values['v4_center']-self.out.best_values['v2_center']
#         print('split between reduced/oxidized component is: ',split)
        
#     def oxid_red_complist(self):
#         allcomps = np.array([[self.lineoutx,self.out.eval_components()[i]] for i in ('v1_','v2_','v3_','v4_','bg_')])
#         comp1 = np.array([allcomps[0][0],allcomps[0][1]+allcomps[1][1]])
#         comp2 = np.array([allcomps[2][0],allcomps[2][1]+allcomps[3][1]])
#         self.complist=[comp1,comp2,allcomps[-1]]
