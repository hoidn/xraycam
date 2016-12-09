from lmfit import minimize, Parameters, models
from plotly import tools
import plotly.offline as offline
import plotly.graph_objs as go
import numpy as np
from xraycam.camcontrol import plt

def norm(y):
    return y/max(y)

def fit_resid_plotly(lmfitoutput,xvalues,xrange=None,poisson=False,comptraces=[],complabels=[],save=False):
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
    
    residtrace = go.Scatter(x=xvalues,y=lmfitoutput.residual,xaxis='x',yaxis='y',mode='lines+markers',name='residuals',
                           marker=dict(size=5),line=dict(width=1,
                                color='rgb(49,54,149)'))
    fittrace = go.Scatter(x=xvalues,y=lmfitoutput.best_fit,xaxis='x',yaxis='y2',name='fit',
                         line=dict(color='rgba(0,0,0,0.7)'))
    datatrace = go.Scatter(x=xvalues,y=lmfitoutput.data,xaxis='x',yaxis='y2',name='data',mode='markers',
                          marker = dict(size=5,color='rgba(200,0,0,0.8)'))
    for trace in[residtrace,datatrace,fittrace]:
        data.append(trace)
    
    if comptraces is not []:
        i=0
        if not complabels:
            complabels=['comp%d'% j for j in range(len(comptraces))]
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

class do_peak_fit:
    
    def __init__(self,lineout, numpeaks=2,sample='sample'):
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
        self.initialize_pars()
        self.complist=[]
    
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
        self.pars['v1_gamma'].set(value=0.3)
        self.pars['v1_center'].set(value=2013)
        self.pars['v2_center'].set(value=2013.8)
        self.pars['v1_amplitude'].set(value=max(self.lineouty)/2)
        self.pars['v2_amplitude'].set(value=max(self.lineouty))
        self.pars['bg_intercept'].set(value=-2014)
        
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
        split = self.out.best_values['v2_center']-self.out.best_values['v1_center']
        print('Ka1/Ka2 spin-split is : ',split)
        
class do_four_peak_fit:
    
    def __init__(self,lineout,linear=True,sample='sample'):
        self.sample=sample
        self.lineoutx=lineout[0]
        self.lineouty=lineout[1]
        self.voigt1 = models.VoigtModel(prefix='v1_')
        self.voigt2 = models.VoigtModel(prefix='v2_')
        self.voigt3 = models.VoigtModel(prefix='v3_')
        self.voigt4 = models.VoigtModel(prefix='v4_')
        self.linbg = models.LinearModel(prefix='bg_')
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
        self.pars['v1_gamma'].set(value=0.3)
        self.pars['v1_center'].set(value=2013)
        self.pars['v2_center'].set(value=2013.8)
        self.pars['v3_center'].set(value=2014.5-.8)
        self.pars['v4_center'].set(value=2014.5)
        self.pars['v1_amplitude'].set(value=max(self.lineouty)/4)
        self.pars['v2_amplitude'].set(value=max(self.lineouty)/2)
        self.pars['v3_amplitude'].set(value=max(self.lineouty)/4)
        self.pars['v4_amplitude'].set(value=max(self.lineouty)/2)
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
