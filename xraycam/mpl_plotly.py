u"""
A matplotlib-esque interface for Plotly-based interactive plots for the Jupyter
notebook.
"""



from __future__ import absolute_import
import matplotlib.pyplot as mplt
import plotly.offline as py
import plotly.graph_objs as go

py.offline.init_notebook_mode()

class PyplotPlot(object):
    def show(self):
        mplt.show()


class Figure(object):
    from . import utils
    u"""Class containing the equivalent of a matplotlib Axis."""
    def __init__(self):
        self.traces = []
        self.xaxis = {u'exponentformat': u'power'}
        self.yaxis = {u'exponentformat': u'power'}
        self.layout = {u'xaxis': self.xaxis, u'yaxis': self.yaxis}


    def plot(self, *args, **kwargs):
        if u'color' in kwargs: color = kwargs[u'color']; del kwargs[u'color']
        else: color =  None
        if u'label' in kwargs: label = kwargs[u'label']; del kwargs[u'label']
        else: label =  None
        if u'mode' in kwargs: mode = kwargs[u'mode']; del kwargs[u'mode']
        else: mode = u'lines'
        u"""
        Add a curve to the figure.
        kwargs are passed through to the argument dictionary for go.Scatter.
        """
        if len(args) == 2:
            x, y = args
        elif len(args) == 1:
            x, y = range(len(args[0])), args[0]
        else:
            raise ValueError(u"Plot accepts one or two positional arguments")
        scatter_kwargs = dict(x = x, y = y, name = label,
                line = dict(color = color), mode = mode)
        if label is None and mode == u'lines':
            scatter_kwargs[u'showlegend'] = False
            scatter_kwargs[u'hoverinfo'] = u'none'
            
        else:
            scatter_kwargs[u'showlegend'] = True
        self.traces.append(
            go.Scatter(**Figure.utils.merge_dicts(scatter_kwargs, kwargs)))

    def hist(self, x, alpha = .75, bins = None, color = None, label = None, **kwargs):
        # TODO: control number of bins
        default = dict(
            x = x,
            opacity = alpha,
            name = label,
            marker = dict(color = color)
        )
        self.traces.append(go.Histogram(**Figure.utils.merge_dicts(default, kwargs)))
        self.layout[u'barmode'] = u'overlay'

    def scatter(self, *args, **kwargs):
        kwargs[u'mode'] = u'markers'
        self.plot(*args, **kwargs)

    def set_xlabel(self, xlabel):
        self.xaxis[u'title'] = xlabel

    def set_ylabel(self, ylabel):
        self.yaxis[u'title'] = ylabel

    def set_title(self, title):
        self.layout[u'title'] = title

    def set_xscale(self, value):
        if value == u'log':
            self.xaxis[u'type'] = u'log'
        else:
            raise NotImplementedError

    def set_yscale(self, value):
        if value == u'log':
            self.yaxis[u'type'] = u'log'
        else:
            raise NotImplementedError

    def show(self):
        data = self.traces
        fig = go.Figure(data = data, layout = go.Layout(**self.layout))
        py.iplot(fig)

class Plt(object):
    def __init__(self):
        self.mode = None
        self.figures = []
        self.plt_global = None
    
    def _clear(self):
        self.__init__()

    def _get_global_plot(self):
        u"""
        Return the Figure instance for this object, creating it if necessary.
        """
        if self.plt_global is None:
            self.plt_global = Figure()
            self.figures.append(self.plt_global)
        return self.plt_global

    def subplots(self, *args, **kwargs):
        self.mode = u'plotly'
        if len(args) > 0:
            n_plots = args[0]
            self.figures = [Figure() for _ in xrange(n_plots)]
            return None, self.figures
        else:
            self.figures.append(Figure())
            return None, self.figures[0]

    def plot(self, *args, **kwargs):
        self._get_global_plot().plot(*args, **kwargs)

    def hist(self, *args, **kwargs):
        self._get_global_plot().hist(*args, **kwargs)

    def scatter(self, *args, **kwargs):
        self._get_global_plot().scatter(*args, **kwargs)

    def xlabel(self, xlabel):
        self._get_global_plot().set_xlabel(xlabel)

    def ylabel(self, ylabel):
        self._get_global_plot().set_ylabel(ylabel)

    def xscale(self, scale):
        self._get_global_plot().set_xscale(scale)

    def yscale(self, scale):
        self._get_global_plot().set_yscale(scale)

    def title(self, title):
        self._get_global_plot().set_title(title)

    def legend(self):
        pass

    def show(self):
        for fig in self.figures:
            fig.show()
        self._clear()

    def imshow(self, *args, **kwargs):
        mplt.imshow(*args, **kwargs)
        self.figures.append(PyplotPlot())

    def savefig(self, *args, **kwargs):
        # TODO: implement this
        pass


plt = Plt()
