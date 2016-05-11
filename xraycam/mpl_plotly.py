"""
A matplotlib-esque interface for Plotly-based interactive plots for the Jupyter
notebook.
"""

import matplotlib.pyplot as mplt
import plotly.offline as py
import plotly.graph_objs as go

from xraycam import config

py.offline.init_notebook_mode()

class PyplotPlot:
    def show(self):
        mplt.show()

class Figure:
    """Class containing the equivalent of a matplotlib Axis."""
    def __init__(self):
        self.lines = []
        self.xaxis = {'exponentformat': 'power'}
        self.yaxis = {}

    def plot(self, *args, label = None, color = None, **kwargs):
        """
        Add a curve to the figure.

        kwargs are passed through to the argument dictionary for go.Scatter.
        """
        if len(args) == 2:
            x, y = args
        elif len(args) == 1:
            x, y = list(range(len(args[0]))), args[0]
        else:
            raise ValueError("Plot accepts one or two positional arguments")
        scatter_kwargs = dict(x = x, y = y, name = label,
                mode = 'lines', line = dict(color = color))
        if label is None:
            scatter_kwargs['showlegend'] = False
            scatter_kwargs['hoverinfo'] = 'none'
            
        else:
            scatter_kwargs['showlegend'] = True
        self.lines.append(
            go.Scatter(**{**scatter_kwargs, **kwargs}))

    def set_xlabel(self, xlabel):
        self.xaxis['title'] = xlabel

    def set_ylabel(self, ylabel):
        self.yaxis['title'] = ylabel

    def set_xscale(self, value):
        if value == 'log':
            self.xaxis['type'] = 'log'
        else:
            raise NotImplementedError

    def show(self):
        layout = go.Layout(xaxis = self.xaxis, yaxis = self.yaxis)
        data = self.lines
        fig = go.Figure(data = data, layout = layout)
        py.iplot(fig)

class Plt:
    def __init__(self):
        self.mode = None
        self.figures = []
        self.plt_global = None
    
    def clear(self):
        self.__init__()

    def subplots(self, *args, **kwargs):
        self.mode = 'plotly'
        if len(args) > 0:
            n_plots = args[0]
            self.figures = [Figure() for _ in range(n_plots)]
            return None, self.figures
        else:
            self.figures.append(Figure())
            return None, self.figures[0]

    def plot(self, *args, **kwargs):
        if self.plt_global is None:
            self.plt_global = Figure()
            self.figures.append(self.plt_global)
        self.plt_global.plot(*args, **kwargs)

    def legend(self):
        pass

    def show(self):
        for fig in self.figures:
            fig.show()
        self.clear()

    def imshow(self, *args, **kwargs):
        mplt.imshow(*args, **kwargs)
        self.figures.append(PyplotPlot())

    def savefig(self, *args, **kwargs):
        # TODO: implement this
        pass

if config.plotting_mode == 'notebook':
    plt = Plt()
else:
    import matplotlib.pyplot as plt
