from itertools import product
import warnings

from plotly.subplots import make_subplots
from dash import dcc, html
from dash.dependencies import Input, Output

import numpy as np
import pandas as pd
from copy import copy

from mne import create_info
from mne.io import RawArray

import plotly.graph_objects as go
from mne.viz.topomap import _get_pos_outlines
from mne.utils.check import _check_sphere
from mne.viz.topomap import _setup_interp, _check_extrapolate

from . import ic_label_cmap

from copy import copy
axis = {'showgrid': False, # thin lines in the background
         'visible': False,  # numbers below
        }
yaxis = copy(axis)
yaxis.update(dict(scaleanchor="x", scaleratio=1))

class TopoData:
    def __init__(self, topo_values=()):
        """topo_values: list of dict """
        self.topo_values = pd.DataFrame(topo_values)

    def add_topomap(self, topomap: dict):
        """topomap: dict"""
        self.topo_values = self.topo_values.append(topomap)

    @property
    def nb_topo(self):
        return self.topo_values.shape[0]


class TopoViz:
    def __init__(self, app, montage, data: TopoData, rows=5, cols=4,
                 margin_x=4/5, width=600, height=800, margin_y=2/5,
                 topo_slider_id=None, head_contours_color="black"):
        """ """
        self.montage = montage
        self.data = data
        self.app = app
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height
        self.heatmap_traces = None
        self.colorbar = False
        fig = make_subplots(rows=rows, cols=cols,
                            horizontal_spacing=0.01,
                            vertical_spacing=0.01)
        self.graph = dcc.Graph(figure=fig, id='topo-graph',
                               className='dcc-graph')
        self.graph_div = html.Div(children=[self.graph],
                                  style={"border":"2px red solid"},
                                  className='dcc-graph-div')

        self.margin_x = margin_x
        self.margin_y = margin_y
        self.offset = 0
        self.topo_slider = None
        self.use_topo_slider = topo_slider_id
        self.head_contours_color = head_contours_color

        names = self.data.topo_values.columns.tolist()
        self.info = create_info(names, sfreq=256, ch_types="eeg")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            RawArray(np.zeros((len(names), 1)), self.info, copy=None,
                     verbose=False).set_montage(montage)

        self.pos = None
        self.contours = None
        self.set_head_pos_contours()

        self.init_slider()
        self.initialize_layout()
        self.set_div()
        self.set_callback()

    def set_head_pos_contours(self, sphere=None, picks=None):
        sphere = _check_sphere(sphere, self.info)
        self.pos, self.outlines = _get_pos_outlines(self.info, picks, sphere,
                                                    to_sphere=True)

    def get_head_scatters(self, color="back"):
        outlines_scat = [go.Scatter(x=x, y=y, line=dict(color=color),
                                    mode='lines', showlegend=False)
                        for key, (x, y) in self.outlines.items()
                        if 'clip' not in key]
        pos_scat = go.Scatter(x=self.pos.T[0], y=self.pos.T[1],
                              line=dict(color=color), mode='markers',
                              showlegend=False)

        return outlines_scat + [pos_scat]

    def get_heatmap_data(self, i, j, ch_type="eeg", res=64,
                         extrapolate='auto'):
        # Get the heatmap
        no = i*self.cols + j
        if no + self.offset >= self.data.nb_topo:  # out of range
            return [[]]
        value_dict = dict(self.data.topo_values.iloc[no+self.offset])

        extrapolate = _check_extrapolate(extrapolate, ch_type)
        # find mask limits and setup interpolation
        _, Xi, Yi, interp = _setup_interp(self.pos, res=res,
                                          image_interp="cubic",
                                          extrapolate=extrapolate,
                                          outlines=self.outlines,
                                          border='mean')
        interp.set_values(np.array(list(value_dict.values())))
        Zi = interp.set_locations(Xi, Yi)()

        # Clip to the outer circler
        x0, y0 = self.outlines["clip_origin"]
        x_rad , y_rad = self.outlines["clip_radius"]
        Zi[np.sqrt(((Xi - x0)/x_rad)**2 + ((Yi-y0)/y_rad)**2) > 1] = np.nan

        return {"x":Xi[0], "y": Yi[:, 0], "z": Zi}

    def initialize_layout(self, slider_val=None):

        if slider_val != None:
            self.offset = self.topo_slider.max-slider_val

        ic_names = self.data.topo_values.index
        ic_names = ic_names[self.offset: self.offset+self.rows*self.cols]
        self.graph.figure = make_subplots(rows=self.rows, cols=self.cols,
                            horizontal_spacing=0.03,
                            vertical_spacing=0.03,
                            subplot_titles=ic_names)

        self.heatmap_traces = [[go.Heatmap(showscale=self.colorbar,
                                           **self.get_heatmap_data(i, j))
                                for j in np.arange(self.cols)]
                                for i in np.arange(self.rows)]

        for no, (i, j) in enumerate(product(np.arange(self.rows),
                                            np.arange(self.cols))):
            if no + self.offset >= self.data.nb_topo: # out of range
                break

            if isinstance(self.head_contours_color, str):
                color = self.head_contours_color
            elif ic_names[no] in self.head_contours_color:
                color = self.head_contours_color[ic_names[no]]
            else:
                color = "black"
            for trace in self.get_head_scatters(color=color):
                self.graph.figure.add_trace(trace, row=i+1, col=j+1)
            self.graph.figure.add_trace(self.heatmap_traces[i][j],
                                        row=i+1, col=j+1)

        for i in range(1, self.rows*self.cols+1):
            self.graph.figure['layout'][f'xaxis{i}'].update(axis)
            self.graph.figure['layout'][f'yaxis{i}'].update(yaxis)

        self.graph.figure.update_layout(
            autosize=False,
            width=self.width,
            height=self.height,
            plot_bgcolor="white")
        self.graph.figure['layout'].update(margin=dict(l=0,r=0,b=0,t=20))

    def init_slider(self):
        self.topo_slider = dcc.Slider(id='topo-slider',
                                min=self.rows * self.cols -1,
                                max=self.data.nb_topo -1,
                                step=1,
                                marks=None,
                                value=self.data.nb_topo -1,
                                included=False,
                                updatemode='mouseup',
                                vertical=True,
                                verticalHeight=self.graph.figure.layout.height)

        
    def set_div(self):
        if self.use_topo_slider is None:
            # outer_div includes slider obj
            outer_div = [html.Div(self.topo_slider,
                                  style={"border":"2px purple solid",
                                         'display':'inline-block'}),
                                  self.graph_div]
        else:
            # outer_div is just the graph
            outer_div = [self.graph_div]
        self.container_plot = html.Div(children=outer_div,
                                       style={"border":"2px black solid"},
                                       className="outer-timeseries-div")

    def set_callback(self):
        args = [Output('topo-graph', 'figure')]
        if self.use_topo_slider:
            args += [Input(self.use_topo_slider, 'value')]
        else: 
            args += [Input('topo-slider', 'value')]

        @self.app.callback(*args, suppress_callback_exceptions=False)
        def callback(slider_val):             
            self.initialize_layout(slider_val=slider_val)
            return self.graph.figure


class TopoVizICA(TopoViz):
    def __init__(self, app, montage, ica, ic_labels=None, **kwargs):
        """ """

        if ic_labels:
            kwargs["head_contours_color"] = {comp: ic_label_cmap[label]
                                             for comp, label
                                             in ic_labels.items()}
        data = TopoData([dict(zip(montage.ch_names, component))
                              for component in ica.get_components().T])
        data.topo_values.index = list(ic_labels.keys())
        super(TopoVizICA, self).__init__(app, montage, data, **kwargs)
