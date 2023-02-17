from pathlib import Path

# file selection
import tkinter
from tkinter import filedialog
import pandas as pd

from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


from .topo_viz import TopoVizICA
from .mne_visualizer import MNEVisualizer, ICVisualizer

from . import ic_label_cmap

from .css_defaults import CSS, STYLE


class QCGUI:

    def __init__(self, app, raw, raw_ica, ica, iclabel_fpath,
                 project_root='./tmp_test_files'):

        self.project_root = Path(project_root)

        self.app = app
        self.raw = raw
        self.raw_ica = raw_ica
        self.ica = ica

        self.ica_visualizer = None
        self.eeg_visualizer = None
        self.ica_topo = None
        self.ic_types = pd.read_csv(iclabel_fpath, sep='\t')
        self.ic_types['component'] = [f'ICA{ic:03}'
                                      for ic in self.ic_types.component]
        self.ic_types = self.ic_types.set_index('component')['ic_type']
        self.ic_types = self.ic_types.to_dict()

        self.set_layout()
        self.set_callbacks()

    def annot_created_callback(self, annotation):
        self.raw.set_annotations(self.raw.annotations + annotation)
        self.raw_ica.set_annotations(self.raw_ica.annotations + annotation)
        self.ica_visualizer.update_layout(ch_slider_val=self.ica_visualizer
                                                            .channel_slider
                                                            .max,
                                          time_slider_val=self.ica_visualizer
                                                              .win_start)
        self.eeg_visualizer.update_layout()

    def set_visualizers(self):
        # Setting time-series and topomap visualizers
        cmap = {ic: ic_label_cmap[ic_type] for ic, ic_type in self.ic_types.items()}
        self.ica_visualizer = ICVisualizer(self.app, self.raw_ica,
                                           dash_id_suffix='ica',
                                           annot_created_callback=self.annot_created_callback,
                                           cmap=cmap,
                                           ic_types=self.ic_types)
        self.eeg_visualizer = MNEVisualizer(self.app,
                                            self.raw,
                                            time_slider=self.ica_visualizer
                                                            .dash_ids['time-slider'],
                                            dcc_graph_kwargs=dict(config={'modeBarButtonsToRemove':['zoom','pan']}),
                                            annot_created_callback=self.annot_created_callback)

        self.ica_topo = TopoVizICA(self.app, self.raw.get_montage(), self.ica, self.ic_types,
                                   topo_slider_id=self.ica_visualizer.dash_ids['ch-slider'],
                                   show_sensors=False)

        self.ica_visualizer.new_annot_desc = 'bad_manual'
        self.eeg_visualizer.new_annot_desc = 'bad_manual'

        self.ica_visualizer.update_layout()

    def set_layout(self):
        # app.layout must not be None for some of the operations of the
        # visualizers.
        self.app.layout = html.Div([])
        self.set_visualizers()

        # Layout for file control row
        derivatives_dir = self.project_root / 'derivatives'
        files_list = [{'label': str(file), 'value': str(file)}
                      for file in sorted(derivatives_dir.rglob("*.edf"))]
        dropdown_text = f'current folder: {self.project_root.resolve()}'
        logo_fpath = '../assets/logo_color_thick.png'
        folder_button = dbc.Button('Folder', id='folder-selector',
                                   color='primary',
                                   outline=True,
                                   className=CSS['button'],
                                   title=dropdown_text)
        save_button = dbc.Button('Save', id='save-button',
                                 color='info',
                                 outline=True,
                                 className=CSS['button'])
        drop_down = dcc.Dropdown(id="fileDropdown",
                                 className=CSS['dropdown'],
                                 options=files_list,
                                 placeholder="Select a file")
        control_header_row = dbc.Row([
                                    dbc.Col([folder_button, save_button],
                                            width={'size': 2}),
                                    dbc.Col([drop_down],
                                            width={'size': 6}),
                                    dbc.Col(
                                        html.Img(src=logo_fpath,
                                                 height='40px',
                                                 className=CSS['logo']),
                                        width={'size': 2, 'offset': 2}),
                                      ],
                                     className=CSS['file-row'],
                                     align='center',
                                     )

        # Layout for EEG/ICA and Topo plots row
        timeseries_div = html.Div([self.eeg_visualizer.container_plot,
                                   self.ica_visualizer.container_plot],
                                  id='channel-and-icsources-div',
                                  className=CSS['timeseries-col'])
        visualizers_row = dbc.Row([dbc.Col([timeseries_div], width=8),
                                   dbc.Col(self.ica_topo.container_plot,
                                           className=CSS['topo-col'],
                                           width=4)],
                                  style=STYLE['plots-row'],
                                  className=CSS['plots-row']
                                  )
        # Final Layout
        qc_app_layout = dbc.Container([control_header_row, visualizers_row],
                                      fluid=True, style=STYLE['qc-container'])
        self.app.layout.children.append(qc_app_layout)

    def set_callbacks(self):
        @self.app.callback(
            Output('fileDropdown', 'options'),
            Input('folder-selector', 'n_clicks')
        )
        def _select_folder(n_clicks):
            if n_clicks:
                root = tkinter.Tk()
                root.withdraw()
                directory = filedialog.askdirectory()
                print('selected directory: ', directory)
                root.destroy()
                # self.eeg_visualizer.change_dir(directory)
                files_list = [{'label': str(file), 'value': str(file)}
                              for file in sorted(directory.rglob("*.edf"))]
                return files_list  # directory
