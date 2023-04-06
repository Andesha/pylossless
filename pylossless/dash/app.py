# Authors: Christian O'Reilly <christian.oreilly@sc.edu>
#          Scott Huberty <seh33@uw.edu>
# License: MIT

"""Launching point for Lossless QC Dash app."""
import dash
import dash_bootstrap_components as dbc
from pylossless.dash.qcgui import QCGUI


def get_app(fpath=None, kind="dash"):
    """Return QCR Dash.app object for Dash or JupyterDash."""
    if kind == "jupyter":
        from jupyter_dash import JupyterDash
        app = JupyterDash(__name__)
    else:
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

    QCGUI(app, fpath=fpath)
    return app


if __name__ == '__main__':
    get_app().run_server(debug=True, use_reloader=False)
