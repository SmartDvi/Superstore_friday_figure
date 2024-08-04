import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import pages  # Assuming pages is a directory with __init__.py
import assets  # Assuming assets is a directory with __init__.py

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the navbar at the top
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink(page['name'], href=page['path']))
        for page in dash.page_registry.values()
    ],
    brand="Super Store Data on Plotly Weekend Project",
    brand_href="/introduction",
    color="primary",
    dark=True,
    className="mb-4 text-center"
)

# Define the sidebar with a dropdown and a full calendar


app.layout = dbc.Container([
    navbar,
    dbc.Row([
        dbc.Col(
            [
                dash.page_container
            ]
        )
    ])
], fluid=True)

if __name__ == "__main__":
    app.run(debug=True, port=6090)
