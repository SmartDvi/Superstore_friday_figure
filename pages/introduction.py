import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

dash.register_page(
    __name__,
    name='Introduction',
    order=0,
)

# Layout for the Introduction page
layout = html.Div([
    html.H1('Welcome to the Comprehensive Business Analytics Dashboard', className='text-center text-primary mb-4'),
    
    html.Div([
        html.H3('Introduction', className='text-center text-dark mb-3'),
        html.P(
            "Welcome to the Comprehensive Business Analytics Dashboard, designed to provide in-depth insights into various aspects of your business. "
            "This application offers a suite of interactive tools to help you analyze data, track performance, and make data-driven decisions. "
            "Each page of the dashboard is dedicated to a specific area of analysis, allowing you to explore your data from different perspectives.",
            className='text-center mb-4'
        ),
    ], className='container'),

    html.Div([
        html.H4('Overview of the Dashboard Pages', className='text-center text-dark mb-3'),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Data Overview Page'),
                    dbc.CardBody([
                        html.P(
                            "The Data Overview Page provides a comprehensive summary of your dataset. Explore the data's structure, key metrics, and essential statistics to understand the foundational elements of your analysis.",
                            className='text-dark'
                        )
                    ]),
                ], className='mb-4')
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Product Page'),
                    dbc.CardBody([
                        html.P(
                            "Analyze product performance with the Product Page. Evaluate metrics such as sales volume and profit margins by product categories and sub-categories to identify trends and opportunities for growth.",
                            className='text-dark'
                        )
                    ]),
                ], className='mb-4')
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Time Analysis Page'),
                    dbc.CardBody([
                        html.P(
                            "The Time Analysis Page lets you explore time-related patterns and trends in sales and profit. Use interactive filters to examine data across different time periods, such as monthly, quarterly, and annually.",
                            className='text-dark'
                        )
                    ]),
                ], className='mb-4')
            ], width=4),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Regression Page'),
                    dbc.CardBody([
                        html.P(
                            "Perform advanced regression analysis on the Regression Page to predict future outcomes and understand variable relationships. Analyze regression results and model performance metrics to forecast business trends.",
                            className='text-dark'
                        )
                    ]),
                ], className='mb-4')
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Classification Page'),
                    dbc.CardBody([
                        html.P(
                            "The Classification Page provides insights into categorical outcomes with classification analysis. Evaluate performance metrics and identify patterns in classified data to make informed decisions.",
                            className='text-dark'
                        )
                    ]),
                ], className='mb-4')
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Recommendation Page'),
                    dbc.CardBody([
                        html.P(
                            "Discover actionable insights with the Recommendation Page. Leverage recommendation systems to suggest products, actions, or strategies based on historical data and user preferences.",
                            className='text-dark'
                        )
                    ]),
                ], className='mb-4')
            ], width=4),
        ]),
    ], className='container'),

    html.Div([
        html.H4('Getting Started', className='text-center text-dark mb-3'),
        html.P(
            "To get started, navigate through the dashboard pages using the menu provided. Each page features interactive elements and visualizations to help you explore and analyze the data effectively. "
            "Should you have any questions or need further assistance, please refer to the help section or contact support.",
            className='text-center mb-4'
        ),
    ], className='container'),

    html.Div([
        html.Hr(),
        html.P(
            "Explore the features of this dashboard to gain valuable insights and make data-driven decisions for your business.",
            className='text-center text-dark'
        ),
    ], className='container'),
])
