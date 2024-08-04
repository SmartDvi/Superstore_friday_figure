import pandas as pd
import numpy as np
import dash
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dcc, Input, Output, callback, html

dash.register_page(
    __name__,
    name='Time Analysis',
    order=5,
)

# Loading and reading the dataset using pandas
df = pd.read_excel('C:\\Users\\Moritus Peters\\Documents\\dash_leaflet_and_dash_aggrid\\Sample_Superstore.xls')

# Formatting the date columns to datetime format to fetch insights
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Calculating additional time metrics
df['Shipping Time'] = (df['Ship Date'] - df['Order Date']).dt.days
df['Customer Lifetime Value'] = df['Sales'] * df['Quantity']

# Develop a simplified churn indicator
df['Churn'] = (df['Order Date'].diff().dt.days > 30).astype(int)

# Developing a simplified sentiment column
df['Sentiment'] = ['Positive' if x % 2 == 0 else 'Negative' for x in range(len(df))]

# Generate sample campaign data matching the length of the Dataframe
campaign_options = ['Campaign 1', 'Campaign 2', 'Campaign 3']
campaign_data = np.random.choice(campaign_options, size=len(df))
df['Campaign'] = campaign_data

# Convert 'Order Date' and 'Ship Date' columns to datetime format
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Extract day of week (Monday=0, Sunday=6)
df['Order Day of Week'] = df['Order Date'].dt.dayofweek
df['Ship Day of Week'] = df['Ship Date'].dt.dayofweek

# Extract month (January=1, December=12)
df['Order Month'] = df['Order Date'].dt.month_name()
df['Ship Month'] = df['Ship Date'].dt.month_name()

# Extract quarter (Q1=1, Q4=4)
df['Order Quarter'] = df['Order Date'].dt.quarter
df['Ship Quarter'] = df['Ship Date'].dt.quarter

# Extract day of month
df['Order Day of Month'] = df['Order Date'].dt.day
df['Ship Day of Month'] = df['Ship Date'].dt.day_name()

# Extract quarter of year
df['Order Quarter of Year'] = df['Order Date'].dt.month.apply(lambda x: (x-1)//3 + 1)
df['Ship Quarter of Year'] = df['Ship Date'].dt.month.apply(lambda x: (x-1)//3 + 1)

# Extract year
df['Order Year'] = df['Order Date'].dt.year
df['Ship Year'] = df['Ship Date'].dt.year

# Calculate Profit Margin for each order
df["Profit Margin"] = df["Profit"] / df["Sales"]

# Calculate total sales, total profit, and overall profit margin
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
overall_profit_margin = total_profit / total_sales

# Layout for the Dash app
layout = html.Div([
    html.H5('Welcome to the Period Analysis Page', className='text-center text-dark'),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.Label("Region", className='dropdown-label text-center text-dark'),
                dbc.Checklist(
                    id='Region_checklist',
                    options=[{'label': region, 'value': region} for region in sorted(df['Region'].unique())],
                    value=[sorted(df['Region'].unique())[0]],
                    inline=True,
                    className='text-center px-2 text-dark'
                )
            ], className='metric-container')
        ], width=4),
        dbc.Col([
            html.Div([
                html.Label("City dropdown of the selected Region", className='dropdown-label text-center text-dark'),
                dcc.Dropdown(
                    id='cities_dp',
                    options=[],
                    value=[],
                    multi = True,
                    className='dropdown'
                )
            ], className='metric-container')
        ], width=3),

        dbc.Col([
            html.Div([
                html.Label("Category", className='dropdown-label text-center, text-dark'),
                dbc.Checklist(
                    id='Category_checklist',
                    options=[{'label': Category, 'value': Category} for Category in sorted(df['Category'].unique())],
                    value=[],
                    inline=True,
                    className='text-center px-2 text-dark'
                )
            ], className='metric-container')
        ], width=4),
    ]),

    html.Hr(),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Annual Sales Analysis'),
                dbc.CardBody([
                    dcc.Graph(id='Annual_Sales', figure={},
                              style={'height': '250px'}),
                ]),
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Quarterly Sales Analysis', className='text-center'),
                dbc.CardBody([
                    dcc.Graph(id='Quarterly_Sales', figure={},
                              style={'height': '250px'}),
                ]),
            ])
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Monthly Sales and Profit Analysis'),
                dbc.CardBody([
                    dcc.Graph(id='Monthly_Sales_Profit', figure={},
                              style={'height': '1000px'}),
                ]),
            ])
        ], width=12),
    ])
])

# Update city options based on selected regions
@callback(
    Output('cities_dp', 'options'),
    Input('Region_checklist', 'value'),
    suppress_callback_exceptions=True

)
def set_cities_options(selected_regions):
    if not isinstance(selected_regions, list):
        selected_regions = [selected_regions]
    filtered_df = df[df['Region'].isin(selected_regions)]
    return [{'label': city, 'value': city} for city in sorted(filtered_df['City'].unique())]

# Update charts based on selected categories, regions, and cities
@callback(
    Output('Monthly_Sales_Profit', 'figure'),
    Output('Annual_Sales', 'figure'),
    Output('Quarterly_Sales', 'figure'),
    Input('Category_checklist', 'value'),
    Input('Region_checklist', 'value'),
    Input('cities_dp', 'value'),
    Input('Monthly_Sales_Profit', 'clickData'),
    Input('Annual_Sales', 'clickData'),
    Input('Quarterly_Sales', 'clickData')
    
)
def update_charts(selected_categories, selected_regions, selected_cities, monthly_click, annual_click, quarterly_click):
    if not selected_categories or not selected_regions:
        return {}, {}, {}

    filtered_df = df[
        (df['Category'].isin(selected_categories)) &
        (df['Region'].isin(selected_regions))
    ]

    if selected_cities:
        filtered_df = filtered_df[filtered_df['City'].isin(selected_cities)]

    # Check for click events to further filter data
    if monthly_click:
        month_clicked = monthly_click['points'][0]['x']
        filtered_df = filtered_df[filtered_df['Order Month'] == month_clicked]
    if annual_click:
        year_clicked = annual_click['points'][0]['x']
        filtered_df = filtered_df[filtered_df['Order Year'] == year_clicked]
    if quarterly_click:
        quarter_clicked = quarterly_click['points'][0]['x']
        filtered_df = filtered_df[filtered_df['Order Quarter'] == int(quarter_clicked.split()[-1])]

    # Monthly Sales and Profit Analysis
    monthly_sales_profit = filtered_df.groupby('Order Month')[['Sales', 'Profit']].sum().reset_index()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_sales_profit['Order Month'] = pd.Categorical(monthly_sales_profit['Order Month'], categories=month_order, ordered=True)
    monthly_sales_profit.sort_values(by='Order Month', inplace=True)

    bar_trace = go.Bar(
        x=monthly_sales_profit['Order Month'],
        y=monthly_sales_profit['Sales'],
        name='Sales',
        marker_color='#1f77b4'
    )

    line_trace = go.Scatter(
        x=monthly_sales_profit['Order Month'],
        y=monthly_sales_profit['Profit'],
        name='Profit',
        mode='lines+markers',
        line=dict(color='firebrick', width=2),
        marker=dict(size=8)
    )

    monthly_fig = go.Figure()
    monthly_fig.add_trace(bar_trace)
    monthly_fig.add_trace(line_trace)
    monthly_fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Total Sales ($) and Profit ($)',
        xaxis={'categoryorder': 'array', 'categoryarray': month_order},
        yaxis=dict(tickformat="$,.0f")
    )

    # Annual Sales Analysis
    annual_sales = filtered_df.groupby('Order Year')['Sales'].sum().reset_index()
    annual_fig = px.bar(annual_sales, x='Order Year', y='Sales')
    annual_fig.update_layout(yaxis=dict(tickformat="$,.0f"))

    # Quarterly Sales Analysis
    quarterly_sales = filtered_df.groupby('Order Quarter')['Sales'].sum().reset_index()
    quarterly_fig = px.bar(quarterly_sales, x='Order Quarter', y='Sales')
    quarterly_fig.update_layout(yaxis=dict(tickformat="$,.0f"))

    return monthly_fig, annual_fig, quarterly_fig
