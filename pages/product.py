import dash
import pandas as pd
import numpy as np
from dash import dcc, Input, Output, callback, html, dash_table
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import dash_ag_grid as dag

dash.register_page(__name__, 
                   name= 'Product Analysis',
                   order= 4)

# Loading and reading the dataset using pandas
df = pd.read_excel('C:\\Users\\Moritus Peters\\Documents\\dash_leaflet_and_dash_aggrid\\Sample_Superstore.xls')

# calculate profit margin
df['profit_margin'] = df['Profit']/df['Sales']

# Calculate profit margin
df['Profit_Margin'] = df['Profit'] / df['Sales']

# Products still profitable with discounts
profitable_with_discount = df[(df['Discount'] > 0) & (df['Profit'] > 0)]
profitable_products = profitable_with_discount.groupby('Product Name').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Discount': 'mean',
    'Profit_Margin': 'mean'
}).reset_index()

# Products not needing discounts
products_no_discount_needed = df[(df['Discount'] == 0) & (df['Profit'] > 0)]
no_discount_needed_products = products_no_discount_needed.groupby('Product Name').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Discount': 'mean',
    'Profit_Margin': 'mean'
}).reset_index()

# Products negatively impacted by discounts
negative_impact_products = df[(df['Discount'] > 0) & (df['Profit'] < 0)]
negative_impact_products_agg = negative_impact_products.groupby('Product Name').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Discount': 'mean',
    'Profit_Margin': 'mean'
}).reset_index()

# Layout for the Dash app
layout = html.Div([
   html.H5('Welcome to the Product Analysis Page', className='text-center text-dark'),

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
        ], width=6),
        dbc.Col([
             html.Div([
                html.Label("Cities from selected Region", className='dropdown-label text-center text-dark'),
            dcc.Dropdown(
                id='cities',
                options=[],
                value=[],
                className='dropdown'
            )
            ], className='metric-container')
        ], width=6),
        
    ]),

    html.Hr(),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Visualize profit by category'),
                dbc.CardBody([
                    dcc.Graph(id='product_details', figure={},
                              style={'height': '390px'}),
                    
                ]),
            ])
    ],width=4),
            dbc.Col([
                dbc.Card([
                dbc.CardHeader('Impact of Discount on Profit'),
                    dbc.CardBody([
                        dcc.Graph(id='Discount_Impact', figure={},
                                  style={'height': '390px'}),
                    ]),
                ])
    ],width=4),
            dbc.Col([
                dbc.Card([
                dbc.CardHeader('Customer Segments', className='text-center'),
                    dbc.CardBody([
                        dcc.Graph(id='customer_segment', figure={},
                                  style={'height': '390px'}),
                    ]),
                ])
    ],width=4)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Impact of Discount on Profit'),
                    dbc.CardBody([
                        dcc.Tabs(id='tabs', value='tab-1', children=[
                            dcc.Tab(label='Profitable with Discount', value='tab-1'),
                            dcc.Tab(label='No Discount Needed', value='tab-2'),
                            dcc.Tab(label='Negative Impact Products', value='tab-3'),
                        ]),
                        html.Div(id='table'),
                    ]),
                ])
        ])
    ])
])

@callback(
    Output('cities', 'options'),
    Input('Region_checklist', 'value')
)
def set_cities_options(selected_regions):
    if not isinstance(selected_regions, list):
        selected_regions = [selected_regions]
    filtered_df = df[df['Region'].isin(selected_regions)]
    return [{'label': city, 'value': city} for city in sorted(filtered_df['City'].unique())]

@callback(
    Output('product_details', 'figure'),
    Input('Region_checklist', 'value'),
    Input('cities', 'value'),
    Input('tabs', 'value')
)
def product_detail(selected_region, selected_cities, tab):
    if not isinstance(selected_region, list):
        selected_region = [selected_region]
    if not isinstance(selected_cities, list):
        selected_cities = [selected_cities]

    filtered_df = df[df['Region'].isin(selected_region) & df['City'].isin(selected_cities)]
    
    if tab == 'tab-1':
        profit_by_category = filtered_df.groupby(['Category', 'Sub-Category']).agg({
            'Sales': 'sum', 'Profit': 'sum', 'Discount': 'mean'
        }).reset_index()
        profit_by_category['Profit_Margin'] = profit_by_category['Profit'] / profit_by_category['Sales']

        return px.bar(
            profit_by_category, x='Sub-Category', y='Profit', color='Category'
        )
    elif tab == 'tab-2':
        high_sales_low_profit = filtered_df[(filtered_df['Sales'] > filtered_df['Sales'].mean()) & (filtered_df['Profit'] < filtered_df['Profit'].mean())]
        return px.scatter(
            high_sales_low_profit, x='Sales', y='Profit', color='Sub-Category'
        )
        

@callback(
    Output('Discount_Impact', 'figure'),
    Input('Region_checklist', 'value'),
    Input('cities', 'value')
)
def Discount_Impact(selected_region, selected_cities):
    if not selected_region or not selected_cities:
        return {}
    
    if not isinstance(selected_region, list):
        selected_region = [selected_region]
    if not isinstance(selected_cities, list):
        selected_cities = [selected_cities]

    filtered_df = df[df['Region'].isin(selected_region) & df['City'].isin(selected_cities)]

    return px.scatter(filtered_df, x='Discount', y='Profit')

@callback(
    Output('customer_segment', 'figure'),
    Input('Region_checklist', 'value'),
    Input('cities', 'value')
)
def customer_segment(selected_region, selected_cities):
    if not selected_region or not selected_cities:
        return {}
    
    if not isinstance(selected_region, list):
        selected_region = [selected_region]
    if not isinstance(selected_cities, list):
        selected_cities = [selected_cities]
    
    filtered_df = df[df['Region'].isin(selected_region) & df['City'].isin(selected_cities)]
    
    customer_segments = filtered_df.groupby('Customer ID').agg({
        'Sales': 'sum', 'Quantity': 'sum', 'Profit': 'sum'
    }).reset_index()

    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_segments['Segment'] = kmeans.fit_predict(customer_segments[['Sales', 'Quantity', 'Profit']])

    return px.scatter_3d(customer_segments, x='Sales', y='Quantity', z='Profit', color='Segment')

@callback(
    Output('table', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-1':
        return dag.AgGrid(
            id='profitable-discounts',
            columnDefs=[
                {'headerName': 'Product Name', 'field': 'Product Name'},
                {'headerName': 'Sales', 'field': 'Sales', 'type': 'numericColumn'},
                {'headerName': 'Profit', 'field': 'Profit', 'type': 'numericColumn'},
                {'headerName': 'Discount', 'field': 'Discount', 'type': 'numericColumn'},
                {'headerName': 'Profit Margin', 'field': 'Profit_Margin', 'type': 'numericColumn'}
            ],
            rowData=profitable_products.to_dict('records'),
            defaultColDef={'sortable': True, 'filter': True, 'resizable': True}
        )
    elif tab == 'tab-2':
        return dag.AgGrid(
            id='no-discount-needed',
            columnDefs=[
                {'headerName': 'Product Name', 'field': 'Product Name'},
                {'headerName': 'Sales', 'field': 'Sales', 'type': 'numericColumn'},
                {'headerName': 'Profit', 'field': 'Profit', 'type': 'numericColumn'},
                {'headerName': 'Discount', 'field': 'Discount', 'type': 'numericColumn'},
                {'headerName': 'Profit Margin', 'field': 'Profit_Margin', 'type': 'numericColumn'}
            ],
            rowData=no_discount_needed_products.to_dict('records'),
            defaultColDef={'sortable': True, 'filter': True, 'resizable': True}
        )
    elif tab == 'tab-3':
        return dag.AgGrid(
            id='negative-impact-discounts',
            columnDefs=[
                {'headerName': 'Product Name', 'field': 'Product Name'},
                {'headerName': 'Sales', 'field': 'Sales', 'type': 'numericColumn'},
                {'headerName': 'Profit', 'field': 'Profit', 'type': 'numericColumn'},
                {'headerName': 'Discount', 'field': 'Discount', 'type': 'numericColumn'},
                {'headerName': 'Profit Margin', 'field': 'Profit_Margin', 'type': 'numericColumn'}
            ],
            rowData=negative_impact_products_agg.to_dict('records'),
            defaultColDef={'sortable': True, 'filter': True, 'resizable': True}
        )
