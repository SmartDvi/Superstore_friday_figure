import pandas as pd
import numpy as np
import dash 
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dcc, Input, Output, callback, html, dash_table
import dash_leaflet as dl
import dash_ag_grid as agg

dash.register_page(__name__, 
                   name= 'Overview Analysis',
                   order= 3,
        
                   )

# Loading and reading the dataset using pandas
df = pd.read_excel('C:\\Users\\Moritus Peters\\Documents\\dash_leaflet_and_dash_aggrid\\Sample_Superstore.xls')

# Dictionary of the Cities Coordinates
city_coords = {
    'New York City': (40.7128, -74.0060),
    'Los Angeles': (34.0522, -118.2437),
    'Philadelphia': (39.9526, -75.1652),
    'San Francisco': (37.7749, -122.4194),
    'Seattle': (47.6062, -122.3321),
    'Houston': (29.7604, -95.3698),
    'Chicago': (41.8781, -87.6298),
    'Columbus': (39.9612, -82.9988),
    'San Diego': (32.7157, -117.1611),
    'Springfield': (39.7817, -89.6501),
    'Dallas': (32.7767, -96.7970),
    'Jacksonville': (30.3322, -81.6557),
    'Detroit': (42.3314, -83.0458),
    'Newark': (40.7357, -74.1724),
    'Richmond': (37.5407, -77.4360),
    'Jackson': (32.2988, -90.1848),
    'Columbia': (34.0007, -81.0348),
    'Aurora': (39.7294, -104.8319),
    'Phoenix': (33.4484, -112.0740),
    'Long Beach': (33.7701, -118.1937),
    'Arlington': (32.7357, -97.1081),
    'San Antonio': (29.4241, -98.4936),
    'Toronto': (43.6510, -79.3470),
    'Louisville': (38.2527, -85.7585),
    'Miami': (25.7617, -80.1918),
    'Rochester': (43.1566, -77.6088),
    'Charlotte': (35.2271, -80.8431),
    'Henderson': (36.0395, -114.9817),
    'Lakewood': (39.7047, -105.0814),
    'Lancaster': (40.0379, -76.3055),
    'Fairfield': (38.2494, -122.0399),
    'Milwaukee': (43.0389, -87.9065),
    'Lawrence': (38.9717, -95.2353),
    'Denver': (39.7392, -104.9903),
    'Baltimore': (39.2904, -76.6122),
    'Pasadena': (34.1478, -118.1445),
    'Cleveland': (41.4993, -81.6944),
    'San Jose': (37.3382, -121.8863),
    'Fayetteville': (35.0527, -78.8784),
    'Salem': (44.9429, -123.0351),
    'Austin': (30.2672, -97.7431),
    'Atlanta': (33.7490, -84.3880),
    'Franklin': (35.9251, -86.8689),
    'Tampa': (27.9506, -82.4572),
    'Huntsville': (34.7304, -86.5861),
    'Wilmington': (34.2257, -77.9447),
    'Decatur': (39.8403, -88.9548),
    'Montreal': (45.5017, -73.5673),
    'Toledo': (41.6528, -83.5379),
    'Tucson': (32.2226, -110.9747),
    'Providence': (41.8240, -71.4128),
    'Lafayette': (30.2241, -92.0198),
    'Concord': (37.9775, -122.0311),
    'Oceanside': (33.1959, -117.3795),
    'Memphis': (35.1495, -90.0490),
    'Clinton': (32.3415, -90.3218),
    'Troy': (42.6056, -83.1499),
    'Nashville': (36.174465, -86.767960),
    'Omaha': (41.257160, -95.995102),
    'Mesa': (33.42227, -111.82264)
}

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

# Prepare data for various visualizations
region_sales = df.groupby('Region')['Sales'].sum().reset_index()
category_profit = df.groupby('Category')['Profit'].sum().reset_index()
top_products = df.groupby('Product Name')['Sales'].sum().nlargest(10).reset_index()
segment_performance = df.groupby('Segment').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
clv = df.groupby('Customer ID')['Customer Lifetime Value'].sum().reset_index()
churn_rate = df.groupby(df['Order Date'].dt.to_period('M'))['Churn'].mean().reset_index()
feedback = df.groupby('Sentiment').size().reset_index(name='Counts')
campaign_performance = df.groupby('Campaign')['Sales'].sum().reset_index()
inventory = df.groupby('Product Name')['Quantity'].sum().reset_index()

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

# Perform necessary calculations
total_unit_sold = df['Quantity'].sum()
total_product = df['Product Name'].nunique()
total_city = df['City'].nunique()
total_sales = df['Sales'].sum()
total_profit = df['Profit'].sum()
profit_margin = total_profit / total_sales


# Layout for the Dash app
layout = html.Div([
    html.H5('Welcome to the Overview Analysis Page', className='text-center text-dark'),
    
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H6('Total Unit Sold', className='text-dark'),
                    html.Div(id='card_total_unity_sold'),
                ])
            )
        ),

        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H6('Total Product', className='text-dark'),
                    html.Div(id='total_product'),
                ])
            ),
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H6('Total City', className='text-dark'),
                    html.Div(id='Total_city'),
                ])
            ),
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H6('Total Sales', className='text-dark'),
                    html.Div(id='total_Sales'),
                ])
            ),
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H6('Total Profit', className='text-dark'),
                    html.Div(id='total_profit'),
                ])
            ),
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H6('Profit Margin', className='text-dark text-center'),
                    html.Div(id='Profit_Margin'),
                ])
            ),
        ),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Label("Country/Region", className='dropdown-label text-center, text-dark'),
                dbc.Checklist(
                    id='country_checklist',
                    options=[{'label': country, 'value': country} for country in sorted(df['Country/Region'].unique())],
                    value=[],
                    inline=True,
                    className='text-center px-2 text-dark'
                )
            ], className='metric-container')
        ], width=6),
        dbc.Col([
            dcc.DatePickerSingle(
                id='date_picker',
                min_date_allowed=df['Order Date'].min(),
                max_date_allowed=df['Order Date'].max(),
                initial_visible_month=df['Order Date'].max(),
                date=df['Order Date'].max(),
                display_format='MMMM D, YYYY',
                className='text-center',
                placeholder='Select a Date'
            )
        ], width=6),
        
    ]),

    html.Hr(),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Shipping Time and Regional Analysis'),
                dbc.CardBody([
                   # html.H5('Shipping Time and Regional Analysis', className='text-center'),
                    dcc.Graph(id='ShippingTime_Regional', figure={},
                              style={'height': '250px'}),
                    
                ]),
                #dbc.CardFooter('Understanding the correlation between shipping time and profit helps identify inefficiencies in the supply chain. Regions with longer shipping times and lower profitability can be targeted for process improvements.')
            ])
    ],width=4),
            dbc.Col([
                dbc.Card([
                dbc.CardHeader('Product Contribution to Business Growth'),
                    dbc.CardBody([
                       # html.H5('Product Contribution to Business Growth', className='text-center'),
                        dcc.Graph(id='Product_Contribution', figure={},
                                   style={"height": "100%", "width": "100%", "padding": "0"}),
                    ]),
                    #dbc.CardFooter('Understanding the correlation between shipping time and profit helps identify inefficiencies in the supply chain. Regions with longer shipping times and lower profitability can be targeted for process improvements.')
                ])
    ],width=4),
            dbc.Col([
                dbc.Card([
                dbc.CardHeader('Day of Month Sales Analysis', className='text-center'),
                    dbc.CardBody([
                        #html.H5('Day of Month Sales Analysis', className='text-center'),
                        dcc.Graph(id='Day_Month_Sales', figure={},
                                   style={"height": "100%", "width": "100%", "padding": "0"}),
                    ]),
                   # dbc.CardFooter('Understanding the correlation between shipping time and profit helps identify inefficiencies in the supply chain. Regions with longer shipping times and lower profitability can be targeted for process improvements.')
                ])

    ],width=4)
    ]),

    dbc.Row([
        dbc.Col([
                dbc.Card([
                dbc.CardHeader('Customer Orders Table details', className='text-center'),
                    dbc.CardBody([
                        #html.H5('Customer Orders Table details', className='text-center'),
                        html.Div(id='Customer_Orders_Table',
                            style={"height": "100%", "width": "100%", "padding": "0"}),
                    ]),
                    #dbc.CardFooter('Understanding the correlation between shipping time and profit helps identify inefficiencies in the supply chain. Regions with longer shipping times and lower profitability can be targeted for process improvements.')
                ])
        ],width=5),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Sales Distribution by City'),
                dbc.CardBody([
                    dl.Map(
                            style={'width': '100%', 'height': '250px'},
                            center=[37.0902, -95.7129],
                            zoom=4,
                            children=[
                                dl.TileLayer(),
                                dl.LayerGroup(id='city_markers')
                            ]
                        )
                    ])
                ])
            ], width=7)
    ])        



])

# developinng the callback 
@callback(Output('ShippingTime_Regional', 'figure'),
          Input('country_checklist', 'value'))

def shipping_analysis(selected_country):
    if not selected_country:
        return {}
    
    
    filtered_df = df[df['Country/Region'].isin(selected_country)]
    # Group by Region and Segment, calculate average shipping time and profit
    shipping_profit = df.groupby(['Region', 'Segment']).agg({'Shipping Time': 'mean', 'Profit': 'sum'}).reset_index()
    # Visualize shipping time and profitability relationship
    fig1 = px.scatter(shipping_profit, x='Shipping Time', y='Profit', color='Region',
                    size='Profit', hover_name='Segment')
    fig1.update_layout(xaxis_title='Average Shipping Time (days)', yaxis_title='Total Profit',
                    coloraxis_colorbar_title='Region', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    #labels={'Shipping Time': 'Average Shipping Time (days)', 'Profit': 'Total Profit'}
                    )
    
    return fig1


# developinng the callback 
@callback(Output('Product_Contribution', 'figure'),
          Input('country_checklist', 'value'))

def product_contributing(selected_country):
    if not selected_country:
        return {}

    filtered_df = df[df['Country/Region'].isin(selected_country)]    
    # Group by Category and calculate total profit
    category_profit = filtered_df.groupby('Category')['Profit'].sum().reset_index()

        # Visualize profitability by category
    fig2 = px.pie(category_profit, values='Profit', names='Category')
    return fig2


@callback(Output('Day_Month_Sales', 'figure'),
          Input('country_checklist', 'value'))

def Day_Month_Sales_Analysis(selected_country):
    if not selected_country:
        return {}

    filtered_df = df[df['Country/Region'].isin(selected_country)]
    # Grouping by Order Day of Month and calculating sum of Sales
    day_of_month_sales = filtered_df.groupby('Order Day of Month')['Sales'].sum().reset_index()

    # Plotting the line chart
    fig_day_of_month = px.line(day_of_month_sales, x='Order Day of Month', y='Sales', 
                            labels={'Order Day of Month': 'Day of Month', 'Sales': 'Total Sales ($)'},
                            color_discrete_sequence=['#d62728'])

    return fig_day_of_month


@callback(Output('Customer_Orders_Table', 'children'),
          Input('country_checklist', 'value'))
def Customer_Orders_Table(selected_country):
    if not selected_country:
        return {}

    filtered_df = df[df['Country/Region'].isin(selected_country)]
    #   # Aggregate the data by Customer Name and other columns
    aggregated_data = filtered_df.groupby('Customer Name').agg(
        Total_Sales=('Sales', 'sum'),
        Total_Profit=('Profit', 'sum'),
        Total_Orders=('Customer Name', 'size'),
        City=('City', lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]),  # Mode or first value
        Segment=('Segment', lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]),  # Mode or first value
        Average_Discount=('Discount', 'mean'),
        Total_Quantity=('Quantity', 'sum'),
        Category=('Category', lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])  # Mode or first value
    ).reset_index()

    # Sort the aggregated data by Total Sales (or Total Profit) and get the top 20 customers
    top_customers = aggregated_data.sort_values(by='Total_Sales', ascending=False).head(20)

    # Define the columns to display
    columns_to_display = ['Customer Name', 'City', 'Segment', 'Category', 'Total_Sales', 'Total_Profit', 'Total_Orders', 'Average_Discount', 'Total_Quantity']

    # Create table using dash_ag_grid
    customer_orders_table = agg.AgGrid(
        columnDefs=[{'headerName': col, 'field': col} for col in columns_to_display],
        rowData=top_customers.to_dict('records'),
        defaultColDef={'sortable': True, 'filter': True, 'resizable': True},
        style={'height': '250px', 'width': '100%'}
    )
    return customer_orders_table

# Callback for Sales Distribution by City
@callback(Output('city_markers', 'children'),
          Input('country_checklist', 'value'))
def update_markers(selected_countries):
    if not selected_countries:
        return []

    filtered_df = df[df['Country/Region'].isin(selected_countries)]
    cities = filtered_df['City'].unique()
    markers = []
    
    for city in cities:
        if city in city_coords:
            lat, lon = city_coords[city]
            total_sales = filtered_df[filtered_df['City'] == city]['Sales'].sum()
            markers.append(dl.Marker(
                position=(lat, lon),
                children=dl.Tooltip(f"{city}: ${total_sales:,.2f}")
            ))

    return markers



# developing the key metrics
@callback(Output('card_total_unity_sold', 'children'),
          Output('total_product', 'children'),
           Output('Total_city', 'children'),
            Output('total_Sales', 'children'),
            Output('total_profit', 'children'),
            Output('Profit_Margin', 'children'),
          Input('country_checklist', 'value'))

def key_metrics(selected_countries):
    if not selected_countries:
       return [total_unit_sold, total_product, total_city, total_sales, total_profit, profit_margin]
    
    filtered_df = df[df['Country/Region'].isin(selected_countries)]
    #filtered_df = df[df['Order Date'] == pd.to_datetime(date)]
    
    card_total_units_sold = html.H4(f'{filtered_df["Quantity"].sum():,}', className='card-title text-dark')
    total_products = html.H4(f'{len(filtered_df["Product Name"].unique()):,}', className='card-title text-dark')
    total_cities = html.H4(f'{len(filtered_df["City"].unique()):,}', className='card-title text-dark')
    total_sales = html.H4(f'${filtered_df["Sales"].sum():,.2f}', className='card-title text-dark')
    total_profit = html.H4(f'${filtered_df["Profit"].sum():,.2f}', className='card-title text-dark')


    return  [card_total_units_sold, total_products,  total_cities, total_sales, total_profit, overall_profit_margin]



# Footer
    html.Div([
        html.P('Copyright © 2024 - Business Insights Dashboard'),
    ], style={'textAlign': 'center', 'marginTop': '20px'})