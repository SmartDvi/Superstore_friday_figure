import pandas as pd
import numpy as np
import dash_ag_grid as agg
from dash import dcc, Input, Output, callback, html
import dash
import dash_bootstrap_components as dbc 

dash.register_page(__name__, 
                   name='Dataset Overview',
                   order=2)

# Loading and reading the dataset using pandas
df = pd.read_excel('C:\\Users\\Moritus Peters\\Documents\\dash_leaflet_and_dash_aggrid\\Sample_Superstore.xls')

# Formatting the date columns to datetime format
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Calculating additional time metrics
df['Shipping Time'] = (df['Ship Date'] - df['Order Date']).dt.days
df['Customer Lifetime Value'] = df['Sales'] * df['Quantity']
df['Churn'] = (df['Order Date'].diff().dt.days > 30).astype(int)

# Adding sentiment and campaign data
df['Sentiment'] = ['Positive' if x % 2 == 0 else 'Negative' for x in range(len(df))]
campaign_options = ['Campaign 1', 'Campaign 2', 'Campaign 3']
df['Campaign'] = np.random.choice(campaign_options, size=len(df))

# Retrieve column names
columns = df.columns

# Generate column definitions for AG Grid
column_defs = []
for col in columns:
    column_type = 'numericColumn' if pd.api.types.is_numeric_dtype(df[col]) else \
                  'dateColumn' if pd.api.types.is_datetime64_any_dtype(df[col]) else \
                  'textColumn'
    
    column_defs.append({
        'headerName': col,
        'field': col,
        'type': column_type
    })

# Define the layout of the app
layout = html.Div([
    html.H5("Superstore Dataset Overview", className='text-center'),
    html.H6('You can Navigate/Filter through the dataset by using the filter and sort icon in each column', className= 'text-center'),
    agg.AgGrid(
        id='data-table',
        columnDefs=column_defs,  # Use columnDefs instead of columns
        rowData=df.to_dict('records'),
        style={'height': '400px', 'width': '100%'},  # Use 'style' instead of 'style_table'
        className='ag-theme-alpine',
        resetColumnState=False,
        exportDataAsCsv=False,
        selectAll=False,
        deselectAll=False,
        enableEnterpriseModules=False,
        updateColumnState=False,
        persisted_props=['selectedRows'],
        persistence_type='local',
        suppressDragLeaveHidesColumns=True,
        dangerously_allow_code=False,
        rowModelType='clientSide',
        defaultColDef={'sortable': True, 'filter': True, 'resizable': True}
    )
])