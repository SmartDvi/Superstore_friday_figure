import pandas as pd
import numpy as np
import dash 
import shap
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dcc, Input, Output, callback, html, dash_table
import dash_leaflet as dl
import dash_ag_grid as agg
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

dash.register_page(__name__, name='Predictive Analysis', order=6)

# Loading and reading the dataset using pandas
df = pd.read_excel('C:\\Users\\Moritus Peters\\Documents\\dash_leaflet_and_dash_aggrid\\Sample_Superstore.xls')

# Renaming the columns for the regression process (removing spaces)
df.columns = [col.replace(' ', '_') for col in df.columns]

# Convert dates to datetime objects
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['Ship_Date'] = pd.to_datetime(df['Ship_Date'])

# Calculating additional time metrics
df['Shipping_Time'] = (df['Ship_Date'] - df['Order_Date']).dt.days
df['Order_month'] = df['Order_Date'].dt.month_name()
df['Customer_Lifetime_Value'] = df['Sales'] * df['Quantity']

# Develop a simplified churn indicator
df['Churn'] = (df['Order_Date'].diff().dt.days > 30).astype(int)

# Developing a simplified sentiment column
df['Sentiment'] = ['Positive' if x % 2 == 0 else 'Negative' for x in range(len(df))]

# Generate sample campaign data matching the length of the DataFrame
campaign_options = ['Campaign 1', 'Campaign 2', 'Campaign 3']
campaign_data = np.random.choice(campaign_options, size=len(df))
df['Campaign'] = campaign_data

# Function to create density plot
def create_density_plot(df, column):
    density_fig = px.density_contour(df, x=column, title=f'Density Plot of {column}')
    density_fig.update_traces(contours_coloring="fill", contours_showlabels=True)
    density_fig.update_layout(xaxis_title=column, yaxis_title='Density')
    return density_fig

layout = html.Div([
    html.H2('Regression Analysis'),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Label("Month Checklist", className='dropdown-label text'),
                dbc.Checklist(
                    id='month_checklist',
                    options=[{'label': str(month), 'value': month} for month in sorted(df['Order_month'].unique())],
                    value=[],
                    inline=True,
                    className='text-center px-2'
                )
            ], className='metric-container')
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5('Actual vs Predicted Visualization', className='text-center'),
                    dcc.Graph(id='actual_vs_predicted', figure={}),
                ])
            ), width=4
        ),
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H5('Residuals vs Predicted', className='text-center'),
                    dcc.Graph(id='residuals_vs_Predicted', figure={}),
                ])
            ),
        ], width=4),
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H5('Top 10 Feature Importances', className='text-center'),
                    dcc.Graph(id='Feature_Importances', figure={}),
                ])
            ),
        ], width=4),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    html.H5('Distribution of residuals', className='text-center'),
                    dcc.Graph(id='Distribution_residuals', figure={}),
                ])
            ),
        ], width=8),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5('Model Performance Test Table', className='text-center'),
                    html.Div(id='model_performance_test_table')
                ])
            ), width=4
        ),
    ]),

    dbc.Row([
        dbc.Col([
            html.Label('Select Column for Density Plot', className='dropdown-label text'),
            dcc.Dropdown(
                id='density-plot-dropdown',
                options=[{'label': col, 'value': col} for col in df.columns if df[col].dtype in [np.float64, np.int64]],
                value='Profit',
                className='dropdown'
            ),
            dcc.Graph(id='density-plot', figure={})
        ], width=12)
    ])
])

@callback(
    Output('actual_vs_predicted', 'figure'),
    Output('residuals_vs_Predicted', 'figure'),
    Output('Feature_Importances', 'figure'),
    Output('Distribution_residuals', 'figure'),
    Output('model_performance_test_table', 'children'),
    Output('density-plot', 'figure'),
    Input('month_checklist', 'value'),
    Input('density-plot-dropdown', 'value')
)
def prediction_model(selected_month, selected_column):
    if not selected_month:
        return {}, {}, {}, {}, {}, {}

    filtered_df = df[df['Order_month'].isin(selected_month)]
    filtered_df.reset_index(drop=True, inplace=True)

    filtered_df_encode = pd.get_dummies(filtered_df, columns=['Ship_Mode', 'Segment', 'Country/Region', 'Region', 'Category', 'Sub-Category', 'Sentiment', 'Campaign'])
    dep_var = 'Profit'
    features = ['Sales', 'Quantity', 'Discount', 'Shipping_Time', 'Customer_Lifetime_Value', 'Churn'] + \
               [col for col in filtered_df_encode.columns if col.startswith(('Ship_Mode_', 'Segment_', 'Country/Region_', 'Region_', 'Category_', 'Sub-Category_', 'Sentiment_', 'Campaign_'))]

    X = filtered_df_encode[features]
    y = filtered_df_encode[dep_var]

    # Splitting and training the dataset
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

    param_grid = {
        'gbr__learning_rate': [0.01, 0.1, 0.05],
        'gbr__n_estimators': [100, 200, 300],
        'gbr__max_depth': [3, 4, 5]
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gbr', GradientBoostingRegressor())
    ])

    # Initialize GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')
    
    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Validation metrics
    preds = best_model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    r2 = r2_score(y_valid, preds)
    mae = mean_absolute_error(y_valid, preds)

    explainer = shap.TreeExplainer(best_model.named_steps['gbr'])
    shap_values = explainer.shap_values(X_valid)

    # Convert SHAP values for plotly
    shap_values_df = pd.DataFrame(shap_values, columns=features)
    shap_values_df = shap_values_df.abs().mean().sort_values(ascending=False).reset_index()
    shap_values_df.columns = ['Feature', 'Mean Absolute SHAP Value']

    # Actual vs. Predicted values plot
    preds_df = pd.DataFrame({'Actual': y_valid, 'Predicted': preds})
    actual_vs_predicted_fig = px.scatter(preds_df, x='Actual', y='Predicted', title='Actual vs Predicted Profit',
                                         labels={'Actual': 'Actual Profit', 'Predicted': 'Predicted Profit'},
                                         trendline='ols')

    # Residuals plot
    residuals = y_valid - preds
    residuals_df = pd.DataFrame({'Predicted': preds, 'Residuals': residuals})
    residuals_fig = px.scatter(residuals_df, x='Predicted', y='Residuals', title='Residuals vs Predicted',
                               labels={'Predicted': 'Predicted Profit', 'Residuals': 'Residuals'},
                               trendline='ols')

    # Distribution of Residuals
    residuals_series = pd.Series(residuals, name='Residuals')
    distribution_residuals_fig = px.histogram(residuals_series, x='Residuals', title='Distribution of Residuals')

    # Feature Importances plot
    top_features = shap_values_df.head(10)
    feature_importances_fig = px.bar(top_features, x='Mean Absolute SHAP Value', y='Feature', orientation='h',
                                     title='Top 10 Feature Importances')

    # Model Performance Table
    model_performance = pd.DataFrame({
        'Metric': ['RMSE', 'R-squared', 'MAE'],
        'Value': [rmse, r2, mae]
    })

    model_performance_table = dash_table.DataTable(
        data=model_performance.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in model_performance.columns],
        style_cell={'textAlign': 'center'},
        style_table={'margin': 'auto'}
    )

    # Density Plot
    density_plot_fig = create_density_plot(filtered_df_encode, selected_column)

    return actual_vs_predicted_fig, residuals_fig, feature_importances_fig, distribution_residuals_fig, model_performance_table, density_plot_fig