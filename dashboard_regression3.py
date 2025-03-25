import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from sklearn import  metrics
import numpy as np
import joblib 


#Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Load data
df = pd.read_csv('forecast_data.csv')
df['Date'] = pd.to_datetime (df['Date']) # create a new column 'data time' of datetime type
df2=df.iloc[:,1:5]
X2=df2.values
fig1 = px.line(df, x="Date", y=df.columns[1:4])# Creates a figure with the raw data


df_real = pd.read_csv('real_results.csv')
y2=df_real['Power (kW) [Y]'].values

#Load and run LR model
LR_model2 = joblib.load('LR_model.sav')
y2_pred_LR = LR_model2.predict(X2)
#y2_pred_LR=y2

#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y2,y2_pred_LR) 
MBE_LR=np.mean(y2-y2_pred_LR)
MSE_LR=metrics.mean_squared_error(y2,y2_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y2,y2_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y2)
NMBE_LR=MBE_LR/np.mean(y2)

#Load RF model
RF_model2 = joblib.load('RF_model.sav')
y2_pred_RF = RF_model2.predict(X2)
#y2_pred_RF=y2

#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y2,y2_pred_RF)
MBE_RF=np.mean(y2-y2_pred_RF) 
MSE_RF=metrics.mean_squared_error(y2,y2_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y2,y2_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y2)
NMBE_RF=MBE_RF/np.mean(y2)

# Create data frames with predictin results and error metrics 
d = {'Methods': ['Linear Regression','Random Forest'], 'MAE': [MAE_LR, MAE_RF],'MBE': [MBE_LR, MBE_RF], 'MSE': [MSE_LR, MSE_RF], 'RMSE': [RMSE_LR, RMSE_RF],'cvMSE': [cvRMSE_LR, cvRMSE_RF],'NMBE': [NMBE_LR, NMBE_RF]}
df_metrics = pd.DataFrame(data=d)
d={'Date':df_real['Date'].values, 'LinearRegression': y2_pred_LR,'RandomForest': y2_pred_RF}
df_forecast=pd.DataFrame(data=d)

# merge real and forecast results and creates a figure with it
df_results=pd.merge(df_real,df_forecast, on='Date')

fig2 = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:4])

# Define auxiliary functions
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server=app.server

app.layout = html.Div([
    html.H1('IST Energy Forecast tool (kWh)'),
    html.P('Representing Data, Forecasting and error metrics for November 2017 using three tabs'),
    html.P(' '),
    html.Label('Regression Method'),
    dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'Linear', 'value': 0},
            {'label': 'RF', 'value': 1},
            {'label': 'All', 'value': 2},
            ],
        value=0,
        style={'display': 'block'},
    ),
    html.P(''),

    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
        dcc.Tab(label='Error Metrics', value='tab-3'),
    ]),
    html.Div(id='tabs-content')
])



@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'),
              Input('dropdown', 'value'))


# def dropdown_show (tab):
#     if tab == 'tab-1':
#         return {'display': 'none'}
#     elif tab == 'tab-2':
#         return {'display': 'block'}
#     elif tab == 'tab-3':
#         return  {'display': 'block'}

    

def render_content(tab,value_dropdown):
    if tab == 'tab-1':
        return html.Div([
            html.H4('IST Raw Data'),
            dcc.Graph(
                id='yearly-data',
                figure=fig1,
            ),
            
        ])
    elif tab == 'tab-2':
        
        #figure=fig2,
        if value_dropdown == 0:
            fig2=px.line(df_results,x=df_results.columns[0],y=df_results.columns[[1,3]])
        elif value_dropdown == 1:
            fig2=px.line(df_results,x=df_results.columns[0],y=df_results.columns[[2,3]])
        elif value_dropdown == 2:
                fig2=px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:4])
        return html.Div([
            html.H4('IST Electricity Forecast (kWh)'),
            dcc.Graph(
                id='yearly-data',
                figure=fig2
                ),
            
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H4('IST Electricity Forecast Error Metrics'),
                        generate_table(df_metrics)
        ])


if __name__ == '__main__':
    app.run_server()
