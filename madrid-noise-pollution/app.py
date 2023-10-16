# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:40:14 2023

@author: Alejandro Moreno
"""
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset
from dash import Dash, dcc, html, Input, Output, callback, State, no_update
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose

from pathlib import Path
THIS_FOLDER = Path(__file__).parent.resolve()
my_file1 = THIS_FOLDER / "data/EstacionesMedidaControlAcustico.csv"
my_file2 = THIS_FOLDER / "data/Ruido_diario_acumulado.csv"

# Load monitoring noise stations data
#df_stations = pd.read_csv('https://datos.madrid.es/egob/catalogo/211346-1-estaciones-acusticas.csv', sep=';' ,encoding = 'utf-8')
df_stations = pd.read_csv(my_file1, sep=';' ,encoding = 'utf-8')
# Load noise pollution data
#df_pollution = pd.read_csv('https://datos.madrid.es/egob/catalogo/215885-10749127-contaminacion-ruido.csv', sep=';', encoding = 'utf-8')
df_pollution = pd.read_csv(my_file2, sep=';', encoding = 'utf-8')

##############################
# PREPROCESSING DATA         #
##############################

# STATIONS DATA

# Clean and fix stations identification
df_stations['NMT'] = df_stations['NMT'].str.replace('RF ', '', regex=True).astype(np.int64)

# Select relevant data
df_stations = df_stations[['NOMBRE','LONGITUD','LATITUD','NMT']]

# Rename columns names
df_stations = df_stations.rename(columns={'NOMBRE': 'id','LONGITUD':'longitude','LATITUD':'latitude'})

# Sort by id
df_stations.sort_values(by='id',inplace=True)


# POLLUTION TIME SERIES

# Clean and fix stations identification
df_pollution['id'] = df_pollution['NMT'].map(df_stations.set_index('NMT')['id'])

# Rename columns names
df_pollution = df_pollution.rename(columns={'tipo': 'timetable'})

# Rename Timetable values
df_pollution['timetable'] = df_pollution['timetable'].replace(['D','N','T','E'], ['Day','Night','Total','Evening'])

# Create a date column
df_pollution['date'] = pd.to_datetime(dict(year=df_pollution.anio, month=df_pollution.mes, day=df_pollution.dia))

# Remove unnecessary columns
df_pollution.drop(columns=['anio','mes','dia'],inplace=True)

# Transform sound measure field values to float
df_pollution['LAEQ'] = df_pollution['LAEQ'].str.replace(",",".").astype(np.float64)
df_pollution['LAS01'] = df_pollution['LAS01'].str.replace(",",".").astype(np.float64)
df_pollution['LAS10'] = df_pollution['LAS10'].str.replace(",",".").astype(np.float64)
df_pollution['LAS50'] = df_pollution['LAS50'].str.replace(",",".").astype(np.float64)
df_pollution['LAS90'] = df_pollution['LAS90'].str.replace(",",".").astype(np.float64)
df_pollution['LAS99'] = df_pollution['LAS99'].str.replace(",",".").astype(np.float64)

# Select only usefull columns
df_stations = df_stations[['id','longitude','latitude']]


########################
# INITIALIZE VARIABLES #
########################

ini_monitoring_stations =  ['Barajas Pueblo','Carlos V','Casa de Campo','Castellana','El Pardo']
ini_timetables = df_pollution['timetable'].unique()
ini_sound_measure = 'LAEQ'
ini_start_date = df_pollution['date'].max() - DateOffset(months=1)
ini_end_date = df_pollution['date'].max()
ini_min_date = df_pollution['date'].min()

list_weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
color_scale = 'jet'
margin=dict(l=20, r=20, t=20, b=20)
legend = dict(yanchor="top",y=0.99,xanchor="left",x=0.01)

#############
# FUNCTIONS #
#############

def generate_resample_fig(df, period, sound_measure):
    
    df_downsample = df.resample(period).mean()
    # Reset index
    df_downsample.reset_index(inplace=True)
    df_downsample.set_index('date', inplace=True)
    fig = px.line(df_downsample,x=df_downsample.index, y=sound_measure, color='id')    
    fig.update_layout(margin=margin,legend_title_text='Stations', legend=legend)
    return fig

def detect_outliers_zscore(index, data):
    outliers = []
    outliers_index = []
    thres = 3
    mean = np.mean(data)
    std = np.std(data)
    # print(mean, std)
    j = 0
    for i in data:
        z_score = (i-mean)/std
        if (np.abs(z_score) > thres):
            outliers.append(i)
            outliers_index.append(index[j])
        j= j+1
    return outliers_index, outliers


def detect_outliers_iqr(data):
    outliers = []
    data = sorted(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    # print(q1, q3)
    IQR = q3-q1
    lwr_bound = q1-(1.5*IQR)
    upr_bound = q3+(1.5*IQR)
    # print(lwr_bound, upr_bound)
    for i in data: 
        if (i<lwr_bound or i>upr_bound):
            outliers.append(i)
    return outliers

app = Dash(__name__, prevent_initial_callbacks=True)

app.layout = html.Div(
    id="app-container",
        children=[
            html.Div(
            id="left-column",
            className="three columns",
            children=[
                html.Div([
                    dcc.ConfirmDialog(
                        id='confirm-alert',
                        message='Danger danger! Are you sure you want to continue?',
                    )
                ]),
                html.Div([
                    html.P(id='err', style={'color': 'red'})
                ]),
                html.Div(
                    id="description-card",
                    children=[
                        html.H5("Noise pollution:"),
                        html.H3("Madrid monitoring stations"),
                        html.Div(
                            id="intro",
                            children="Analysis of noise pollution data from Madrid monitoring stations.",
                        ),
                    ],
                ),
                html.Div(
                    id="control-card",
                    children=[                        
                        html.Div(
                            children=[
                                html.Label("Timetable"),
                                dcc.Dropdown(ini_timetables, 'Day', id='timetable-dropdown')
                                ]
                            ),
                        html.Div(
                            children=[
                                html.Label("Measurement"),
                                dcc.Dropdown(['LAEQ','LAS01','LAS10','LAS50','LAS90','LAS99'], ini_sound_measure, id='measurement-dropdown'),
                                ]
                            ),
                        html.Br(),
                        html.Div(
                            children=[
                            html.Label("Week days"),
                            dcc.Checklist(
                                options=[
                                       {'label': 'Mon', 'value': 0},
                                       {'label': 'Tue', 'value': 1},
                                       {'label': 'Wed', 'value': 2},
                                       {'label': 'Thu', 'value': 3},
                                       {'label': 'Fri', 'value': 4},
                                       {'label': 'Sat', 'value': 5},
                                       {'label': 'Sun', 'value': 6},
                                   ],                                        
                                value = [0,1,2,3,4,5,6],
                                inline=True,
                                id='weekdays-checklist'
                                )  
                            ]
                        ),
                        html.Br(),
                        html.Div(
                            children=[
                                    html.Label("Noise monitoring stations"),
                                    html.Div(
                                        id="checklist-container",
                                        children=dcc.Checklist(
                                            id="stations-select-all",
                                            options=[{"label": "Select all stations", "value": "All"}],
                                            value=[],
                                            )
                                        ),
                                    dcc.Dropdown(
                                        df_stations['id'],
                                        ini_monitoring_stations,
                                        multi=True,id='stations-dropdown'
                                    )
                                ]
                            ),
                       
                        html.Br(),
                        html.Div(
                            id="reset-btn-outer",
                            children=html.Button('Apply', id='apply-button', n_clicks=0),
                        ),
                        html.Br(),
                        #dash_table.DataTable(id='data-table', page_size=5),
                        #dcc.Loading(id="loading-1", type="default", children=dash_table.DataTable(id='data-table', page_size=7)),
                    ]
                )
            ]),
        html.Div(
            id="right-column",
            className="nine columns",
            children=[
                html.H4("Noise map"),
                dcc.Loading(id="loading-1", type="default", children=[
                    html.Div([
                        
                        html.Div(
                            children=[
                        html.Label("Select interval"),
                        dcc.DatePickerRange(
                            id='date-range',
                            month_format='D-M-Y',
                            end_date_placeholder_text='D-M-Y',
                            start_date_placeholder_text='D-M-Y',
                            min_date_allowed=ini_min_date,
                            max_date_allowed=ini_end_date,
                            start_date=ini_start_date,
                            end_date=ini_end_date
                        )
                        ]
                            ),
                        html.Br(),
                        dcc.Tabs([
                            dcc.Tab(label='Bubble map', children=[dcc.Graph(id='graph-bubble-map')]),
                            dcc.Tab(label='Density map', children=[dcc.Graph(id='graph-density-map')]),
                            dcc.Tab(label='Heat map', children=[dcc.Graph(id='graph-heat-map')]),
                            dcc.Tab(label='Bar map', children=[dcc.Graph(id='graph-bar-map')])
                            ])
                            
                    ])
                ]),
                html.Br(),
                html.Hr(),
                #dcc.Loading(id="loading-3", type="default", children=dcc.Graph(id='graph-heat-map')),
                html.H4("Noise time series"),
                dcc.Loading(id="loading-4", type="default", children=[
                    html.Div([                       
                        
                        html.Div(
                            children=[
                            html.Label("Select interval"),
                            dcc.DatePickerRange(
                                id='date-range-time-series',
                                month_format='D-M-Y',
                                end_date_placeholder_text='D-M-Y',
                                start_date_placeholder_text='D-M-Y',
                                min_date_allowed=ini_min_date,
                                max_date_allowed=ini_end_date,
                                start_date=ini_min_date,
                                end_date=ini_end_date
                                )
                            ]
                        ),
                        html.Br(),
                        dcc.Tabs([
                            dcc.Tab(label='Days', children=[dcc.Graph(id='graph-days-map')]),
                            dcc.Tab(label='Weeks', children=[dcc.Graph(id='graph-weeks-map')]),
                            dcc.Tab(label='Months', children=[dcc.Graph(id='graph-months-map')]),
                            dcc.Tab(label='Years', children=[dcc.Graph(id='graph-years-map')])]
                            )
                    ])
                ]),
              
                html.Br(),
                html.H5("Decomposition and Analysis"),
                html.Div(
                    id='select-period',
                    children=[
                        html.Label("Period"),
                        dcc.Dropdown([{'label': 'Month', 'value': 'M'},
                                      {'label': 'Week', 'value': 'W'}],
                                     value = 'M', id='period-dropdown'),
                        ]
                    ),
                html.Br(),
                dcc.Loading(id="loading-6", type="default", children=[
                    
                    html.Div([
                        dcc.Tabs([
                            #dcc.Tab(label='Rolling', children=[dcc.Graph(id='graph-smooth-map')]),
                            dcc.Tab(label='Trend', children=[dcc.Graph(id='graph-trend-map')]),
                            dcc.Tab(label='Seasonality', children=[dcc.Graph(id='graph-season-map')]),
                            dcc.Tab(label='Residual', children=[dcc.Graph(id='graph-resid-map')]),
                            dcc.Tab(label='Anomalies', children=[dcc.Graph(id='graph-anomalies-map')])]
                            )
                    ])
                ]),
            ])
       
])
             
@callback(
    [Output('graph-bubble-map', 'figure'),
     Output('graph-density-map', 'figure'),
     Output('graph-heat-map', 'figure'),
     Output('graph-bar-map', 'figure'),
     Output('graph-days-map', 'figure'),
     Output('graph-weeks-map', 'figure'),
     Output('graph-months-map', 'figure'),
     Output('graph-years-map', 'figure'),
     Output('graph-trend-map', 'figure'),
     Output('graph-season-map', 'figure'),
     Output('graph-resid-map', 'figure'),
     Output('graph-anomalies-map', 'figure'),
     Output('err', 'children')
     ],
    [Input('apply-button', 'n_clicks')],
    State('timetable-dropdown', 'value'),
    State('measurement-dropdown', 'value'),
    State('weekdays-checklist', 'value'),
    State('stations-dropdown', 'value'),
    State('date-range', 'start_date'),
    State('date-range', 'end_date'),
    State('date-range-time-series', 'start_date'),
    State('date-range-time-series', 'end_date'),
    State('period-dropdown', 'value')
    #prevent_initial_call=True
)
def update_output(n_clicks, timetable, sound_measure, weekdays, stations, start_date, end_date, start_date_time_series, end_date_time_series, period):
    
    # No weekdays selected
    if len(weekdays)==0:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, 'Must select at least one day of the week'
    
    # No station selected
    if len(stations)==0:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, 'Must select at least one monitoring station'
           
    ###############
    # FILTER DATA #
    ###############
    
    # Sort selected stations
    stations = sorted(stations)
    
    # Filter by selected stations
    df_selected_stations = df_stations[df_stations['id'].isin(stations)]
    
    # Filter by selected stations
    df_selected = df_pollution[df_pollution['id'].isin(stations)]
    
    # Filter by selected stations
    df_selected = df_selected[df_selected['timetable'] == timetable]
         
    # Filter by selected week days
    df_selected = df_selected[df_selected['date'].dt.dayofweek.isin(weekdays)]
    
    # Set date as dataframe index
    df_selected.set_index('date', inplace=True)

    #############################################################
    # MERGE NOISE TIME SERIES AND STATIONS DATA OF THE INTERVAL #
    #############################################################

    # Filter by selected interval for maps
    df_pollution_interval = df_selected[(df_selected.index >= start_date) & (df_selected.index <= end_date)]
    
    # Group by stations id and compute the mean sound measure of the interval
    df_pollution_mean = df_pollution_interval[['id',sound_measure]].groupby(['id']).mean().round(2)
    
    # Merge stations with noise time series data
    df_pollution_stations = df_pollution_mean.merge(df_selected_stations, on='id')
    
    # Generate Bubble Map plot
    fig1 = px.scatter_mapbox(df_pollution_stations, lat = 'latitude', lon = 'longitude', size = sound_measure, color = sound_measure,
                             opacity = 0.8, zoom = 9, hover_name='id', mapbox_style = 'open-street-map', color_continuous_scale = color_scale)
    fig1.update_layout(margin=margin)

    # Generate Density Map plot
    fig2 = px.density_mapbox(df_pollution_stations, lat = 'latitude', lon = 'longitude', z = sound_measure,
                             radius = 50,opacity = 0.9, zoom = 9, hover_name="id",mapbox_style = 'open-street-map', color_continuous_scale = color_scale)
    fig2.update_layout(margin=margin)
    
    ##############################################
    # GROUP NOISE TIME SERIES BY DAY OF THE WEEK #
    ##############################################
    
    # Group by day of the week and compute the mean sound measure of the interval
    df_weekday_pollution = df_pollution_interval[['id',sound_measure]].groupby(['id',df_pollution_interval.index.day_name()]).mean().round(2)
    
    # Reset index
    df_weekday_pollution.reset_index(inplace=True)
    
    # Sort by weekday
    df_weekday_pollution['date'] = pd.Categorical(df_weekday_pollution['date'],categories=list_weekdays)

    # Create matrix Stations X Weekday
    matrix_weekday = df_weekday_pollution.pivot_table(index="id", columns="date", values=sound_measure)
    matrix_weekday.sort_index(axis = 1)
    
    # Generate Heatmap
    fig3 = px.imshow(matrix_weekday, color_continuous_scale = color_scale, text_auto=True,aspect="auto", labels=dict(x="Day of week", y="Monitoring stations", color=sound_measure))
    fig3.update_xaxes(side="top")
    fig3.update_layout(yaxis_title=None,xaxis_title=None)
    
    fig = px.bar(df_pollution_stations, x="id", y=sound_measure, color=sound_measure, color_continuous_scale = color_scale)
    fig.update_layout(margin=margin)
    fig.update_layout(yaxis_title=None,xaxis_title=None)
    
    #####################################
    # RESAMPLE (DOWNSAMPLE) TIME SERIES #
    #####################################
    
    # Filter by selected interval for time series
    df_selected = df_selected[(df_selected.index >= start_date_time_series) & (df_selected.index <= end_date_time_series)]
    
    df_groupby_id = df_selected[['id',sound_measure]].groupby(['id'])
    # Resample Day, Week, Month, Year
    fig4_1 = generate_resample_fig(df_groupby_id, 'D', sound_measure)
    fig4_2 = generate_resample_fig(df_groupby_id, 'W', sound_measure)
    fig4_3 = generate_resample_fig(df_groupby_id, 'M', sound_measure)
    fig4_4 = generate_resample_fig(df_groupby_id, 'Y', sound_measure)
    
    ###############################################
    # DECOMPOSE TIME SERIES TREND AND SEASONALITY #
    ###############################################
    
    
    # Initialize figure data
    data_trend = []
    data_seasonal = []
    data_resid = []
    data_downsample = []
    data_outliers = []
          
    # For each station compute trend and seasonality
    for st in stations:
        
        # Fill missing dates with PAD
        df_fillna = df_selected[df_selected['id']==st].asfreq(pd.offsets.BDay(), method="pad")
        
        # Use Month downsample data
        df_downsample = df_fillna.resample(period).mean()
        
        try:
            # Compute decomposition
            decomposition = seasonal_decompose(x=df_downsample[sound_measure], model='multiplicative')
            #decomposition = seasonal_decompose(x=df_downsample_m[df_downsample_m['id']==st][sound_measure], model='multiplicative')
        
            # Append plots
            plot = go.Scatter(x=decomposition.trend.index,y=decomposition.trend,mode="lines",name=st) 
            data_trend.append(plot)
            plot = go.Scatter(x=decomposition.seasonal.index,y=decomposition.seasonal,mode="lines",name=st) 
            data_seasonal.append(plot)
            plot = go.Scatter(x=decomposition.seasonal.index,y=decomposition.resid,mode="lines",name=st)            
            data_resid.append(plot)      
            plot = go.Scatter(x=df_downsample.index,y=df_downsample[sound_measure],mode="lines",name=st)            
            data_downsample.append(plot) 
            index_outliers, outliers = detect_outliers_zscore(decomposition.seasonal.index, decomposition.resid)          
            plot = go.Scatter(x=index_outliers, y=df_downsample[df_downsample.index.isin(index_outliers)][sound_measure],
                              mode="markers", text= st, showlegend=False, marker_symbol='square',
                              marker_color='LightSkyBlue',marker_line_color="midnightblue",marker_line_width=2,
                              marker_size=10, hovertemplate="""Station: """+ st + """ <br> Anomaly: %{y} <br> Date: %{x} <br><extra></extra>""")
            data_outliers.append(plot)
            
        except ValueError:
            return fig1, fig2, fig3, fig, fig4_1, fig4_2, fig4_3, fig4_4, no_update, no_update, no_update, no_update, 'Not enough data from date interval. Must have 2 complete cycles to compute decomposition'
                        

    # Append anomalies
    for i in data_outliers:
        data_downsample.append(i)   
        
    #Generate figures
    fig5_1 = go.Figure(data=data_trend,layout=go.Layout(margin=margin, legend=legend))
    fig5_2 = go.Figure(data=data_seasonal,  layout=go.Layout(margin=margin, legend=legend))
    fig5_3 = go.Figure(data=data_resid,  layout=go.Layout(margin=margin, legend=legend))
    fig5_4 = go.Figure(data=data_downsample,  layout=go.Layout(margin=margin, legend=legend))       
    
    return fig1, fig2, fig3, fig, fig4_1, fig4_2, fig4_3, fig4_4, fig5_1, fig5_2, fig5_3, fig5_4, ''



@app.callback(
    [
        Output("stations-dropdown", "value"),
        Output("stations-dropdown", "options")
    ],
    [Input("stations-select-all", "value"), State("stations-dropdown", "options")],
    prevent_initial_call=True
)

def update_stations_dropdown(select_all, options):
  
    if select_all == ["All"]:
        value = options
        
    else:
        value = no_update

    return value, options

@app.callback(
    Output("checklist-container", "children"),
    [Input("stations-dropdown", "value")],
    [State("stations-dropdown", "options"), State("stations-select-all", "value")],
    prevent_initial_call=True
)

def update_checklist(selected, select_options, checked):
    if len(selected) < len(select_options) and len(checked) == 0:
        raise PreventUpdate()

    elif len(selected) < len(select_options) and len(checked) == 1:
        return dcc.Checklist(
            id="stations-select-all",
            options=[{"label": "Select all stations", "value": "All"}],
            value=[],
        )
        
    elif len(selected) == len(select_options) and len(checked) == 1:
        raise PreventUpdate()

    return dcc.Checklist(
        id="stations-select-all",
        options=[{"label": "Select all stations", "value": "All"}],
        value=['All'],
    )
    



if __name__ == '__main__':
    app.run(debug=True)