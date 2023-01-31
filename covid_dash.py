import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# ----------------------------------------------------------------------------

covid = pd.read_csv('final_covid.csv')


def feature_bar():
    feature = []
    number = []
    for col in covid.columns[2:]:  # Exclude Age and Sex
        count = covid[col].sum()
        feature.append(col)
        number.append(count)
    df_count = pd.concat([pd.DataFrame(feature, columns=['Feature']),
                         pd.DataFrame(number, columns=['Count'])], axis=1).sort_values('Count', ascending=True)
    feature_bar = px.bar(data_frame=df_count, x=df_count.Feature, y=df_count.Count,
                         opacity=0.7,
                         title='Count of each feature within this dataset',
                         template='plotly_dark')
    feature_bar.update_traces(marker=dict(color='turquoise'))
    return feature_bar


app = Dash(external_stylesheets=[dbc.themes.SLATE])

app_tabs = html.Div(
    [dbc.Tabs(
        [dbc.Tab(label='Tab 1', tab_id='tab-id-1',
                 labelClassName='text-success font-weight-bold',
                 activeLabelClassName='text-danger'),
         dbc.Tab(label='Tab 2', tab_id='tab-id-2',
                 labelClassName='text-success font-weight-bold',
                 activeLabelClassName='text-danger')],
        id='tabs',
        active_tab='tab-id-1'
    )]
)

# ----------------------------------------------------------------------------

app.layout = html.Div([

    html.H1("Predicting the survival of COVID-19 patients",
            style={'text-align': 'center'}),

    dcc.Input(id='input_age',
              type='number',
              min=1, max=120, step=1,
              value='35'),
    dcc.RadioItems(id='radio_sex',
                   options=['Male', 'Female']),
    dcc.RadioItems(id='radio_pneumonia',
                   options=['Yes', 'No']),
    dcc.RadioItems(id='radio_copd',
                   options=['Yes', 'No']),
    dcc.RadioItems(id='radio_asthma',
                   options=['Yes', 'No']),
    dcc.RadioItems(id='radio_immuno',
                   options=['Yes', 'No']),
    dcc.RadioItems(id='radio_htn',
                   options=['Yes', 'No']),
    dcc.RadioItems(id='radio_cardio',
                   options=['Yes', 'No']),
    dcc.RadioItems(id='radio_obese',
                   options=['Yes', 'No']),
    dcc.RadioItems(id='radio_renal',
                   options=['Yes', 'No']),
    dcc.RadioItems(id='radio_smoke',
                   options=['Yes', 'No']),
    dcc.RadioItems(id='radio_hosp',
                   options=['Yes', 'No']),
    html.Br(),
    dcc.Dropdown(id='age_dropdown',
                 options=[{'label': 'Sex', 'value': 'Sex'},
                          {'label': 'Pneumonia', 'value': 'Pneumonia'},
                          {'label': 'COPD', 'value': 'COPD'},
                          {'label': 'Asthma', 'value': 'Asthma'},
                          {'label': 'Immunosuppressed',
                              'value': 'Immunosuppressed'},
                          {'label': 'Hypertension', 'value': 'Hypertension'},
                          {'label': 'Cardiovascular', 'value': 'Cardiovascular'},
                          {'label': 'Obese', 'value': 'Obese'},
                          {'label': 'Chronic renal', 'value': 'Chronic renal'},
                          {'label': 'Smoke', 'value': 'Smoke'},
                          {'label': 'Hospitalized', 'value': 'Hospitalized'},
                          {'label': 'Died', 'value': 'Died'}],
                 multi=False,
                 value='Hospitalized',
                 style={'width': '40%'}),
    html.Br(),
    dcc.Graph(id='age_hist',
              figure={},
              style={'width': '50%',
                     'display': 'inline-block'}),  # Side-by-side
    dcc.Graph(id='feature_bar',
              figure=feature_bar(),
              style={'width': '50%',
                     'float': 'right',
                     'display': 'inline-block'})  # Side-by-side
])

# ----------------------------------------------------------------------------


@ app.callback(
    Output(component_id='age_hist', component_property='figure'),
    Input(component_id='age_dropdown', component_property='value'),
)
def update_hist(age_dropdown):
    covid.sort_values(by=[age_dropdown],  # Ensures color with fewest count is in front
                      ascending=True, inplace=True)
    if age_dropdown == 'Sex':  # Change the legend label depending on feature
        legend_name = {'0': 'Male', '1': 'Female'}
    else:
        legend_name = {'0': 'No', '1': 'Yes'}
    age_hist = px.histogram(data_frame=covid, x='Age', color=age_dropdown,
                            barmode='overlay', nbins=24, range_x=[0, 120], opacity=0.7,
                            title='Age distribution of Covid-19 patients',
                            template='plotly_dark')
    age_hist.update_layout(yaxis_title='Count')
    age_hist.update_traces(
        marker=dict(line=dict(color='black', width=1)))  # Separate each bin
    age_hist.for_each_trace(
        lambda x: x.update(name=legend_name[x.name],  # Legend title
                           legendgroup=legend_name[x.name],  # Legend_name
                           # Hover
                           hovertemplate=x.hovertemplate.replace(x.name, legend_name[x.name])))
    return age_hist


# ----------------------------------------------------------------------------


if __name__ == '__main__':
    app.run_server(debug=True)
