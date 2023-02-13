import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)
server = app.server

# ---------------------------------------------------------------
model = joblib.load('covid_randomforest')  # load model

covid = pd.read_csv('final_covid.csv')  # load dataset for plots


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


# app = dash.Dash(external_stylesheets=[dbc.themes.CYBORG])
# ---------------------------------------------------------------
app.layout = html.Div([

    html.H3("Predicting the survival of COVID-19 patients",
            style={'text-align': 'center'}),
    html.Br(),
    html.H5(id="prediction_result",
            style={'text-align': 'center'}),
    html.Div([
        html.Label("Age:",
                   style={'display': 'inline-block', 'width': '120px'}),
        dcc.Input(id='input_age',
                  type='number',
                  min=1, max=120, step=1,
                  value='35')]),
    html.Div([
        html.Label("Sex:",
                   style={'display': 'inline-block', 'width': '120px'}),
        dcc.RadioItems(id='radio_sex',
                       options=[{'label': 'Female', 'value': 1},
                                {'label': 'Male', 'value': 0}],
                       value=0,
                       style={'display': 'inline-block'})]),
    html.Div([
        html.Label("Pneumonia:",
                   style={'display': 'inline-block', 'width': '120px'}),
        dcc.RadioItems(id='radio_pneu',
                       options=[{'label': 'Yes', 'value': 1},
                                {'label': 'No', 'value': 0}],
                       value=0,
                       style={'display': 'inline-block'})]),
    html.Div([
        html.Label("Diabetes:",
                   style={'display': 'inline-block', 'width': '120px'}),
        dcc.RadioItems(id='radio_diab',
                       options=[{'label': 'Yes', 'value': 1},
                                {'label': 'No', 'value': 0}],
                       value=0,
                       style={'display': 'inline-block'})]),
    html.Div([
        html.Label("COPD:",
                   style={'display': 'inline-block', 'width': '120px'}),
        dcc.RadioItems(id='radio_copd',
                       options=[{'label': 'Yes', 'value': 1},
                                {'label': 'No', 'value': 0}],
                       value=0,
                       style={'display': 'inline-block'})]),
    html.Div([
        html.Label("Asthma:",
                   style={'display': 'inline-block', 'width': '120px'}),
        dcc.RadioItems(id='radio_asth',
                       options=[{'label': 'Yes', 'value': 1},
                                {'label': 'No', 'value': 0}],
                       value=0,
                       style={'display': 'inline-block'})]),
    html.Div([
        html.Label("Immunosupp:",
                   style={'display': 'inline-block', 'width': '120px'}),
        dcc.RadioItems(id='radio_imm',
                       options=[{'label': 'Yes', 'value': 1},
                                {'label': 'No', 'value': 0}],
                       value=0,
                       style={'display': 'inline-block'})]),
    html.Div([
        html.Label("Hypertension:",
                   style={'display': 'inline-block', 'width': '120px'}),
        dcc.RadioItems(id='radio_htn',
                       options=[{'label': 'Yes', 'value': 1},
                                {'label': 'No', 'value': 0}],
                       value=0,
                       style={'display': 'inline-block'})]),
    html.Div([
        html.Label("Heart disease:",
                   style={'display': 'inline-block', 'width': '120px'}),
        dcc.RadioItems(id='radio_cardio',
                       options=[{'label': 'Yes', 'value': 1},
                                {'label': 'No', 'value': 0}],
                       value=0,
                       style={'display': 'inline-block'})]),
    html.Div([
        html.Label("Obese:",
                   style={'display': 'inline-block', 'width': '120px'}),
        dcc.RadioItems(id='radio_obese',
                       options=[{'label': 'Yes', 'value': 1},
                                {'label': 'No', 'value': 0}],
                       value=0,
                       style={'display': 'inline-block'})]),
    html.Div([
        html.Label("Renal disease:",
                   style={'display': 'inline-block', 'width': '120px'}),
        dcc.RadioItems(id='radio_renal',
                       options=[{'label': 'Yes', 'value': 1},
                                {'label': 'No', 'value': 0}],
                       value=0,
                       style={'display': 'inline-block'})]),
    html.Div([
        html.Label("Smoke:",
                   style={'display': 'inline-block', 'width': '120px'}),
        dcc.RadioItems(id='radio_smoke',
                       options=[{'label': 'Yes', 'value': 1},
                                {'label': 'No', 'value': 0}],
                       value=0,
                       style={'display': 'inline-block'})]),
    html.Div([
        html.Label("Hospitalized:",
                   style={'display': 'inline-block', 'width': '120px'}),
        dcc.RadioItems(id='radio_hosp',
                       options=[{'label': 'Yes', 'value': 1},
                                {'label': 'No', 'value': 0}],
                       value=0,
                       style={'display': 'inline-block'})]),
    html.Br(),
    dcc.Dropdown(id='feature_dropdown',
                 options=[{'label': 'Sex', 'value': 'Sex'},
                          {'label': 'Pneumonia', 'value': 'Pneumonia'},
                          {'label': 'Diabetes', 'value': 'Diabetes'},
                          {'label': 'COPD', 'value': 'COPD'},
                          {'label': 'Asthma', 'value': 'Asthma'},
                          {'label': 'Immunosuppressed',
                           'value': 'Immunosuppressed'},
                          {'label': 'Hypertension', 'value': 'Hypertension'},
                          {'label': 'Heart disease', 'value': 'Cardiovascular'},
                          {'label': 'Obese', 'value': 'Obese'},
                          {'label': 'Renal disease', 'value': 'Chronic renal'},
                          {'label': 'Smoke', 'value': 'Smoke'},
                          {'label': 'Hospitalized', 'value': 'Hospitalized'},
                          {'label': 'Died', 'value': 'Died'}],
                 multi=False,
                 value='Hospitalized',
                 style={'width': '40%', 'display': 'inline-block'}),
    html.Br(),
    dcc.Graph(id='age_hist',
              figure={},
              style={'width': '50%',
                     'display': 'inline-block'}),  # Side-by-side
    dcc.Graph(id='feature_bar',
              figure=feature_bar(),
              style={'width': '50%',
                     'float': 'right',
                     'display': 'inline-block'}),  # Side-by-side
    html.H5("Savij Apichon, 2022",
            style={'text-align': 'right'})
])

# ------------------------------------------------------------------


@app.callback(
    Output(component_id='age_hist', component_property='figure'),
    Input(component_id='feature_dropdown', component_property='value')
)
def update_hist(feature_dropdown):
    covid.sort_values(by=[feature_dropdown],  # Ensures color with fewest count is in front
                      ascending=True, inplace=True)
    if feature_dropdown == 'Sex':  # Change the legend label depending on feature
        legend_name = {'0': 'Male', '1': 'Female'}
    else:
        legend_name = {'0': 'No', '1': 'Yes'}
    age_hist = px.histogram(data_frame=covid, x='Age', color=feature_dropdown,
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


@app.callback(
    Output(component_id='prediction_result', component_property='children'),
    [Input(component_id='input_age', component_property='value'),
     Input(component_id='radio_sex', component_property='value'),
     Input(component_id='radio_pneu', component_property='value'),
     Input(component_id='radio_diab', component_property='value'),
     Input(component_id='radio_copd', component_property='value'),
     Input(component_id='radio_asth', component_property='value'),
     Input(component_id='radio_imm', component_property='value'),
     Input(component_id='radio_htn', component_property='value'),
     Input(component_id='radio_cardio', component_property='value'),
     Input(component_id='radio_obese', component_property='value'),
     Input(component_id='radio_renal', component_property='value'),
     Input(component_id='radio_smoke', component_property='value'),
     Input(component_id='radio_hosp', component_property='value')]
)
def model_prediction(input_age, radio_sex, radio_pneu, radio_diab, radio_copd,
                     radio_asth, radio_imm, radio_htn, radio_cardio,
                     radio_obese, radio_renal, radio_smoke, radio_hosp):
    # prevents warning message: X does not have valid feature names...
    X_pred = pd.DataFrame(np.array([[input_age, radio_sex, radio_pneu, radio_diab, radio_copd,
                                     radio_asth, radio_imm, radio_htn, radio_cardio,
                                     radio_obese, radio_renal, radio_smoke, radio_hosp]]),
                          columns=['Age', 'Sex', 'Pneumonia', 'Diabetes', 'COPD', 'Asthma', 'Immunosuppressed',
                                   'Hypertension', 'Cardiovascular', 'Obese', 'Chronic renal', 'Smoke', 'Hospitalized'])
    survival_prob = model.predict_proba(X_pred)
    prediction_result = f"Probability of survival: {round(survival_prob[0,0], 3)}"
    return prediction_result

# ------------------------------------------------------------------


if __name__ == '__main__':
    app.run_server(debug=True)
