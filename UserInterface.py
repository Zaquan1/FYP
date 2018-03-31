import dash
import base64
import os
from Feature import Feature
from pandas import DataFrame
from pandas import concat
from dash.dependencies import Input, Output, Event, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly import tools
import numpy as np
import random
from keras.models import load_model
from keras import backend as K


app = dash.Dash()
music = {
    'name': None,
    'filepath': None,
    'duration': None,
    'energy_feature': None,
    'timbre_feature': None,
    'rhythm_feature': None,
    'melody_feature': None,
    'arousal_predict': None,
    'valance_predict': None,
    'binary': None
}
model = load_model("resource/model/LSTM.h5")
model._make_predict_function()

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
app.layout = html.Div(children=[
    html.H1('Music Emotion Recognition', style={'textAlign': 'center'}),
    html.Hr(),
    dcc.Upload(
        id='upload-audio',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files'),
        ]),
        style={
            'width': '98%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False,
        accept='.mp3'

    ),

    html.Div(id='output-data-upload'),
])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-audio', 'contents'),
               Input('upload-audio', 'filename')])
def update_output(contents, filename):
    if filename is not None and str(filename).split('.')[1] == 'mp3':
        get_audio_contents(contents, filename)
        return html.Div([
            html.H2(music.get('name').split('.')[0]),
            html.Audio(id='music_audio', src=music.get('binary'), controls="audio", style={'width': '99%'}),
            html.Div([
                dcc.Graph(id='arousal-valance-graph', figure=arousal_valance_graph())
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='rhythm-graph', figure=feature_graph())
            ], style={'width': '50%', 'height': '99%', 'display': 'inline-block'})
    ])


def get_audio_contents(c, filename):
    if c is not None:
        ctype, cstring = str(c).split(',')
        decoded = base64.b64decode(cstring)
        music['filepath'] = os.path.dirname(os.path.realpath(__file__)) + "/resource/temp/" + filename
        try:
            file = open(music.get('filepath'), 'wb')
            file.write(decoded)
            file.close()
        except:
            print("something went wrong")
        # record all necessary information into dictionary
        music['name'] = filename
        music['binary'] = c
        music_feature = Feature(music['filepath'])
        # get all extracted feature
        music['energy_feature'] = (Feature.sync_frames(music_feature, music_feature.extract_energy_features())).mean(
            axis=0)[:-3]
        music['timbre_feature'] = (Feature.sync_frames(music_feature, music_feature.extract_timbre_features())).mean(
            axis=0)[:-3]
        music['melody_feature'] = (Feature.sync_frames(music_feature, music_feature.extract_melody_features())).mean(
            axis=0)[:-3]
        music['rhythm_feature'] = (Feature.sync_frames(music_feature, music_feature.extract_rhythm_features()))[:-3]
        # data preparation for prediction
        data = series_to_supervised(np.transpose(music_feature.get_all_features()), 3, 1)
        data = data.values.reshape(data.values.shape[0], 4, 146)
        predict = model.predict(data)
        # save prediction into dictionary
        music['arousal_predict'] = predict[:, 0]
        music['valance_predict'] = predict[:, 1]
        music['duration'] = [i for i in frange(0.5, len(predict) * 0.5, 0.5)]
        '''['%d:%2.1f' % (int((i + 1.5) / 60), (i + 1.5) % 60) for i in
                             frange(0.5, len(predict) * 0.5, 0.5)]'''


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


def arousal_valance_graph():
    trace = go.Scatter(
        x=[i for i in music['valance_predict']],
        y=[i for i in music['arousal_predict']],
        mode='markers+text',
        text=music['duration'],
        textposition='bottom'
    )
    fig = tools.make_subplots(1, 1)
    fig.append_trace(trace, 1, 1)
    fig['layout'].update(title='Arousal Valance Graph')
    fig['layout']['xaxis1'].update(title='Valance')
    fig['layout']['yaxis1'].update(title='Arousal')
    return fig


def feature_graph():
    trace_rhythm = go.Scatter(
        x=music['duration'],
        y=music['rhythm_feature'],
    )
    trace_timbre = go.Scatter(
        x=music['duration'],
        y=music['timbre_feature'],
    )
    trace_energy = go.Scatter(
        x=music['duration'],
        y=music['energy_feature'],
    )
    trace_melody = go.Scatter(
        x=music['duration'],
        y=music['melody_feature'],
    )
    fig = tools.make_subplots(4, 1, subplot_titles=('Timbre Feature', 'energy Feature',
                                                    'melody Feature', 'rhythm Feature'))
    fig.append_trace(trace_timbre, 1, 1)
    fig.append_trace(trace_energy, 2, 1)
    fig.append_trace(trace_melody, 3, 1)
    fig.append_trace(trace_rhythm, 4, 1)
    fig['layout']['xaxis4'].update(title='Duration')
    return fig

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        agg = concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg



if __name__ == '__main__':
    app.run_server(debug=False)