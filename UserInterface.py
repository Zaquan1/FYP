import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly import tools

from Feature import Feature
from keras.models import load_model

import base64
import os
import numpy as np

app = dash.Dash()
stop = True
music = {
    'name': None,
    'filepath': None,
    'duration': None,
    'energy_feature': [],
    'timbre_feature': [],
    'rhythm_feature': [],
    'melody_feature': [],
    'arousal_predict': [],
    'valance_predict': [],
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
    html.Div(
        id='features-graph-container'
    ),
])


def generate_graph_id(value):
    return '{}_graph'.format(value)


DYNAMIC_GRAPH = {
    'Valance-arousal': dcc.Graph(
        id=generate_graph_id('Valance-arousal'),
        figure={}
    ),
    'Features': dcc.Graph(
        id=generate_graph_id('Features'),
        figure={}
    )
}


def generate_interval_id(value):
    return '{}_interval'.format(value)


@app.callback(
    Output('features-graph-container', 'children'),
    [Input('upload-audio', 'contents'),
     Input('upload-audio', 'filename')])
def display_controls(contents, filename):
    # generate 2 dynamic controls based off of the datasource selections
    if contents is not None:
        get_audio_contents(contents, filename)
        return html.Div([
            html.Div([
                DYNAMIC_GRAPH['Valance-arousal']
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                DYNAMIC_GRAPH['Features']
            ], style={'width': '50%', 'display': 'inline-block'}),
            dcc.Interval(
                id=generate_interval_id('test'),
                interval=1*500,
                n_intervals=0
            ),
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
        data = Feature.series_to_supervised(np.transpose(music_feature.get_all_features()), 3, 1)
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


def generate_output_callback(key):
    def output_callback(n_interval):
        # This function can display different outputs depending on
        # the values of the dynamic controls
        print('wrapper called')
        if key == 'Features':
            trace_rhythm = go.Scatter(
                x=music['duration'][:n_interval],
                y=music['rhythm_feature'][:n_interval]
            )
            trace_timbre = go.Scatter(
                x=music['duration'][:n_interval],
                y=music['timbre_feature'][:n_interval]
            )
            trace_energy = go.Scatter(
                x=music['duration'][:n_interval],
                y=music['energy_feature'][:n_interval]
            )
            trace_melody = go.Scatter(
                x=music['duration'][:n_interval],
                y=music['melody_feature'][:n_interval]
            )
            fig = tools.make_subplots(4, 1, subplot_titles=('Timbre Feature', 'energy Feature',
                                                            'melody Feature', 'rhythm Feature'))
            fig.append_trace(trace_timbre, 1, 1)
            fig.append_trace(trace_energy, 2, 1)
            fig.append_trace(trace_melody, 3, 1)
            fig.append_trace(trace_rhythm, 4, 1)
            fig['layout']['xaxis4'].update(title='Duration')
        else:
            trace = go.Scatter(
                x=music['valance_predict'][n_interval-3:n_interval],
                y=music['arousal_predict'][n_interval-3:n_interval],
            )
            fig = tools.make_subplots(1, 1)
            fig.append_trace(trace, 1, 1)
            fig['layout'].update(title='Arousal Valance Graph')
            fig['layout']['xaxis1'].update(title='Valance')
            fig['layout']['yaxis1'].update(title='Arousal')

        return fig
    return output_callback

app.config.supress_callback_exceptions = True

for key in DYNAMIC_GRAPH:
    print('all callback created: ', key)
    app.callback(
        Output(generate_graph_id(key), 'figure'),
        [Input(generate_interval_id('test'), 'n_intervals')])(
        generate_output_callback(key)
    )

if __name__ == '__main__':
    app.run_server(debug=False)