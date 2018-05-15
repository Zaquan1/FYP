import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly import tools

from Feature import Feature
from keras.models import load_model

import base64
import os, errno
import numpy as np
import miscellaneous as misc


# create dir for temporarily store the music
try:
    os.mkdir('resource')
    os.mkdir('resource/temp')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

app = dash.Dash()
stop = True
empty_dict_music = {
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
music_list = []

# prepare the model
model_arousal = load_model("resource/model/LSTMArousal.h5")
model_valance = load_model("resource/model/LSTMValance.h5")

app.scripts.config.serve_locally = True
# creating the main app layout
app.layout = html.Div(children=[
    html.H1('Music Emotion Recognition', style={'textAlign': 'center'}),
    html.Hr(),
    # drag and drop
    dcc.Upload(
        id='upload-audio',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select a File'),
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
    html.Div([
        "Pick existing song:",
        dcc.Dropdown(
            id='musicList',
            options=[],
            searchable=False,
            clearable=False
        )
    ]),
    html.Div(
        id='features-graph-container'
    )
])

# create new graph id
def generate_graph_id(value):
    return '{}_graph'.format(value)


# initialise dynamic graph
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

# generate new interval id
def generate_interval_id(value):
    return '{}_interval'.format(value)

# listen to any upload or changes in the dropdown menu
@app.callback(
    Output('features-graph-container', 'children'),
    [Input('upload-audio', 'contents'),
     Input('musicList', 'value')],
    [State('upload-audio', 'filename')])
def display_controls(contents, curr_music, filename):
    if contents is not None:
        filename_temp = str(filename).split('.')
        # check file type
        if filename_temp[-1] != 'mp3':
            return html.Div([
                html.H3('Error: file is not in mp3 format', style={'textAlign': 'center'})
            ])
        # if dropdown menu is empty, create new music
        if curr_music is not None:
            music = searchMusic(curr_music, music_list)
            if music:
                filename = music[0]['name']
                contents = music[0]['binary']
            else:
                get_audio_contents(contents, filename)
        else:
            get_audio_contents(contents, filename)

        return html.Div([

            html.H3(filename, style={'textAlign': 'center'}),
            html.Div([
               html.Button('Stop', id='stp-button', n_clicks=0)
            ], style={'textAlign': 'center'}),
            html.Div(id='clicked-button', children='rep:0 stp:0 last:nan', style={'display': 'none'}),

            html.Div([
                DYNAMIC_GRAPH['Valance-arousal']
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                DYNAMIC_GRAPH['Features']
            ], style={'width': '50%', 'display': 'inline-block'}),

            html.Audio(src=contents, id='music-audio', autoPlay='audio'),
            dcc.Interval(
                id=generate_interval_id('interval'),
                interval=1*500,
                n_intervals=0
            ),
        ])

# update dropdown menu option
@app.callback(
    Output('musicList', 'options'),
    [Input('upload-audio', 'filename')])
def update_option_menu(filename):
    music = searchMusic(filename, music_list)
    if not music:
        return [{'label': filename, 'value': filename}] + \
               [{'label': music['name'], 'value': music['name']} for music in music_list]

# update dropdown menu value
@app.callback(
    Output('musicList', 'value'),
    [Input('upload-audio', 'filename')])
def update_value_menu(filename):
    return filename


# extract the features from audio file and make prediction
def get_audio_contents(c, filename):
    if c is not None:
        ctype, cstring = str(c).split(',')
        decoded = base64.b64decode(cstring)
        music = empty_dict_music.copy()
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
        music['energy_feature'] = (Feature.sync_frames(music_feature, music_feature.extract_energy_features()))[1, :]
        music['timbre_feature'] = (Feature.sync_frames(music_feature, music_feature.extract_timbre_features())).mean(
            axis=0)
        music['melody_feature'] = (Feature.sync_frames(music_feature, music_feature.extract_melody_features())).mean(
            axis=0)
        music['rhythm_feature'] = (Feature.sync_frames(music_feature, music_feature.extract_rhythm_features()))[:-3]
        # data preparation for prediction
        data = misc.series_to_supervised(np.transpose(music_feature.get_all_features()), 3, 1)
        data = data.values.reshape(data.values.shape[0], 4, 146)
        predict_arousal = model_arousal.predict(data)
        predict_valance = model_valance.predict(data)
        # save prediction into dictionary
        music['arousal_predict'] = predict_arousal[:, 0]
        music['valance_predict'] = predict_valance[:, 0]
        music['duration'] = [i for i in misc.frange(0.5, len(music['timbre_feature']) * 0.5, 0.5)]
        '''['%d:%2.1f' % (int((i + 1.5) / 60), (i + 1.5) % 60) for i in
                             frange(0.5, len(predict) * 0.5, 0.5)]'''
        music_list.insert(0, music)


# create graph
def generate_output_callback(key):
    def output_callback(n_interval, curr_music):

        music = searchMusic(curr_music, music_list)[0]
        # This function can display different outputs depending on
        # the values of the dynamic controls
        if key == 'Features':
            # create/update rhythm feature graph
            trace_rhythm = go.Scatter(
                x=music['duration'][:n_interval],
                y=music['rhythm_feature'][:n_interval],
                name='Rhythm Feature'
            )
            # create/update timbre feature graph
            trace_timbre = go.Scatter(
                x=music['duration'][:n_interval],
                y=music['timbre_feature'][:n_interval],
                name='Timbre Feature'
            )
            # create/update energy feature graph
            trace_energy = go.Scatter(
                x=music['duration'][:n_interval],
                y=music['energy_feature'][:n_interval],
                name='Energy Feature'
            )
            # create/update melody feature graph
            trace_melody = go.Scatter(
                x=music['duration'][:n_interval],
                y=music['melody_feature'][:n_interval],
                name='Melody Feature'
            )
            fig = tools.make_subplots(4, 1, subplot_titles=('Timbre Feature', 'energy Feature',
                                                            'melody Feature', 'rhythm Feature'))
            fig.append_trace(trace_timbre, 1, 1)
            fig.append_trace(trace_energy, 2, 1)
            fig.append_trace(trace_melody, 3, 1)
            fig.append_trace(trace_rhythm, 4, 1)
            fig['layout']['xaxis4'].update(title='Duration')
        else:
            # create valance arousal graph
            trace = go.Scatter(
                x=music['valance_predict'][:misc.negative_to_zero(n_interval-3)],
                y=music['arousal_predict'][:misc.negative_to_zero(n_interval-3)],
                name='prediction-traces',
                mode='markers'
            )
            trace_curr = go.Scatter(
                x=music['valance_predict'][misc.negative_to_zero(n_interval-4):misc.negative_to_zero(n_interval - 3)],
                y=music['arousal_predict'][misc.negative_to_zero(n_interval-4):misc.negative_to_zero(n_interval - 3)],
                name='curr-prediction',
                mode='markers+text',
                text=[str(music['duration'][n_interval]) + ' sec.'],
                textposition='bottom',
                textfont=dict(
                    size=15,
                )
            )
            # plotting the emotion label on the graph
            trace_emotion = go.Scatter(
                x=[0,
                   0.367, 0.687, 0.842, 0.964,
                   1,
                   0.964, 0.842, 0.687, 0.367,
                   0,
                   -0.367, -0.687, -0.842, -0.964,
                   -1,
                   -0.964, -0.842, -0.687, -0.367],
                y=[1,
                   0.929, 0.7267, 0.5395, 0.2677,
                   0,
                   -0.2677, -0.5395, -0.7267, -0.929,
                   -1,
                   -0.929, -0.7267, -0.5395, -0.2677,
                   0,
                   0.2677, 0.5395, 0.7267, 0.929],
                name='emotion',
                mode='markers+text',
                text=['Activation',
                      'Alert', 'Excited', 'Elated', 'Happy',
                      'Pleasant',
                      'Contented', 'Serene', 'Relaxed', 'Calm',
                      'Deactivation',
                      'Tired', 'Bored', 'Depressed', 'Sad',
                      'Unpleasant',
                      'Upset', 'Stressed', 'Nervous', 'Tense'],
                textposition='bottom'
            )
            data = [trace, trace_curr, trace_emotion]
            fig = go.Figure(data=data)
            fig['layout'].update(title='Arousal Valance Graph')
            fig['layout']['xaxis1'].update(title='Valance', range=[-1.2, 1.2])
            fig['layout']['yaxis1'].update(title='Arousal', range=[-1.2, 1.2])

        return fig
    return output_callback

# handle the interval for updating the graph
def generate_interval_callback():
    def interval_callback(n_interval, clicks, curr_music):
        music = searchMusic(curr_music, music_list)[0]
        if n_interval >= len(music['duration']) or clicks > 0:
            return 60*60*1000
        else:
            return 1*500
    return interval_callback


app.config.supress_callback_exceptions = True

# interval listener to update graph for every 0.5 second
for key in DYNAMIC_GRAPH:
    print('all callback created: ', key)
    app.callback(
        Output(generate_graph_id(key), 'figure'),
        [Input(generate_interval_id('interval'), 'n_intervals')],
        [State('musicList', 'value')]
    )(
        generate_output_callback(key)
    )
app.callback(
    Output(generate_interval_id('interval'), 'interval'),
    [Input(generate_interval_id('interval'), 'n_intervals'),
     Input('stp-button', 'n_clicks')],
    [State('musicList', 'value')]
)(generate_interval_callback())

# stop audio
@app.callback(Output('music-audio', 'src'),
              [Input('stp-button', 'n_clicks')])
def stop_play_audio(clicks):
    if clicks > 0:
        return ''

# search for existing music in the list
def searchMusic(name, musics):
    return [music for music in musics if music['name'] == name]



app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=False)