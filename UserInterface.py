import dash
import base64
import os
from Feature import Feature
from pandas import DataFrame
from pandas import concat
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from keras.models import load_model


app = dash.Dash()
model = load_model("resource/model/LSTM.h5")
model.compile(loss='mean_squared_error', metrics=['mse'], optimizer='adam')

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

app.layout = html.Div(children=[
    html.H1('Dash tutorialsss'),
    dcc.Upload(
        id='upload-audio',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files'),
        ]),
        style={
            'width': '99%',
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

    html.Div(id='output-data-upload')
])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-audio', 'contents'),
               Input('upload-audio', 'filename')])
def update_output(contents, filename):
    print(filename)
    if filename is not None:
        return html.Div([
            html.H2(str(filename).split('.')[0]),
            get_contents(contents, filename),

    ])


def get_contents(c, filename):
    if c is not None:
        ctype, cstring = str(c).split(',')
        decoded = base64.b64decode(cstring)
        filepath = os.path.dirname(os.path.realpath(__file__)) + "/resource/temp/" + filename
        try:
            file = open(filepath, 'wb')
            file.write(decoded)
            file.close()
        except:
            print("something went wrong")

        music_feature = get_feature(filepath)
        return html.Div([
            audio_process(c),
            audio_graph(music_feature)
        ])
    else:
        return 'none'


def audio_process(c):
    return html.Audio(src=c, controls="audio",
                          style={
                              'width': '99%'
                          })

def get_feature(filepath):
    music_feature = Feature(filepath)
    music_feature.extract_energy_features()
    music_feature.extract_timbre_features()
    music_feature.extract_melody_features()
    music_feature.extract_rhythm_features()
    return music_feature


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


def audio_graph(music_feature):
    data = series_to_supervised(np.transpose(music_feature.get_all_features()), 3, 1)
    data = data.values.reshape(data.values.shape[0], 4, 146)
    print(data.shape)
    predict = model.predict(data)
    print(predict.shape)
    return dcc.graph(
        figure={
            'data': [
                {'x': predict[:, 0], 'y': predict[:, 1]}
            ]
        }
    )


if __name__ == '__main__':
    app.run_server(debug=True)