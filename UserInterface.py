import dash
import base64
import os
import io
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import soundfile as sf

app = dash.Dash()

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

app.layout = html.Div(children=[
    html.H1('Dash tutorialsss'),
    dcc.Upload(
        id='upload-image',
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
    ),

    html.Div(id='output-data-upload')
])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-image', 'contents'),
               Input('upload-image', 'filename')])
def update_output(contents, filename):
    print(filename)
    if filename is not None:
        return html.Div([
            html.H5(filename),
            test(contents, filename),
            html.H5(contents)
    ])


def test(c, filename):
    if c is not None:
        ctype, cstring = str(c).split(',')
        decoded = base64.b64decode(cstring)
        filepath = os.path.dirname(os.path.realpath(__file__)) + "/" + filename
        try:
            file = open(filename, 'wb')
            file.write(decoded)
            file.close()
        except:
            print("something went wrong")

        return html.Audio(src=c, autoPlay="audio", controls="audio")
    else:
        return 'none'


if __name__ == '__main__':
    app.run_server(debug=True)