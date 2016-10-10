from flask import Flask, render_template, request
from werkzeug import secure_filename
from src.ml import mlp
from src.tools.dataset_helper import DatParser
import os
import webbrowser

app = Flask(__name__)

mlperc = mlp.MultilayerPerceptron(494, 2, 64, 16)
__NN_NAME = "../../res/saved_nns/symmetry_64_16_multi_slice_4_and_5.dat"
mlperc.load(__NN_NAME)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train')
def train():
    return render_template('train.html')


@app.route('/evaluate')
def evaluate():
    return render_template('evaluate.html')


@app.route('/classify')
def upload_file():
    return render_template('upload.html', nnname=__NN_NAME)


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file1():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))

        sample = DatParser.parse_file(f.filename, 1)
        if not mlperc.classify(sample, 1):
            res = "STROKE"
            col = "red"
        else:
            res = "HEALTHY"
            col = "green"
        os.remove(f.filename)
        return render_template('classification_output.html', filename=f.filename, result=res, nnname=__NN_NAME,
                               color=col)


if __name__ == '__main__':
    app.run(debug=True)
