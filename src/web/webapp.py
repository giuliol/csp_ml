from flask import Flask, render_template, request
from werkzeug import secure_filename
from src.ml import mlp
from src.tools.dataset_helper import DatParser
import os

app = Flask(__name__)

mlperc = mlp.MultilayerPerceptron(494, 2, 64, 16)
mlperc.load("../../res/saved_nns/symmetry_64_16.dat")
__NN_NAME = "../../res/saved_nns/symmetry_64_16.dat"


@app.route('/upload')
def upload_file():
    return render_template('upload.html', nnname=__NN_NAME)


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file1():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))

        sample = DatParser.parse_file(f.filename, 1)
        if mlperc.classify(sample, 1):
            res = "STROKE"
        else:
            res = "HEALTHY"
        os.remove(f.filename)
        return render_template('classification_output.html', filename=f.filename, result=res, nnname=__NN_NAME)


if __name__ == '__main__':
    app.run(debug=True)
