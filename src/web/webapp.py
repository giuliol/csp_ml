from flask import Flask, render_template, request, Markup
from werkzeug import secure_filename
from src.ml import mlp
from src.tools.dataset_helper import DatParser
import os

app = Flask(__name__)

mlperc = mlp.MultilayerPerceptron(494, 2, 64, 16)
__NN_NAME = "../../res/saved_nns/symmetry_64_16_multi_slice_4_and_5.dat"
mlperc.load(__NN_NAME)

upload_form = """
<div class="panel panel-default">
    <div class="panel-heading">
        Using {}
    </div>
    <div class="panel-body">
        <br>Choose a <i>.dat</i> file to classify.
        <br><br>
        <form action="http://localhost:5000/uploader" method="POST" enctype="multipart/form-data">
            <input type="file" name="file"/><br>
                <input class="btn btn-default" type="submit"/><br></form>
    </div>
</div>
"""

classification_result = """
<div class="panel panel-default">
    <div class="panel-heading">
        Using {}
    </div>
    <div class="panel-body">
        <i> {} </i> <br><br> has been classified as <b><font color="{}">{}</font></b>.

    </div>

"""

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train')
def train():
    return render_template('index.html', page_inner="TBD", train=True)


@app.route('/evaluate')
def evaluate():
    return render_template('index.html', page_inner="TBD", evaluate=True)


@app.route('/classify', methods=['GET', 'POST'])
def upload_file():
    return render_template('index.html', page_inner=Markup(upload_form.format(__NN_NAME)), classify=True)


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

        return render_template('index.html',
                               page_inner=Markup(classification_result.format(__NN_NAME, f.filename, col, res)))


if __name__ == '__main__':
    app.run(debug=True)
