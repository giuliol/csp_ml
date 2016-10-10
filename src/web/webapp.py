from flask import Flask, render_template, request, Markup
from werkzeug import secure_filename
from src import mlp_main
import os

app = Flask(__name__)

root = "../../"

# mlperc = mlp.MultilayerPerceptron(494, 2, 64, 16)
__NN_NAME = "symmetry_64_16_multi_slice_4_and_5"
# mlperc.load(__NN_NAME)

upload_dataset = """
<div class="panel panel-primary">
    <div class="panel-heading">
        New custom Neural Net
    </div>
    <div class="panel-body">
        <form action="http://localhost:5000/new_neural_net" method="POST" enctype="multipart/form-data">
            <label> Neural network name </label> <input class="form-control" name="nn_filename" placeholder="nome.dat" />
            <label> Visible Layers</label> <br> values separated by commas!<input class="form-control" name="nn_visible_layers" placeholder="494, 2" />
            <label> Hidden Layers</label> <br> values separated by commas!<input class="form-control" name="nn_hidden_layers" placeholder="64, 16" />
            <br>
            <label> Healthy training</label> <input type="file" name="healthy_training"/><br>
            <label> Stroke training</label> <input type="file" name="stroke_training"/><br>
            <input class="btn btn-default" type="submit"/><br></form>

    </div>
</div>
"""

upload_sample = """
<div class="panel panel-primary">
    <div class="panel-heading">
        Upload a .dat file, choose the neural network and classify
    </div>
    <div class="panel-body">

        <form action="http://localhost:5000/sample_classifier" method="POST" enctype="multipart/form-data">
            <div>
                    Choose an existing neural network.
                <select class="form-control" name="nn_name">
                    {}
                </select>
            </div>
            <br><br>
            <div>
                Choose a <i>.dat</i> file to classify.
                <input type="file" name="file"/><br>
            </div>
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
        <div class="alert alert-info"> <i>{}</i> </div> has been classified as <b><font color="{}">{}</font></b>.

    </div>

"""


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train')
def train():
    return render_template('index.html', page_inner=Markup(upload_dataset), train=True)


@app.route('/evaluate')
def evaluate():
    return render_template('index.html', page_inner="TBD", evaluate=True)


@app.route('/classify', methods=['GET', 'POST'])
def upload_file():
    select_html = ""
    for nn in get_saved_nns():
        select_html += "<option>{}</option>".format(nn)

    return render_template('index.html', page_inner=Markup(upload_sample.format(select_html)), classify=True)


@app.route('/sample_classifier', methods=['GET', 'POST'])
def upload_file1():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))

        nn_name = request.form['nn_name']

        nn_filepath = root + "userspace/saved_nns/{}".format(nn_name)

        if not mlp_main.classify(nn_filepath, f.filename):
            res = "STROKE"
            col = "red"
        else:
            res = "HEALTHY"
            col = "green"
        os.remove(f.filename)

        return render_template('index.html',
                               page_inner=Markup(classification_result.format(nn_name, f.filename, col, res)))


def get_saved_nns():
    return [name for name in os.listdir(root + "userspace/saved_nns/")
            if os.path.isdir(os.path.join(root + "userspace/saved_nns/", name))]
    # dirs = [x[0] for x in os.walk(root + "userspace/saved_nns/")]
    # print(dirs[1::])


if __name__ == '__main__':
    app.run(debug=True)
