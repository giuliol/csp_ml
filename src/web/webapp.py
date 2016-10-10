from flask import Flask, render_template, request, Markup
from werkzeug import secure_filename
from src import mlp_main
import os

app = Flask(__name__)

root = "../../"

train_new_network_html = """
<div class="panel panel-primary">
    <div class="panel-heading">
        New custom Neural Net
    </div>
    <div class="panel-body">
        <form action="http://localhost:5000/new_neural_net" method="POST" enctype="multipart/form-data">
            <label> Neural network name </label> <input class="form-control" name="nn_filename" placeholder="nome.dat" />
            <label> Hidden Layers</label> <br> values separated by commas!<input class="form-control" name="nn_hidden_layers" placeholder="64, 16" />
            <br>
            <label> Compute symmetry features <input type="checkbox" value="True" name="symmetry"> </label>
            <br>
            <br>
            <label> Healthy training</label> <input type="file" name="healthy_training"/><br>
            <label> Stroke training</label> <input type="file" name="stroke_training"/><br>
            <label> No. of training epochs </label> <input class="form-control" name="epochs" placeholder="300" />

            <input class="btn btn-default" type="submit"/><br></form>

    </div>
</div>
"""

upload_sample_html = """
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

classification_result_html = """
<div class="panel panel-default">
    <div class="panel-heading">
        Using neural network: <i>{}</i>
    </div>
    <div class="panel-body">
        <div class="alert alert-info"> <i>{}</i> </div> has been classified as <b><font color="{}">{}</font></b>.

    </div>

"""

evaluate_existing_nn_html = """
<div class="panel panel-primary">
    <div class="panel-heading">
        Evaluate existing Neural Net
    </div>
    <div class="panel-body">
        <form action="http://localhost:5000/evaluate_existing" method="POST" enctype="multipart/form-data">
            <div>
            Choose an existing neural network.
            <select class="form-control" name="nn_name">
                    {}
             </select>
            </div>
            <br>
            <br>
            <label> Healthy training</label> <input type="file" name="healthy_training"/><br>
            <label> Stroke training</label> <input type="file" name="stroke_training"/><br>
            <input class="btn btn-default" type="submit"/><br></form>

    </div>
</div>
"""


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train')
def train():
    return render_template('index.html', page_inner=Markup(train_new_network_html), train=True)


@app.route('/new_neural_net', methods=['GET', 'POST'])
def new_nn():
    if request.method == 'POST':
        nn_filename = request.form['nn_filename']
        hidden_layers = [int(el) for el in request.form['nn_hidden_layers'].split(',')]
        symmetry = bool(request.form.getlist('symmetry'))

        healthy_training_file = request.files['healthy_training']
        h_tr_filepath = "{}tmp/{}".format(root, healthy_training_file.filename)
        healthy_training_file.save(h_tr_filepath)

        stroke_training_file = request.files['stroke_training']
        s_tr_filepath = "{}tmp/{}".format(root, stroke_training_file.filename)
        stroke_training_file.save(s_tr_filepath)
        epochs = int(request.form['epochs'])

        # TODO qualche bel controllino qui no eh?
        mlp_main.train_new(root, nn_filename, h_tr_filepath,
                           s_tr_filepath, epochs, symmetry, *hidden_layers)

        os.remove(h_tr_filepath)
        os.remove(s_tr_filepath)

        return render_template('index.html', train=True)


@app.route('/evaluate')
def evaluate():
    select_html = ""
    for nn in get_saved_nns():
        select_html += "<option>{}</option>".format(nn)
    return render_template('index.html', page_inner=Markup(evaluate_existing_nn_html.format(select_html)),
                           evaluate=True)


@app.route('/evaluate_existing', methods=['GET', 'POST'])
def evaluate_existing():
    nn_filename = request.form['nn_filename']
    healthy_training_file = request.files['healthy_training']
    h_tr_filepath = "{}tmp/{}".format(root, healthy_training_file.filename)
    healthy_training_file.save(h_tr_filepath)

    stroke_training_file = request.files['stroke_training']
    s_tr_filepath = "{}tmp/{}".format(root, stroke_training_file.filename)
    stroke_training_file.save(s_tr_filepath)

    mlp_main.test_existing()
    os.remove(h_tr_filepath)
    os.remove(s_tr_filepath)
    return


@app.route('/classify')
def upload_file():
    select_html = ""
    for nn in get_saved_nns():
        select_html += "<option>{}</option>".format(nn)

    return render_template('index.html', page_inner=Markup(upload_sample_html.format(select_html)), classify=True)


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
                               page_inner=Markup(classification_result_html.format(nn_name, f.filename, col, res)))


def get_saved_nns():
    return [name for name in os.listdir(root + "userspace/saved_nns/")
            if os.path.isdir(os.path.join(root + "userspace/saved_nns/", name))]
    # dirs = [x[0] for x in os.walk(root + "userspace/saved_nns/")]
    # print(dirs[1::])


if __name__ == '__main__':
    app.run(debug=True)
