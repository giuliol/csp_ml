from flask import Flask, render_template, request, Markup
from werkzeug import secure_filename
from src import mlp_wrapper
import os

app = Flask(__name__)

root = "../../"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train')
def train():
    content = Markup(render_template('train_new_network.html'))
    return render_template('index.html', page_inner=content, train=True)


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

        if mlp_wrapper.check_exists_nn(root, nn_filename):
            content = Markup(render_template('nn_exists_error.html', nn_name=nn_filename))
            return render_template('index.html', page_inner=content)

        mlp_wrapper.train_new(root, nn_filename, h_tr_filepath,
                              s_tr_filepath, epochs, symmetry, *hidden_layers)

        os.remove(h_tr_filepath)
        os.remove(s_tr_filepath)

        content = Markup(render_template('trained.html', nn_name=nn_filename))
        return render_template('index.html', train=True, page_inner=content)


@app.route('/evaluate')
def evaluate():
    select_html = ""
    for nn in get_saved_nns():
        select_html += "<option>{}</option>".format(nn)

    content = Markup(render_template('evaluate_existing_nn.html', select=Markup(select_html)))
    return render_template('index.html', page_inner=content, evaluate=True)


@app.route('/evaluate_existing', methods=['GET', 'POST'])
def evaluate_existing():
    if request.method == 'POST':
        nn_filename = request.form['nn_name']

        healthy_test_file = request.files['healthy_test']
        h_test_filepath = "{}tmp/{}".format(root, healthy_test_file.filename)
        healthy_test_file.save(h_test_filepath)

        stroke_test_file = request.files['stroke_test']
        s_test_filepath = "{}tmp/{}".format(root, stroke_test_file.filename)
        stroke_test_file.save(s_test_filepath)

        auc, figure = mlp_wrapper.evaluate_existing_nn(root, nn_filename, h_test_filepath, s_test_filepath)

        print("AUC: {}".format(auc))

        content = Markup(render_template('evaluate_results.html', nn_name=nn_filename, img=figure,
                                         healthy_set=healthy_test_file.filename,
                                         stroke_set=stroke_test_file.filename))

        os.remove(h_test_filepath)
        os.remove(s_test_filepath)
        return render_template('index.html', page_inner=content, evaluate=True)


@app.route('/classify')
def upload_file():
    select_html = ""
    for nn in get_saved_nns():
        select_html += "<option>{}</option>".format(nn)

    select_html = Markup(select_html)

    content = Markup(render_template('upload_sample.html', select=select_html))

    return render_template('index.html', page_inner=content, classify=True)


@app.route('/sample_classifier', methods=['GET', 'POST'])
def upload_file1():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))

        nn_name = request.form['nn_name']

        nn_filepath = root + "userspace/saved_nns/{}".format(nn_name)

        if not mlp_wrapper.classify(nn_filepath, f.filename):
            res = "STROKE"
            col = "red"
        else:
            res = "HEALTHY"
            col = "green"
        os.remove(f.filename)

        content = Markup(
            render_template('classification_result.html', nn_name=nn_name, sample_name=f.filename, color=col,
                            result=res))

        return render_template('index.html', page_inner=content)


def get_saved_nns():
    return [name for name in os.listdir("{}userspace/saved_nns/".format(root))
            if os.path.isdir(os.path.join("{}userspace/saved_nns/".format(root), name))]


if __name__ == '__main__':
    app.run(debug=True)
