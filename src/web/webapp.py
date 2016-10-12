import os
from flask import Flask, render_template, request, Markup
from werkzeug import secure_filename
from src import mlp_wrapper

app = Flask(__name__)

root = "../../"

try:
    os.mkdir("{}/tmp".format(root))
except FileExistsError:
    print("Folder already exists")


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

        os.mkdir("{}userspace/saved_nns/{}/".format(root, nn_filename))

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

        return render_template('index.html', page_inner=content, nn_details=get_nn_details(nn_filename), evaluate=True)


@app.route('/classify')
def classify():
    select_html = ""
    for nn in get_saved_nns():
        select_html += "<option>{}</option>".format(nn)

    select_html = Markup(select_html)

    content = Markup(render_template('upload_sample.html', select=select_html))

    return render_template('index.html', page_inner=content, classify=True)


@app.route('/sample_classifier', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))

        nn_name = request.form['nn_name']

        nn_filepath = "{}userspace/saved_nns/{}".format(root, nn_name)

        print(
            "mlp_wrapper.classify(nn_filepath, f.filename) = {}".format(mlp_wrapper.classify(nn_filepath, f.filename)))
        if mlp_wrapper.classify(nn_filepath, f.filename):
            res = "STROKE"
            col = "red"
        else:
            res = "HEALTHY"
            col = "green"

        healthy, stroke = mlp_wrapper.score(nn_filepath, f.filename)
        os.remove(f.filename)

        content = Markup(
            render_template('classification_result.html', nn_name=nn_name, sample_name=f.filename, color=col,
                            result=res, stroke_score=stroke, healthy_score=healthy))

        return render_template('index.html', page_inner=content, nn_details=get_nn_details(nn_name))


def get_saved_nns():
    return [name for name in os.listdir("{}userspace/saved_nns/".format(root))
            if os.path.isdir(os.path.join("{}userspace/saved_nns/".format(root), name))]


def get_nn_details(nn_filename):
    details = mlp_wrapper.get_details(root, nn_filename)

    content = Markup(render_template('nn_details.html',
                                     nn_name=nn_filename,
                                     symmetry=details['symmetry'],
                                     input_layer=details['input_layer'],
                                     output_layer=details['output_layer'],
                                     hidden_layers=details['hidden_layers']))
    return content


if __name__ == '__main__':
    app.run(debug=True)
