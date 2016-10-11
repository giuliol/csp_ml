from src.tools.dataset_helper import DatasetHelper, DatasetLoader
from src.tools import xml_tools
from src.tools.perftools import *
from src.ml import mlp
import os
from src.tools.simulator_data_parser import DatParser


def __self_test():
    mlperc = mlp.MultilayerPerceptron(494, 2, 64, 16)

    # training_set, training_labels, test_set, test_labels = DatasetLoader.load_archives(
    #     "res/datasets/set_5/healthy_training.tar.gz",
    #     "res/datasets/set_5/healthy_test.tar.gz",
    #     "res/datasets/set_5/stroke_training.tar.gz",
    #     "res/datasets/set_5/stroke_test.tar.gz")

    # mlperc.train(training_set, training_labels, 400)
    # mlperc.save("res/saved_nns/symmetry_64_16_2_multi_slice_4_and_5.dat")
    #
    #       symmetry_64_16.dat                      ###   97% accuracy, 1.125% false alarm, trained on set_4
    #       symmetry_64_16_multi_slice.dat          ###   98% accuracy, 0 false alarm,      trained on set_5
    #       symmetry_64_16_multi_slice_4_and_5.dat  ###   trained on (set 4 and 5)

    mlperc.load("res/saved_nns/symmetry_64_16_multi_slice_4_and_5.dat")

    print("evaluating...")

    stroke_test_set = DatasetHelper.load_archive("res/datasets/set_3/stroke_test.tar.gz", 1)
    healthy_test_set = DatasetHelper.load_archive("res/datasets/set_3/healthy_test.tar.gz", 1)
    # stroke_test_set = DatasetHelper.load_data("res/datasets/set_2/stroke/test", 1)
    # healthy_test_set = DatasetHelper.load_data("res/datasets/set_2/healthy/test", 1)

    test_mlp(mlperc, healthy_test_set, stroke_test_set)


def train_new(root, name, healthy_training, stroke_training, epochs, symmetry, *hidden_layers):
    """
    Train new neural network
    :param root: root path (i.e., where userspace/ folder is)
    :param name: neural network name (a folder with this name will be created)
    :param healthy_training: path to healthy training set archive
    :param stroke_training:  path to stroke training set archive
    :param epochs: training epochs
    :param symmetry: uses symmetry features
    :param hidden_layers: list of integers describing the sizes of the hidden layers
    :return:
    """
    training_set, training_labels = DatasetLoader.load_archives_training(
        healthy_training, stroke_training, symmetry)
    input_layer_size = training_set.shape[1]
    mlperc = mlp.MultilayerPerceptron(input_layer_size, 2, *hidden_layers)

    mlperc.train(training_set, training_labels, epochs)
    try:
        os.mkdir("{}userspace/saved_nns/{}/".format(root, name))
    except FileExistsError:
        print("Folder already exists")
    xml_tools.create_topology_xml(root, name, symmetry, [input_layer_size, 2], *hidden_layers)
    mlperc.save("{}userspace/saved_nns/{}/{}.dat".format(root, name, name))
    mlperc.destroy()


def evaluate_existing_nn(root, name, healthy_test, stroke_test):
    """
    Evaluates an existing neural network and computes the ROC curve and the auc figure
    :param root: root path (i.e., where userspace/ folder is)
    :param name: neural network name (the related folder  must exist)
    :param healthy_test: path to healthy test set archive
    :param stroke_test: path to stroke test set archive
    :return: two values: auc, figure    where auc is the area under the ROC curve, and figure is the path of
             the generated png of the ROC curve
    """
    dir = "static/"
    files = os.listdir(dir)

    for file in files:
        if file.endswith(".png"):
            os.remove(os.path.join(dir, file))

    mlperc = mlp.MultilayerPerceptron.load_folder("{}userspace/saved_nns/{}".format(root, name))

    stroke_test_set = DatasetHelper.load_archive(stroke_test, mlperc.uses_symmetry_features())
    healthy_test_set = DatasetHelper.load_archive(healthy_test, mlperc.uses_symmetry_features())
    mlperc.destroy()

    return test_mlp(mlperc, healthy_test_set, stroke_test_set)


def check_exists_nn(root, nn_name):
    """
    Check if, given a name, the related folder exists
    :param root: root path (i.e., where userspace/ folder is)
    :param name: neural network name
    :return: True if it exists
    """
    names = [name for name in os.listdir(root + "userspace/saved_nns/")
             if os.path.isdir(os.path.join(root + "userspace/saved_nns/", name))]

    for name in names:
        if nn_name == name:
            return True

    return False


def classify(nn_filepath, sample_filepath):
    """
    Classifies a given sample, using the neural network with the provided name
    :param nn_filepath:
    :param sample_filepath:
    :return: 1 or 0 (classification result)
    """
    mlperc = mlp.MultilayerPerceptron.load_folder(nn_filepath)
    symmetry = mlperc.uses_symmetry_features()
    sample = DatParser.parse_file(sample_filepath, symmetry)
    out = mlperc.classify(sample, 1)
    mlperc.destroy()
    return out
