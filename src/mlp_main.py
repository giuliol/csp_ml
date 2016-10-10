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


# TODO Controllare che non esista name
def train_new(name, visible_layers, hidden_layers, healthy_training, healthy_test, stroke_training, stroke_test,
              epochs):
    mlperc = mlp.MultilayerPerceptron(visible_layers[0], visible_layers[1], *hidden_layers)
    training_set, training_labels, test_set, test_labels = DatasetLoader.load_archives(
        healthy_training,
        healthy_test,
        stroke_training,
        stroke_test)

    mlperc.train(training_set, training_labels, epochs)
    try:
        os.mkdir("userspace/saved_nns/{}/".format(name))
    except FileExistsError:
        print("Folder already exists")
    xml_tools.create_topology_xml(name, visible_layers, *hidden_layers)
    mlperc.save("userspace/saved_nns/{}/{}.dat".format(name, name))


def test_existing(name, healthy_test, stroke_test):
    mlperc = mlp.MultilayerPerceptron.load_folder("userspace/saved_nns/{}".format(name))
    stroke_test_set = DatasetHelper.load_archive(stroke_test, 1)
    healthy_test_set = DatasetHelper.load_archive(healthy_test, 1)
    return


def classify(nn_filepath, sample_filepath):
    mlperc = mlp.MultilayerPerceptron.load_folder(nn_filepath)
    sample = DatParser.parse_file(sample_filepath, 1)
    out = mlperc.classify(sample, 1)
    mlperc.destroy()
    return out

# __self_test()

# train_new("test", [494, 2], [64, 16, 2],
#           "res/datasets/set_5/healthy_training.tar.gz",
#           "res/datasets/set_5/healthy_test.tar.gz",
#           "res/datasets/set_5/stroke_training.tar.gz",
#           "res/datasets/set_5/stroke_test.tar.gz",
#           2
#           )

# mlp.MultilayerPerceptron.load_folder("userspace/saved_nns/test")
