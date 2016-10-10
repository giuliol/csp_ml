from src.tools.dataset_helper import DatasetHelper
from src.tools.perftools import *
from src.ml import mlp


def __self_test():
    mlperc = mlp.MultilayerPerceptron(494, 2, 64, 16)

    # training_set, training_labels, test_set, test_labels = DatasetLoader.load_archives(
    #     "res/set_5/healthy_training.tar.gz",
    #     "res/set_5/healthy_test.tar.gz",
    #     "res/set_5/stroke_training.tar.gz",
    #     "res/set_5/stroke_test.tar.gz")

    # mlperc.train(training_set, training_labels, 400)
    # mlperc.save("res/saved_nns/symmetry_64_16_2_multi_slice_4_and_5.dat")
    #
    #       symmetry_64_16.dat                      ###   97% accuracy, 1.125% false alarm, trained on set_4
    #       symmetry_64_16_multi_slice.dat          ###   98% accuracy, 0 false alarm,      trained on set_5
    #       symmetry_64_16_multi_slice_4_and_5.dat  ###   trained on (set 4 and 5)

    mlperc.load("res/saved_nns/symmetry_64_16_multi_slice_4_and_5.dat")

    print("evaluating...")

    stroke_test_set = DatasetHelper.load_archive("res/set_3/stroke_test.tar.gz", 1)
    healthy_test_set = DatasetHelper.load_archive("res/set_3/healthy_test.tar.gz", 1)
    # stroke_test_set = DatasetHelper.load_data("res/set_2/stroke/test", 1)
    # healthy_test_set = DatasetHelper.load_data("res/set_2/healthy/test", 1)

    test_mlp(mlperc, healthy_test_set, stroke_test_set)
