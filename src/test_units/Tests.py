from src.tools import simulator_data_parser
from src.ml import tentativo_ml, tflearn_autoencoder, autoencoder
from src.tools.dataset_helper import DatasetHelper


def tools_test():
    print("------- TEST BEGIN -------")

    sv = simulator_data_parser.DatParser(10, 1000)
    sv.parse_file("res/ETotExt.SNR15.dat")

    vals = sv.values
    print(vals)
    print(sv.flatten())
    print("------- TEST   END -------")
    return 0


def tensorflow_test():
    tflearn_autoencoder.tflearn_tutorial()
    return 0


def _machine_learning_sandbox():
    pred, X0 = tentativo_ml.ml_sandbox()

    return 0


def autoencoder_test():
    ae = autoencoder.Autoencoder()

    X, Y, testX, testY = DatasetHelper.load_data("res/dummy_set/training"), \
                         DatasetHelper.generate_labels(30, DatasetHelper.LABEL_HEALTHY), \
                         DatasetHelper.load_data("res/dummy_set/test"), \
                         DatasetHelper.generate_labels(2, DatasetHelper.LABEL_HEALTHY)

    ae.train_net(X, Y, testX, testY)


def dataset_helper_test():
    training_set = DatasetHelper.load_data("res/dummy_set/training")
    test_set = DatasetHelper.load_data("res/dummy_set/test")

    print(training_set.shape)
    print(test_set.shape)

    labels_healthy = DatasetHelper.generate_labels(len(training_set), DatasetHelper.LABEL_HEALTHY)
    labels_stroke = DatasetHelper.generate_labels(len(training_set), DatasetHelper.LABEL_STROKE)

    print(labels_healthy[1:10, :])
    print(labels_stroke[1:10, :])

    return 0
