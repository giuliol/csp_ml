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

    training_set, training_labels, test_set, test_labels = DatasetHelper.load_data("res/set_1/healthy/training"), \
                                                           DatasetHelper.generate_labels(700,
                                                                                         DatasetHelper.LABEL_HEALTHY), \
                                                           DatasetHelper.load_data("res/set_1/healthy/test"), \
                                                           DatasetHelper.generate_labels(300,
                                                                                         DatasetHelper.LABEL_HEALTHY)

    ae.train_net(training_set, training_labels, test_set, test_labels)


def dataset_helper_test():
    training_set = DatasetHelper.load_data("res/set_1/healthy/training")
    test_set = DatasetHelper.load_data("res/set_1/healthy/test")

    print(training_set.shape)
    print(test_set.shape)

    labels_healthy = DatasetHelper.generate_labels(len(training_set), DatasetHelper.LABEL_HEALTHY)
    labels_stroke = DatasetHelper.generate_labels(len(training_set), DatasetHelper.LABEL_STROKE)

    print(labels_healthy[1:10, :])
    print(labels_stroke[1:10, :])

    return 0


def __load_set1():
    training_set, training_labels, test_set, test_labels = DatasetHelper.load_data("res/set_1/healthy/training"), \
                                                           DatasetHelper.generate_labels(700,
                                                                                         DatasetHelper.LABEL_HEALTHY), \
                                                           DatasetHelper.load_data("res/set_1/healthy/test"), \
                                                           DatasetHelper.generate_labels(300,
                                                                                         DatasetHelper.LABEL_HEALTHY)
    return training_set, training_labels, test_set, test_labels


def __load_dummy():
    training_set, training_labels, test_set, test_labels = DatasetHelper.load_data("res/dummy/training"), \
                                                           DatasetHelper.generate_labels(30,
                                                                                         DatasetHelper.LABEL_HEALTHY), \
                                                           DatasetHelper.load_data("res/dummy/test"), \
                                                           DatasetHelper.generate_labels(2,
                                                                                         DatasetHelper.LABEL_HEALTHY)
    return training_set, training_labels, test_set, test_labels


def classification_test():
    import numpy as np

    """
    Inizializzazione: ho usato 16 antenne nel dataset. Il vettore di "feature" (le misure em.)
    ha dimensione 16*15*2 (modulo e fase)
    256 e 64 sono le dimensioni dei due layer dell'autoencoder
    """
    ae = autoencoder.Autoencoder_2Layers(480, 256, 64)

    """
    Carico il dataset. Uso due piccoli helper definiti sopra
    """
    training_set, training_labels, test_set, test_labels = __load_set1()

    """
    Addestro l'autoencoder sui dati di cervello sano: 128 è il batch size, dove batch è il training batch
    """
    ae.train_net(training_set, training_labels, test_set, test_labels, 128)

    """
    Carico il dataset con stroke
    """
    stroke_set = DatasetHelper.load_data("res/set_1/stroke")

    # """
    # Versione media delle distanze dal control set (sani)
    # Calcolo la proiezione di un sample stroke e uno healthy sul sottospazio dell'autoencoder
    # """
    # stroke_proj = ae.project(stroke_set[0, :])
    # healthy_proj = ae.project(test_set[0, :])
    #
    # stroke_proj_distance = 0.0
    # healthy_proj_distance = 0.0
    #
    # c = 0
    # for sample in training_set:
    #     proj = ae.project(sample)
    #     stroke_proj_distance += np.linalg.norm(stroke_proj - proj) / 30
    #     healthy_proj_distance += np.linalg.norm(healthy_proj - proj) / 30
    #     c += 1
    #     # print(c)
    #     if c > 30:
    #         break
    #
    # print("Avg. distance of a healthy sample from training samples: {}".format(healthy_proj_distance))
    # print("Avg. distance of a stroke sample from training samples: {}".format(stroke_proj_distance))

    """
    Versione con centro della classe
    """
    MAXITER = 20
    print("computing class center...")
    center = ae.compupte_class_center(training_set)

    print("computing average stroke distance...")
    i = 0
    d = 0.0
    for sample in stroke_set:
        proj = ae.project(sample)
        d += np.linalg.norm(proj - center) / MAXITER
        i += 1
        if i > MAXITER: break

    print("Avg. stroke distance={}".format(d))
    print("computing average control set distance...")
    i = 0
    d = 0.0
    for sample in training_set:
        proj = ae.project(sample)
        d += np.linalg.norm(proj - center) / MAXITER
        i += 1
        if i > MAXITER: break

    print("Avg. healthy distance={}".format(d))
