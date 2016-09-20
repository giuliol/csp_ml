from src.tools import simulator_data_parser
from src.ml import autoencoder, mlp
from src.tools.dataset_helper import DatasetHelper
import numpy as np


def tools_test():
    print("------- TEST BEGIN -------")

    sv = simulator_data_parser.DatParser(10, 1000)
    sv.parse_file("res/ETotExt.SNR15.dat")

    vals = sv.values
    print(vals)
    print(sv.flatten())
    print("------- TEST   END -------")
    return 0


def autoencoder_test():
    ae = autoencoder.Autoencoder()

    training_set, training_labels, test_set, test_labels = DatasetHelper.load_data("res/set_1/healthy/training"), \
                                                           DatasetHelper.generate_labels(700,
                                                                                         DatasetHelper.LABEL_HEALTHY), \
                                                           DatasetHelper.load_data("res/set_1/healthy/test"), \
                                                           DatasetHelper.generate_labels(300,
                                                                                         DatasetHelper.LABEL_HEALTHY)

    ae.train(training_set, training_labels, test_set, test_labels)


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


def __load_set5():
    import numpy as np

    healthy_training_set, healthy_training_labels, healthy_test_set, healthy_test_labels = DatasetHelper.load_archive(
        "res/set_5/healthy_training.tar.gz", 1), \
                                                                                           DatasetHelper.generate_labels(
                                                                                               700,
                                                                                               DatasetHelper.LABEL_HEALTHY), \
                                                                                           DatasetHelper.load_archive(
                                                                                               "res/set_4/healthy_test.tar.gz",
                                                                                               1), \
                                                                                           DatasetHelper.generate_labels(
                                                                                               300,
                                                                                               DatasetHelper.LABEL_HEALTHY)

    stroke_training_set, stroke_training_labels, stroke_test_set, stroke_test_labels = DatasetHelper.load_archive(
        "res/set_5/stroke_training.tar.gz", 1), \
                                                                                       DatasetHelper.generate_labels(
                                                                                           700,
                                                                                           DatasetHelper.LABEL_STROKE), \
                                                                                       DatasetHelper.load_archive(
                                                                                           "res/set_4/stroke_test.tar.gz",
                                                                                           1), \
                                                                                       DatasetHelper.generate_labels(
                                                                                           300,
                                                                                           DatasetHelper.LABEL_STROKE)

    training_set = np.row_stack((healthy_training_set, stroke_training_set))
    test_set = np.row_stack((healthy_test_set, stroke_test_set))

    training_labels = np.row_stack((healthy_training_labels, stroke_training_labels))
    test_labels = np.row_stack((healthy_test_labels, stroke_test_labels))
    return training_set, training_labels, test_set, test_labels


def __load_set4():
    import numpy as np

    healthy_training_set, healthy_training_labels, healthy_test_set, healthy_test_labels = DatasetHelper.load_archive(
        "res/set_4/healthy_training.tar.gz", 1), \
                                                                                           DatasetHelper.generate_labels(
                                                                                               1600,
                                                                                               DatasetHelper.LABEL_HEALTHY), \
                                                                                           DatasetHelper.load_archive(
                                                                                               "res/set_4/healthy_test.tar.gz",
                                                                                               1), \
                                                                                           DatasetHelper.generate_labels(
                                                                                               400,
                                                                                               DatasetHelper.LABEL_HEALTHY)

    stroke_training_set, stroke_training_labels, stroke_test_set, stroke_test_labels = DatasetHelper.load_archive(
        "res/set_4/stroke_training.tar.gz", 1), \
                                                                                       DatasetHelper.generate_labels(
                                                                                           1600,
                                                                                           DatasetHelper.LABEL_STROKE), \
                                                                                       DatasetHelper.load_archive(
                                                                                           "res/set_4/stroke_test.tar.gz",
                                                                                           1), \
                                                                                       DatasetHelper.generate_labels(
                                                                                           400,
                                                                                           DatasetHelper.LABEL_STROKE)

    training_set = np.row_stack((healthy_training_set, stroke_training_set))
    test_set = np.row_stack((healthy_test_set, stroke_test_set))

    training_labels = np.row_stack((healthy_training_labels, stroke_training_labels))
    test_labels = np.row_stack((healthy_test_labels, stroke_test_labels))
    return training_set, training_labels, test_set, test_labels


def __load_set3():
    import numpy as np

    healthy_training_set, healthy_training_labels, healthy_test_set, healthy_test_labels = DatasetHelper.load_archive(
        "res/set_3/healthy_training.tar.gz", 0), \
                                                                                           DatasetHelper.generate_labels(
                                                                                               700,
                                                                                               DatasetHelper.LABEL_HEALTHY), \
                                                                                           DatasetHelper.load_archive(
                                                                                               "res/set_3/healthy_test.tar.gz",
                                                                                               0), \
                                                                                           DatasetHelper.generate_labels(
                                                                                               300,
                                                                                               DatasetHelper.LABEL_HEALTHY)

    stroke_training_set, stroke_training_labels, stroke_test_set, stroke_test_labels = DatasetHelper.load_archive(
        "res/set_3/stroke_training.tar.gz", 0), \
                                                                                       DatasetHelper.generate_labels(
                                                                                           700,
                                                                                           DatasetHelper.LABEL_STROKE), \
                                                                                       DatasetHelper.load_archive(
                                                                                           "res/set_3/stroke_test.tar.gz",
                                                                                           0), \
                                                                                       DatasetHelper.generate_labels(
                                                                                           300,
                                                                                           DatasetHelper.LABEL_STROKE)

    training_set = np.row_stack((healthy_training_set, stroke_training_set))
    test_set = np.row_stack((healthy_test_set, stroke_test_set))

    training_labels = np.row_stack((healthy_training_labels, stroke_training_labels))
    test_labels = np.row_stack((healthy_test_labels, stroke_test_labels))
    return training_set, training_labels, test_set, test_labels


def __load_set2b():
    import numpy as np
    healthy_training_set, healthy_training_labels, healthy_test_set, healthy_test_labels = DatasetHelper.load_data(
        "res/set_2/healthy/training", 0), \
                                                                                           DatasetHelper.generate_labels(
                                                                                               400,
                                                                                               DatasetHelper.LABEL_HEALTHY), \
                                                                                           DatasetHelper.load_data(
                                                                                               "res/set_2/healthy/test",
                                                                                               0), \
                                                                                           DatasetHelper.generate_labels(
                                                                                               100,
                                                                                               DatasetHelper.LABEL_HEALTHY)

    stroke_training_set, stroke_training_labels, stroke_test_set, stroke_test_labels = DatasetHelper.load_data(
        "res/set_2/stroke/training", 0), \
                                                                                       DatasetHelper.generate_labels(
                                                                                           400,
                                                                                           DatasetHelper.LABEL_STROKE), \
                                                                                       DatasetHelper.load_data(
                                                                                           "res/set_2/stroke/test", 0), \
                                                                                       DatasetHelper.generate_labels(
                                                                                           100,
                                                                                           DatasetHelper.LABEL_STROKE)

    training_set = np.row_stack((healthy_training_set, stroke_training_set))
    test_set = np.row_stack((healthy_test_set, stroke_test_set))

    training_labels = np.row_stack((healthy_training_labels, stroke_training_labels))
    test_labels = np.row_stack((healthy_test_labels, stroke_test_labels))

    return training_set, training_labels, test_set, test_labels


def __load_set2():
    training_set, training_labels, test_set, test_labels = DatasetHelper.load_data("res/set_2/healthy/training", 0), \
                                                           DatasetHelper.generate_labels(400,
                                                                                         DatasetHelper.LABEL_HEALTHY), \
                                                           DatasetHelper.load_data("res/set_2/healthy/test", 0), \
                                                           DatasetHelper.generate_labels(100,
                                                                                         DatasetHelper.LABEL_HEALTHY)
    return training_set, training_labels, test_set, test_labels


def __load_set1():
    training_set, training_labels, test_set, test_labels = DatasetHelper.load_data("res/set_1/healthy/training", 0), \
                                                           DatasetHelper.generate_labels(700,
                                                                                         DatasetHelper.LABEL_HEALTHY), \
                                                           DatasetHelper.load_data("res/set_1/healthy/test", 0), \
                                                           DatasetHelper.generate_labels(300,
                                                                                         DatasetHelper.LABEL_HEALTHY)
    return training_set, training_labels, test_set, test_labels


def __load_dummy():
    training_set, training_labels, test_set, test_labels = DatasetHelper.load_data("res/dummy/training", 0), \
                                                           DatasetHelper.generate_labels(30,
                                                                                         DatasetHelper.LABEL_HEALTHY), \
                                                           DatasetHelper.load_data("res/dummy/test", 0), \
                                                           DatasetHelper.generate_labels(2,
                                                                                         DatasetHelper.LABEL_HEALTHY)
    return training_set, training_labels, test_set, test_labels


def mlp_classification_test_with_symmetry_features():
    """"""
    """
    Inizializzazione: ho usato 16 antenne nel dataset. Il vettore di "feature" (le misure em.)
    ha dimensione 494=16*15*2 (modulo e fase) + 14 feature di simmetria
    """
    mlperc = mlp.MultilayerPerceptron(494, 2, 64, 16)

    """
    Carico il dataset. Uso helper definito sopra
    """
    training_set, training_labels, test_set, test_labels = __load_set5()

    """
    Addestro l'autoencoder sui dati di cervello sano:
    """
    # mlperc.train(training_set, training_labels, 400)
    # mlperc.save("res/saved_nns/symmetry_64_16_multi_slice.dat")
    mlperc.load("res/saved_nns/symmetry_64_16_multi_slice.dat")  # mlp.64_16.dat 97% accuracy, trained on set_4

    print("evaluating...")
    thresholds = np.hstack(
        (np.array([0.001, 0.005, 0.01, 0.02, 0.08, 0.095, 0.096, 0.0991, 1, 10, 20]), np.linspace(60, 600, num=5)))

    correct_decisions = np.zeros(thresholds.shape)
    true_positives = np.zeros(thresholds.shape)
    true_negatives = np.zeros(thresholds.shape)

    false_negatives = np.zeros(thresholds.shape)
    false_positives = np.zeros(thresholds.shape)

    stroke_test_set = DatasetHelper.load_archive("res/set_5/stroke_test.tar.gz", 1)
    healthy_test_set = DatasetHelper.load_archive("res/set_5/healthy_test.tar.gz", 1)

    for i, THRESH in enumerate(thresholds):

        for sample in stroke_test_set:
            if mlperc.classify(sample, THRESH):
                false_negatives[i] += 1
            else:
                correct_decisions[i] += 1
                true_positives[i] += 1

        for sample in healthy_test_set:
            if mlperc.classify(sample, THRESH):
                correct_decisions[i] += 1
                true_negatives[i] += 1
            else:
                false_positives[i] += 1

        total = (len(healthy_test_set) + len(stroke_test_set))
        correct_decisions[i] /= total
        false_negatives[i] /= total
        false_positives[i] /= total
        true_negatives[i] /= len(healthy_test_set)
        true_positives[i] /= len(stroke_test_set)
        print("threshold {}, corr.{}".format(THRESH, correct_decisions[i]))
        # print("Total {} test samples.\nCorrect decisions:{} false alarms:{} missed detections:{}".format(total,
        #                                                                                                  correct_decisions,
        #                                                                                                  false_positives,
        #                                                                                                  false_negatives))
    with open("res/ROC.txt", "w") as f:
        f.write("# thresholds, true_negatives, true_positives, false_negatives, false_positives\n")
        for i, t in enumerate(thresholds):
            f.write("{} , {} , {} , {} , {}\n".format(t, true_negatives[i], true_positives[i], false_negatives[i],
                                                      false_positives[i]))


def mlp_classification_test():
    """"""
    """
    Inizializzazione: ho usato 16 antenne nel dataset. Il vettore di "feature" (le misure em.)
    ha dimensione 480=16*15*2 (modulo e fase)
    """
    mlperc = mlp.MultilayerPerceptron(480, 2, 64, 16)

    """
    Carico il dataset. Uso helper definito sopra
    """
    training_set, training_labels, test_set, test_labels = __load_set3()

    """
    Addestro l'autoencoder sui dati di cervello sano:
    """
    # mlperc.train(training_set, training_labels, test_set, test_labels, 400)
    # mlperc.save("res/saved_nns/mlp.64_16.dat")
    mlperc.load("res/saved_nns/mlp.64_16.dat")  # mlp.64_16.dat 97% accuracy

    print("evaluating...")

    correct_decisions = 0
    missed_detections = 0
    false_alarms = 0

    stroke_test_set = DatasetHelper.load_archive("res/set_3/stroke_test.tar.gz", 0)
    # stroke_test_set = DatasetHelper.load_data("res/set_2/stroke/test")
    for sample in stroke_test_set:
        if mlperc.classify(sample):
            missed_detections += 1
        else:
            correct_decisions += 1

    healthy_test_set = DatasetHelper.load_archive("res/set_3/healthy_test.tar.gz", 0)
    # healthy_test_set = DatasetHelper.load_data("res/set_2/healthy/test")
    for sample in healthy_test_set:
        if mlperc.classify(sample):
            correct_decisions += 1
        else:
            false_alarms += 1

    total = (len(healthy_test_set) + len(stroke_test_set))
    correct_decisions /= total
    missed_detections /= total
    false_alarms /= total
    print("Total {} test samples.\nCorrect decisions:{} false alarms:{} missed detections:{}".format(total,
                                                                                                     correct_decisions,
                                                                                                     false_alarms,
                                                                                                     missed_detections))


def autoencoder_classification_test():
    """"""
    """
    Inizializzazione: ho usato 16 antenne nel dataset. Il vettore di "feature" (le misure em.)
    ha dimensione 16*15*2 (modulo e fase)
    256 e 64 sono le dimensioni dei due layer dell'autoencoder
    """
    # ae = autoencoder.Autoencoder( features, [dimensioni dei vari layer, a piacere])
    ae = autoencoder.Autoencoder(480, 256, 64)

    """
    Carico il dataset. Uso due piccoli helper definiti sopra
    """
    training_set, training_labels, test_set, test_labels = __load_set2()

    """
    Addestro l'autoencoder sui dati di cervello sano: 128 è il batch size, dove batch è il training batch
    """
    ae.train(training_set, training_labels, test_set, test_labels, 128)

    """
    Carico il dataset con stroke
    """
    stroke_set = DatasetHelper.load_data("res/set_2/stroke/training")

    """
    Versione con loss come metrica di bontà
    """

    # avg. loss su control set
    avg_healthy_loss = 0
    for i, sample in enumerate(training_set):
        avg_healthy_loss += ae.get_encoding_error(sample)
    avg_healthy_loss /= len(training_set)

    # avg. loss su stroke set
    avg_stroke_loss = 0
    for i, sample in enumerate(stroke_set):
        avg_stroke_loss += ae.get_encoding_error(sample)
    avg_stroke_loss /= len(stroke_set)

    print("Avg. Healthy loss={}, avg. Stroke loss={}".format(avg_healthy_loss, avg_stroke_loss))

    """
    Versione media delle distanze dal control set (sani)
    Calcolo la proiezione di un sample stroke e uno healthy sul sottospazio dell'autoencoder
    """
    # stroke_proj = ae.get_encoding(stroke_set[0, :])
    # healthy_proj = ae.get_encoding(test_set[0, :])
    #
    # stroke_proj_distance = 0.0
    # healthy_proj_distance = 0.0
    #
    # c = 0
    # for sample in training_set:
    #     proj = ae.get_encoding(sample)
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
    # MAXITER = 20
    # print("computing class center...")
    # center = ae.compute_class_center(training_set, 50)
    #
    # print("computing average stroke distance...")
    # d = 0.0
    # for i, sample in enumerate(stroke_set):
    #     proj = ae.get_encoding(sample)
    #     _d = d
    #     d += np.linalg.norm(proj - center)
    #     if i > 1:
    #         diff = np.abs(d / i - _d / (i + 1))
    #         print("Avg. stroke distance moved {}".format(diff))
    #         if (i > MAXITER) or (diff < 3): break
    #
    # stroke_d = d / MAXITER
    #
    # print("Avg. stroke distance={}".format(stroke_d))
    # print("computing average control set distance...")
    # d = 0.0
    # for i, sample in enumerate(test_set):
    #     proj = ae.get_encoding(sample)
    #     _d = d
    #     d += np.linalg.norm(proj - center)
    #     if i > 1:
    #         diff = np.abs(d / i - _d / (i + 1))
    #         print("Avg. healthy distance moved {}".format(diff))
    #         if (i > MAXITER) or (diff < 3): break
    #
    # healthy_d = d / MAXITER
    # THRESH = stroke_d - (stroke_d - healthy_d) / 2
    #
    # print("Avg. healthy distance={}".format(healthy_d))
    # print("Test classification (THRESH={}):".format(THRESH))
    #
    # correct_detections = 0.0
    # missed_detections = 0.0
    # false_alarms = 0.0
    #
    # i = 0
    # for sample in stroke_set:
    #     proj = ae.get_encoding(sample)
    #     if np.linalg.norm(proj - center) > THRESH:
    #         correct_detections += 1
    #     else:
    #         missed_detections += 1
    #     i += 1
    #     if i >= MAXITER: break
    #
    # i = 0
    # for sample in test_set:
    #     proj = ae.get_encoding(sample)
    #     if np.linalg.norm(proj - center) > THRESH:
    #         false_alarms += 1
    #     else:
    #         correct_detections += 1
    #     i += 1
    #     if i >= MAXITER: break
    #
    # correct_detections /= 2 * MAXITER
    # false_alarms /= 2 * MAXITER
    # missed_detections /= 2 * MAXITER
    # print("Correct detections: {}\nFalse alarms: {}\nMissed Detections: {}".format(correct_detections, false_alarms,
    #                                                                                missed_detections))
