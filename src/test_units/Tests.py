from src.ml import autoencoder, mlp
from src.tools.dataset_helpers import DatasetHelper
from src.tools.dataset_helpers import QuickLoader
from src.tools.perftools import *


def austin_test():
    """"""
    """
    Inizializzazione: ho usato 16 antenne nel dataset. Il vettore di "feature" (le misure em.)
    ha dimensione 494=16*15*2 (modulo e fase) + 14 feature di simmetria
    """
    mlperc = mlp.MultilayerPerceptron(494, 2, 64, 16)
    mlperc.load("res/saved_nns/symmetry_64_16_multi_slice_4_and_5.dat")
    healthy = DatasetHelper.load_data("res/datasets/austin_test/healthy", 1)
    stroke = DatasetHelper.load_data("res/datasets/austin_test/stroke", 1)

    print("healthy")
    for sample in healthy:
        print(mlperc.classify(sample, 1))

    print("stroke")
    for sample in stroke:
        print(mlperc.classify(sample, 1))

    do_ROC(mlperc, stroke, healthy)


def mlp_classification_test_with_symmetry_features():
    """"""

    mlperc = mlp.MultilayerPerceptron(494, 2, 64, 16)
    training_set, training_labels, test_set, test_labels = QuickLoader.load_archives(
        "res/datasets/set_5/healthy_training.tar.gz",
        "res/datasets/set_5/healthy_test.tar.gz",
        "res/datasets/set_5/stroke_training.tar.gz",
        "res/datasets/set_5/stroke_test.tar.gz")
    mlperc.train(training_set, training_labels, 400)
    mlperc.save("userspace/saved_nns/symmetry_64_16_2_multi_slice_4_and_5.dat")
    #
    #       symmetry_64_16.dat                      ###   97% accuracy, 1.125% false alarm, trained on set_4
    #       symmetry_64_16_multi_slice.dat          ###   98% accuracy, 0 false alarm,      trained on set_5
    #       symmetry_64_16_multi_slice_4_and_5.dat  ###   trained on (set 4 and 5)
    #

    # mlperc = mlp.MultilayerPerceptron.load_folder("userspace/saved_nns/symmetry_64_16_multi_slice_4_and_5")

    """
    Valuto le prestazioni e calcolo dati per curva ROC.
    parametro della curva: soglia con cui il classificatore decide, confrontando (rapporto) delle likelihood delle
    due classi.
    """
    print("evaluating...")

    stroke_test_set = DatasetHelper.load_archive("res/datasets/set_3/stroke_test.tar.gz", 1)
    healthy_test_set = DatasetHelper.load_archive("res/datasets/set_3/healthy_test.tar.gz", 1)
    # stroke_test_set = DatasetHelper.load_data("res/datasets/set_2/stroke/test", 1)
    # healthy_test_set = DatasetHelper.load_data("res/datasets/set_2/healthy/test", 1)

    test_mlp(mlperc, healthy_test_set, stroke_test_set)


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
    training_set, training_labels, test_set, test_labels = QuickLoader.set2()

    """
    Addestro l'autoencoder sui dati di cervello sano: 128 è il batch size, dove batch è il training batch
    """
    ae.train(training_set, training_labels, test_set, test_labels, 128)

    """
    Carico il dataset con stroke
    """
    stroke_set = DatasetHelper.load_data("res/datasets/set_2/stroke/training")

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
