import matplotlib.pyplot as plt
import numpy as np
import random
import string
import os


def test_mlp(mlperc, healthy_test_set, stroke_test_set):
    thresholds = np.hstack(
        (np.array([0.001, 0.005, 0.01, 0.02, 0.08, 0.095, 0.096, 0.0991, 1, 10, 20]), np.linspace(60, 600, num=5)))

    correct_decisions = np.zeros(thresholds.shape)
    true_positives = np.zeros(thresholds.shape)
    true_negatives = np.zeros(thresholds.shape)

    false_negatives = np.zeros(thresholds.shape)
    false_positives = np.zeros(thresholds.shape)

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
        false_negatives[i] /= len(stroke_test_set)
        false_positives[i] /= len(healthy_test_set)
        true_negatives[i] /= len(healthy_test_set)
        true_positives[i] /= len(stroke_test_set)

        print("threshold {}, corr.{}".format(THRESH, correct_decisions[i]))

    return plot_ROC(correct_decisions, false_negatives, false_positives, true_negatives, true_positives, thresholds)


def plot_ROC(correct_decisions, false_negatives, false_positives, true_negatives, true_positives, thresholds):
    plt.fill_between(false_positives, 0, true_positives, facecolor='#ADD8E6')
    plt.plot(false_positives, true_positives, 'ro', clip_on=False)
    plt.xlabel("False Positives")
    plt.ylabel("True Positives")

    auc = 0
    true_positives = true_positives[::-1]
    false_positives = false_positives[::-1]
    thresholds = thresholds[::-1]
    correct_decisions = correct_decisions[::-1]

    for i, tp in enumerate(true_positives[1:]):
        rec = (false_positives[i + 1] - false_positives[i]) * (true_positives[i])
        tri = (false_positives[i + 1] - false_positives[i]) * (true_positives[i + 1] - true_positives[i]) / 2.0
        auc += rec + tri
        if thresholds[i + 1] == 1.0:
            plt.plot(false_positives[i + 1], true_positives[i + 1], 'gd')
            cd = correct_decisions[i + 1]

    plt.text(0.2, 0.5, "AUC: {}\n{}% correct decisions at 1.0 threshold".format(auc, cd))
    plt.grid()

    try:
        figurepath = "static/{}.png".format(randomword(10))
        plt.savefig(figurepath)

    except FileNotFoundError:
        print("File not found error: does static exist in {} ?".format(os.getcwd()))
        figurepath = None

    plt.close()
    return auc, figurepath


def randomword(length):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))
