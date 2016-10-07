import matplotlib.pyplot as plt


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
    plt.show()
