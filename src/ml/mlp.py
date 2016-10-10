import tflearn
import tensorflow as tf
import numpy as np
import os
from src.tools import xml_tools


class MultilayerPerceptron:
    input_layer = None

    symmetry = False

    def set_symmetry(self, symm):
        self.symmetry = symm

    def uses_symmetry_features(self):
        return self.symmetry

    def __init__(self, features, classes, *layers_size):
        """
        MultilayerPerceptron class constructor.
        Example: build a 480, 200, 100, 16, 2 layer size network:
            MultilayerPerceptron(480, 2, 200, 100, 16)
        :param features: Number of features (size of first layer)
        :param classes: Number of classes (size of last layer)
        :param layers_size: size of hidden layers.
        """
        # Building deep neural network

        self.nn = tflearn.input_data(shape=[None, features])
        for i, layer_size in enumerate(layers_size):
            self.nn = tflearn.fully_connected(self.nn, layer_size, activation='tanh',
                                              regularizer='L2', weight_decay=0.001)
            self.nn = tflearn.dropout(self.nn, 0.8)

        self.softmax = tflearn.fully_connected(self.nn, classes, activation='softmax')

        # Regression using SGD with learning rate decay and Top-3 accuracy
        self.sgd = tflearn.SGD(learning_rate=0.05, lr_decay=0.96, decay_step=1000)
        self.top_k = tflearn.metrics.Top_k(3)
        self.net = tflearn.regression(self.softmax, optimizer=self.sgd, metric=self.top_k,
                                      loss='categorical_crossentropy')

        self.model = tflearn.DNN(self.net, tensorboard_verbose=0)

    def train(self, training_set, training_labels, epochs):
        """
        Train the network with the provided train and test sets.
        Labels must be provided as one-hot vectors.
        :param training_set: the training set
        :param training_labels: the training labels
        :param epochs: Duration of training (in epochs).
        :return:
        """
        # Training
        self.model.fit(training_set, training_labels, n_epoch=epochs, validation_set=0.2,
                       show_metric=True, run_id="dense_model")

    def save(self, filename):
        """
        Save NN to file
        :param filename:
        :return:
        """
        self.model.save(filename)

    def load(self, filename):
        """
        Loads the NN from a provided file name
        :param filename:
        :return:
        """

        self.model.load(filename)

    def score(self, sample):
        """
        Returns the class scores for the provided sample
        :param sample:
        :return: class scores as ndarray
        """
        return np.asarray(self.model.predict([sample]))

    def classify(self, sample):
        """
        Returns the classification
        :param sample:
        :return:
        """
        scores = self.score(sample)
        return np.argmax(scores)

    def classify(self, sample, threshold):
        """
        Returns the classification comparing the ratio of the two scores.
        :param threshold: the threshold over which scores[0]/scores[1] returns 0
        :param sample: the sample to be classified
        :return:
        """
        scores = self.score(sample)
        if scores[0][0] / scores[0][1] > threshold:
            return 0
        else:
            return 1

    @staticmethod
    def load_folder(folder):
        """
        Returns a whole MultilayerPerceptron object parsing a folder
        :param folder:
        :return:
        """
        name = os.path.basename(folder)
        visible_layers, hidden_layers, symmetry = xml_tools.parse_topology_xml("{}/{}.xml".format(folder, name))
        out = MultilayerPerceptron(int(visible_layers[0]), int(visible_layers[1]), *hidden_layers)
        out.set_symmetry(symmetry)
        out.load("{}/{}.dat".format(folder, name))
        return out

    def destroy(self):
        tf.reset_default_graph()
