import tflearn
import numpy as np


class MultilayerPerceptron:
    input_layer = None

    def __init__(self, features, classes, *layers_size):
        # def __init__(self, features, classes):
        # Building deep neural network

        self.nn = tflearn.input_data(shape=[None, features])
        for i, layer_size in enumerate(layers_size):
            self.nn = tflearn.fully_connected(self.nn, layer_size, activation='tanh',
                                              regularizer='L2', weight_decay=0.001)
            self.nn = tflearn.dropout(self.nn, 0.8)

        # self.dense1 = tflearn.fully_connected(self.input_layer, 64, activation='tanh',
        #                                       regularizer='L2', weight_decay=0.001)
        # self.dropout1 = tflearn.dropout(self.dense1, 0.8)
        # self.dense2 = tflearn.fully_connected(self.dropout1, 64, activation='tanh',
        #                                       regularizer='L2', weight_decay=0.001)
        # self.dropout2 = tflearn.dropout(self.dense2, 0.8)

        self.softmax = tflearn.fully_connected(self.nn, classes, activation='softmax')

        # Regression using SGD with learning rate decay and Top-3 accuracy
        self.sgd = tflearn.SGD(learning_rate=0.05, lr_decay=0.96, decay_step=1000)
        self.top_k = tflearn.metrics.Top_k(3)
        self.net = tflearn.regression(self.softmax, optimizer=self.sgd, metric=self.top_k,
                                      loss='categorical_crossentropy')

        self.model = tflearn.DNN(self.net, tensorboard_verbose=0)

    def train(self, training_set, training_labels, test_set, test_labels, epochs):
        # Training
        self.model.fit(training_set, training_labels, n_epoch=epochs, validation_set=(test_set, test_labels),
                       show_metric=True, run_id="dense_model")

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)

    def classify(self, sample):
        return np.asarray(self.model.predict([sample]))

    def evaluate(self, test_set, test_labels):
        return self.model.evaluate(test_set, test_labels, batch_size=64)
