import tflearn
import numpy as np


class Autoencoder:
    encoder = None
    decoder = None
    net = None
    model = None
    encoding_model = None
    healthy_prototype = None
    __trained = False

    def is_trained(self):
        return self.__trained

    def __init__(self, features, *layers_size):

        self.l_size = np.zeros((len(layers_size)))
        for i, l_size in enumerate(layers_size):
            self.l_size[i] = l_size

        self.encoding_size = self.l_size[len(self.l_size) - 1]

        # Building the encoder
        self.encoder = tflearn.input_data(shape=[None, features])
        for layer_size in self.l_size:
            self.encoder = tflearn.fully_connected(self.encoder, layer_size)

        # Building the decoder
        for i, layer_size in enumerate(self.l_size):
            if i > 0: self.decoder = tflearn.fully_connected(self.encoder, layer_size)
        self.decoder = tflearn.fully_connected(self.decoder, features)

        # Regression, with mean square error
        self.net = tflearn.regression(self.decoder, optimizer='adam', learning_rate=0.001,
                                      loss='mean_square', metric=None)

    def train(self, training_set, training_labels, test_set, test_labels, batch):
        # Training the auto encoder
        self.model = tflearn.DNN(self.net, tensorboard_verbose=0)
        self.model.fit(training_set, training_set, n_epoch=15, validation_set=(test_set, test_set),
                       run_id="auto_encoder", batch_size=batch)
        self.healthy_prototype = 0
        self.__trained = True
        self.encoding_model = tflearn.DNN(self.encoder, session=self.model.session)

    def get_encoding(self, sample):
        if self.is_trained():
            return np.asarray(self.encoding_model.predict([sample]))

        ValueError("Encoding model not yet learned!")
        return 0

    def compute_class_center(self, _set, MAXITER):
        size = len(_set)
        center = np.zeros(self.encoding_size)

        for i, sample in enumerate(_set):
            proj = self.get_encoding(sample).flatten()
            _center = center
            center = center + proj
            diff = np.linalg.norm(_center / i - center / (i + 1))
            if i > 0: print("Center moved of {}".format(diff))
            if (i >= MAXITER) or (diff < 0.1): break

        return center / MAXITER

    def classify(self, sample):
        return True

    def get_encoding_error(self, sample):
        if self.is_trained():
            reconstruction = np.asarray(self.model.predict([sample]))
            loss = np.linalg.norm(reconstruction - sample)
            return loss


class Autoencoder_2Layers:
    encoder = None
    decoder = None
    net = None
    model = None
    healthy_prototype = None
    __trained = False

    def is_trained(self):
        return self.__trained

    def __init__(self, features, layer1_size, layer2_size):

        self.l1_size = layer1_size
        self.l2_size = layer2_size

        # Building the encoder
        self.encoder = tflearn.input_data(shape=[None, features])
        self.encoder = tflearn.fully_connected(self.encoder, layer1_size)
        self.encoder = tflearn.fully_connected(self.encoder, layer2_size)

        # Building the decoder
        self.decoder = tflearn.fully_connected(self.encoder, layer1_size)
        self.decoder = tflearn.fully_connected(self.decoder, features)

        # Regression, with mean square error
        self.net = tflearn.regression(self.decoder, optimizer='adam', learning_rate=0.002,
                                      loss='mean_square', metric=None)

    def train_net(self, training_set, training_labels, test_set, test_labels, batch):
        # Training the auto encoder
        self.model = tflearn.DNN(self.net, tensorboard_verbose=0)
        self.model.fit(training_set, training_set, n_epoch=20, validation_set=(test_set, test_set),
                       run_id="auto_encoder", batch_size=batch)
        self.healthy_prototype = 0
        self.__trained = True

    def project(self, sample):
        if self.is_trained():
            encoding_model = tflearn.DNN(self.encoder, session=self.model.session)
            return np.asarray(encoding_model.predict([sample]))

        ValueError("Encoding model not yet learned!")
        return 0

    def compute_class_center(self, set, MAXITER):
        size = len(set)
        center = np.zeros((self.l2_size))

        for i, sample in enumerate(set):
            proj = self.project(sample).flatten()
            _center = center
            center = center + proj
            if i > 0: print("Center moved of {}".format(np.linalg.norm(_center / i - center / (i + 1))))
            if i >= MAXITER: break

        return center / MAXITER

    def classify(self, sample):
        return True
