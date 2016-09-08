import tflearn
import numpy as np


class Autoencoder_2Layers:
    encoder = None
    decoder = None
    net = None
    model = None
    healty_prototype = None
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
        self.net = tflearn.regression(self.decoder, optimizer='adam', learning_rate=0.001,
                                      loss='mean_square', metric=None)

    def train_net(self, training_set, training_labels, test_set, test_labels, batch):
        # Training the auto encoder
        self.model = tflearn.DNN(self.net, tensorboard_verbose=0)
        self.model.fit(training_set, training_set, n_epoch=10, validation_set=(test_set, test_set),
                       run_id="auto_encoder", batch_size=batch)
        self.healty_prototype = 0
        self.__trained = True

    def project(self, sample):
        if self.is_trained():
            encoding_model = tflearn.DNN(self.encoder, session=self.model.session)
            return np.asarray(encoding_model.predict([sample]))

        ValueError("Encoding model not yet learned!")
        return 0

    def compupte_class_center(self, set):
        size = len(set)
        center = np.zeros((self.l2_size))

        i = 0
        for sample in set:
            proj = self.project(sample).flatten()
            center = center + proj / 30
            i += 1
            if i > 30: break

        return center

def classify(self, sample):
    return True
