import tflearn


class Autoencoder:
    encoder = None
    decoder = None
    net = None
    model = None
    healty_prototype = None

    def __init__(self):
        # Building the encoder
        self.encoder = tflearn.input_data(shape=[None, 180])
        self.encoder = tflearn.fully_connected(self.encoder, 64)
        self.encoder = tflearn.fully_connected(self.encoder, 8)

        # Building the decoder
        self.decoder = tflearn.fully_connected(self.encoder, 64)
        self.decoder = tflearn.fully_connected(self.decoder, 180)

        # Regression, with mean square error
        self.net = tflearn.regression(self.decoder, optimizer='adam', learning_rate=0.001,
                                      loss='mean_square', metric=None)

        self.dummy = 0

    def train_net(self, training_set, training_labels, test_set, test_labels):
        # Training the auto encoder
        self.model = tflearn.DNN(self.net, tensorboard_verbose=0)
        self.model.fit(training_set, training_set, n_epoch=10, validation_set=(test_set, test_set),
                       run_id="auto_encoder", batch_size=5)

        # TODO: creare e calcolare prototipo di healthy.

    def classify(self, sample):
        return True
