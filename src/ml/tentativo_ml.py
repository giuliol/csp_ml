import numpy as np
import tflearn
import tflearn.datasets.mnist as mnist
from src.tools.dataset_helper import DatasetHelper


def ml_sandbox():
    X, Y, testX, testY = DatasetHelper.load_data("res/dummy/training"), \
                         DatasetHelper.generate_labels(30, DatasetHelper.LABEL_HEALTHY), \
                         DatasetHelper.load_data("res/dummy/test"), \
                         DatasetHelper.generate_labels(2, DatasetHelper.LABEL_HEALTHY)


    # Building the encoder
    encoder = tflearn.input_data(shape=[None, 180])
    encoder = tflearn.fully_connected(encoder, 64)
    encoder = tflearn.fully_connected(encoder, 8)

    # Building the decoder
    decoder = tflearn.fully_connected(encoder, 64)
    decoder = tflearn.fully_connected(decoder, 180)

    # Regression, with mean square error
    net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.001,
                             loss='mean_square', metric=None)

    # Training the auto encoder
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(X, X, n_epoch=10, validation_set=(testX, testX),
              run_id="auto_encoder", batch_size=5)

    # Encoding X[0] for test
    print("\nTest encoding of X[0]:")
    # New model, re-using the same session, for weights sharing
    encoding_model = tflearn.DNN(encoder, session=model.session)
    pred = encoding_model.predict([X[0]])
    print(pred)

    # Testing the image reconstruction on new data (test set)
    print("\nVisualizing results after being encoded and decoded:")
    print(X[0])

    return pred, X[0]

    # print("pred.shape={}, X[0].shape={}".format(pred.shape,X[0].shape))
    testX = tflearn.data_utils.shuffle(testX)[0]

    # Applying encode and decode over test set
    encode_decode = model.predict(testX)

    # Compare original images with their reconstructions
    # f, a = plt.subplots(2, 10, figsize=(10, 2))
    # for i in range(10):
    #     a[0][i].imshow(np.reshape(testX[i], (28, 28)))
    #     a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    # f.show()
    # plt.draw()
    # plt.waitforbuttonpress()