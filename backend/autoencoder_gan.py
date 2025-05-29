import os # to interact with os
import keras # to build neural networks
from keras import layers # contains various layer types (e.g., Dense, Conv2D) to build nn
import keras.backend.tensorflow_backend as tb # to modify TensorFlow's backend behavior

from backend.utils import get_filepath, prediction_check, get_metrics, load_model, show_confusion_matrix, \
    save_model_components_to_files

def create_autoencoder_gan(x, x_train, x_test, y_train, y_test):
    # set parameters for training
    dense_dimension = 256  # encoding dimension (each image filtered 256 times to get featured map)
    # This is the number of neurons in the encoder part

    # create input layer using Keras (number of columns in dataset) = 10
    input_layer = keras.Input(shape=(x.shape[1],))

    # creates fully-connected (dense) layer using Keras.layers and ReLu activation func (0 to 1)
    d_encoding_layer = layers.Dense(dense_dimension, activation='relu')(input_layer)
    # d - discriminator
    # it compresses the data
    # relu gives non-linearity to the model, enabling it to generate complex data.
    # keeps only positive values, throws away all negatives (turns them into 0)

    # creates fully-connected (dense) layer using Keras.layers and softmax activation func (0 to 1), input size = 3
    g_decoding_layer = layers.Dense(y_train.shape[1], activation='softmax')(d_encoding_layer)
    # g - generator
    # it reconstructs or predicts the data.
    # SoftMax activation function transforms raw outputs of the neural network into a vector of probabilities
    # raw [2.5, 1.2, 0.3] -> softmax [0.73, 0.20, 0.07]

    # creates keras training model using input layer (10) and dense layer (3) with softmax activation function
    # combines encoder + decoder into a full model — input to output.
    autoencoder_gan_model = keras.Model(input_layer, g_decoding_layer)

    # create keras training model using input layer (10) and dense layer (256) with relu activation function
    # first half – the compressor
    encoder_gan_model = keras.Model(input_layer, d_encoding_layer)

    # create another input layer (256) using Keras
    encoded_input_layer = keras.Input(shape=(dense_dimension,))

    # merge decoder (last layer) with encoded input layer to take encoded data as input and produce reconstructed output
    last_decoder_layer = autoencoder_gan_model.layers[-1]  # holding last layer

    # second half – the rebuilder.
    decoder_gan_model = keras.Model(encoded_input_layer, last_decoder_layer(encoded_input_layer))
    # last layer represents the decoder part for rebuilding input from encoded representation

    # compile model
    autoencoder_gan_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # compile model
    # adam optimizer - dynamically adjusts the learning rate (internal weights) during training based on loss func
    # loss crossentropy - how far off the model's predictions are from the true labels
    # accuracy - measures the percentage of correctly classified samples out of the total number of samples between 0 and 1

    # train (30 epochs) autoencoder model with Keras, dataset x as input and Y as desired output
    training_results = autoencoder_gan_model.fit(x_train, y_train, epochs=30, batch_size=32, shuffle=True, verbose=0,
                           validation_data=(x_test, y_test))
    # The batch size defines the number of samples that will be propagated through the network.
    # It takes the first 32 samples from the training dataset and trains the network.
    # Next, it takes the second 32 samples (from 32nd to 64th) and trains the network again.
    # Usually networks train faster with mini-batches, because we update the weights after each propagation.

    # save weights, model and history to files
    save_model_components_to_files(autoencoder_gan_model, training_results, "autoencoder_model")

    return autoencoder_gan_model


def train_autoencoder_gan(x, x_train, x_test, y_train, y_test, create_model=False, show_plot=False):
    # x - entire dataset
    # change TensorFlow setting
    tb._SYMBOLIC_SCOPE.value = True  # operations should be added to the TensorFlow symbolic graph

    model_name = "autoencoder_gan_model"
    # check if the model is already trained if not create_autoencoder() is called
    if create_model or not os.path.exists(get_filepath(f"model/{model_name}.json")):
        autoencoder_gan = create_autoencoder_gan(x, x_train, x_test, y_train, y_test)
    else:
        autoencoder_gan = load_model(model_name)

    # check if model trained properly and get metrics for the table
    predicted_y, test_y = prediction_check(autoencoder_gan, x_test, y_test)
    f1, acc, average_fpr, average_tpr = get_metrics(predicted_y, test_y)

    if show_plot:
        plot_title = "AutoEncoder-GAN Confusion matrix"
        plot = show_confusion_matrix(predicted_y, test_y, plot_title)

    return acc, average_fpr, average_tpr, f1, plot



