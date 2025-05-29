import os
import numpy as np
from sklearn.model_selection import train_test_split # splits data into training and test sets
from keras.layers import MaxPooling2D
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential # model type where layers are stacked one after the other.
import keras.backend.tensorflow_backend as tb

from backend.utils import prediction_check, get_metrics, get_filepath, load_model, show_confusion_matrix, \
    save_model_components_to_files

def create_cnn_model(x_reshaped, x_train, x_test, y_train, y_test):
    # create sequential model (linear stack of layers)
    cnn_model = Sequential()
    # layers forming a "sequence" of layers

    # Applies 32 independent 1×1 filters across the “height” dimension (features).
    # Each filter learns a per‑feature transformation, with ReLU non‑linearity.
    cnn_model.add(
        Convolution2D(32, 1, 1, input_shape=(x_reshaped.shape[1], x_reshaped.shape[2], x_reshaped.shape[3]), activation='relu'))
    # add max pooling layer (the maximum value in the neighborhood of size 1x1 around each pixel is taken)
    cnn_model.add(MaxPooling2D(pool_size=(1, 1)))   # pooling window size 1x1, doesn't reduce the size

    # add other convolutional and max pooling layers
    cnn_model.add(Convolution2D(32, 1, 1, activation='relu'))
    cnn_model.add(MaxPooling2D(pool_size=(1, 1)))

    # add flatten layer to convert multi-dimensional input into a one-dimensional array
    cnn_model.add(Flatten())
    # In CNN, output of convolutional and pooling layers is a 3D tensor with dims (height, width, channels).
    # To pass data to fully connected layer for classification, it needs to be converted into a 1D vector.

    # add 2 fully connected layers, 1st - 256 neurons, 2nd - 3 neurons and compile the model
    cnn_model.add(Dense(output_dim=256, activation='relu'))

    # Final classification layer with one neuron per class.
    # Softmax turns raw scores into probabilities that sum to 1.
    cnn_model.add(Dense(output_dim=y_train.shape[1], activation='softmax'))
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
    # adam optimizer - dynamically adjusts the learning rate (internal weights) during training based on loss func
    # loss crossentropy - how far off the model's predictions are from the true labels
    # accuracy - measures the percentage of correctly classified samples out of the total number of samples between 0 and 1

    # train the model
    training_results = cnn_model.fit(x_train, y_train, batch_size=16, epochs=30, shuffle=True, validation_data=(x_test, y_test),
                         verbose=0)
    # model will update its weights after every batch of 16 samples
    # model will be trained for 30 epochs
    # verbose=2 means it will display a progress bar for each epoch

    # save weights, model and history to files
    save_model_components_to_files(cnn_model, training_results, "cnn_model")

    return cnn_model

def train_cnn(x, y, create_model=False, show_plot=False):
    # change TensorFlow setting
    tb._SYMBOLIC_SCOPE.value = True  # operations should be added to the TensorFlow symbolic graph

    # # Reshape original feature vector into “fake image” format (input)
    x_reshaped = np.reshape(x, (x.shape[0], x.shape[1], 1, 1))
    # x.shape[0] (samples number) - same, x.shape[1] (features number) - same, data height - 1, data width - 1
    # each sample now has dimensions (num_features, 1, 1)
    # used when preparing data for CNNs where input data should have shape (num_samples, height, width, channels)
    # the height and width are set to 1 (data is one-dimensional)

    # prepare datasets for training and testing using sklearn (input - reshaped x, 0.2 means 20% of data used for test)
    x_train, x_test, y_train, y_test = train_test_split(x_reshaped, y, test_size=0.2)

    model_name = "cnn_model"
    # check if the model is already trained
    if create_model or not os.path.exists(get_filepath(f"model/{model_name}.json")):
        cnn_model = create_cnn_model(x_reshaped, x_train, x_test, y_train, y_test)
    else:
        cnn_model = load_model(model_name)

    # check the correctness
    predicted_y, test_y = prediction_check(cnn_model, x_test, y_test)
    f1, acc, average_fpr, average_tpr = get_metrics(predicted_y, test_y)

    if show_plot:
        plot_title = "CNN Confusion matrix"
        plot = show_confusion_matrix(predicted_y, test_y, plot_title)

    return acc, average_fpr, average_tpr, f1, plot





# additional notes:
# input shape (10, 1, 1):
# [[[a1]]
#  [[a2]]
#  [[a3]]
#  [[a4]]
#  [[a5]]
#  [[a6]]
#  [[a7]]
#  [[a8]]
#  [[a9]]
#  [[a10]]]
