import os
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from keras.models import Sequential
import keras.backend.tensorflow_backend as tb
from backend.utils import get_filepath, prediction_check, get_metrics, show_confusion_matrix, load_model, \
    save_model_components_to_files

def create_lstm_model(x_reshaped, x_train, x_test, y_train, y_test):
    # create sequential model (linear stack of layers) using Keras
    lstm_model = Sequential()

    # add LSTM layer with 100 filters and input dimension (10, 1)
    lstm_model.add(keras.layers.LSTM(100, input_shape=(x_reshaped.shape[1], x_reshaped.shape[2])))

    # add dropout layer to introduce noise into the training process
    lstm_model.add(Dropout(0.5))  # half of input units will be set to zero during each iteration
    # Dropout is a regularization technique used to prevent over fitting in neural networks

    # add 2 fully connected layers with 100 and 3 neurons
    lstm_model.add(Dense(100, activation='relu'))
    lstm_model.add(Dense(y_train.shape[1], activation='softmax'))

    # compile the model
    lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # adam optimizer - dynamically adjusts the learning rate during training
    # crossentropy - loss function when there are two or more label classes
    # measures performance of a classification model whose output is a probability value between 0 and 1
    # accuracy - measures the proportion of correctly classified samples out of the total number of samples

    # train the model and save history
    training_results = lstm_model.fit(x_train, y_train, batch_size=16, epochs=30, shuffle=True, verbose=0,
                          validation_data=(x_test, y_test))
    # model will update its weights after every batch of 16 samples
    # model will be trained for 30 epochs
    # verbose=2 means it will display a progress bar for each epoch

    # save weights, model and history to files
    save_model_components_to_files(lstm_model, training_results, "lstm_model")

    return lstm_model


def train_lstm(x, y, create_model=False, show_plot=False):
    tb._SYMBOLIC_SCOPE.value = True

    # reshape input dataset with NumPy
    x_reshaped = np.reshape(x, (x.shape[0], x.shape[1], 1))

    # prepare datasets for training and testing using sklearn (input - reshaped x, 0.2 means 20% of data used for test)
    x_train, x_test, y_train, y_test = train_test_split(x_reshaped, y, test_size=0.2)

    model_name = "lstm_model"
    # check if the model is already trained
    if create_model or not os.path.exists(get_filepath(f"model/{model_name}.json")):
        lstm_model = create_lstm_model(x_reshaped, x_train, x_test, y_train, y_test)
    else:
        lstm_model = load_model(model_name)

    # check the correctness
    predicted_y, test_y = prediction_check(lstm_model, x_test, y_test)
    f1, acc, average_fpr, average_tpr = get_metrics(predicted_y, test_y)

    if show_plot:
        plot_title = "LSTM Confusion matrix"
        plot = show_confusion_matrix(predicted_y, test_y, plot_title)

    return acc, average_fpr, average_tpr, f1, plot



