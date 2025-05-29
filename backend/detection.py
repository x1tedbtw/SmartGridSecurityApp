import logging
import os

import numpy as np
import pandas as pd

import keras.backend.tensorflow_backend as tb

from backend.constants import LABELS
from backend.utils import get_filepath, load_model

logger = logging.getLogger(__name__)


def detect_anomalies(model_name, encode, columns):
    tb._SYMBOLIC_SCOPE.value = True  # operations should be added to the TensorFlow symbolic graph
    # load model
    model = load_model(model_name)

    path = get_filepath('temp_files')
    files = os.listdir(path)

    # read test data and fill NA values with 0
    if files:
        file = get_filepath(f'temp_files/{files[0]}') # user-uploaded file
    else:
        file = get_filepath("data/testing_dataset.csv") # fallback file

    test = pd.read_csv(file)
    logger.warning(test.values)
    test.fillna(0, inplace=True)

    # extract the values of the DataFrame into a NumPy array
    temp = test.values

    # encode values by transforming the categorical values to the corresponding LabelEncoder object from encode list
    for i in range(len(encode) - 1):
        test[columns[i]] = pd.Series(encode[i].transform(test[columns[i]].astype(str)))  # using pandas

    # convert DataFrame to NumPy Array
    test = test.values

    if model_name == "cnn_model":
        # reshape input for CNM to dimension (batch_size, height, width, channels)
        test = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))

    elif model_name == "lstm_model":
        # reshape input dataset with NumPy
        test = np.reshape(test, (test.shape[0], test.shape[1], 1))

    elif model_name == "autoencoder_gan_model":
        x = test[:, 0:test.shape[1] - 1]

    try:
        predicted_data = model.predict(test) #gets model output
        processed_prediction = np.argmax(predicted_data, axis=1) # get the predicted label index
        processed_prediction = np.array(LABELS)[processed_prediction] # fet label
        result = np.concatenate((temp, processed_prediction[:, np.newaxis]), axis=1)
        #Combine original data (temp) with predictions as the last column
        is_successful = True
    except ValueError:
        result = "Format of test data is unappropriated. The csv file should have 10 columns"
        is_successful = False
    return result, is_successful
