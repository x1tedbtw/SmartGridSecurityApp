import logging
import os

import numpy as np
import pandas as pd

import keras.backend.tensorflow_backend as tb

from backend.constants import LABELS
from backend.utils import get_filepath, load_model

logger = logging.getLogger(__name__)


def detect_anomalies(model_name, encode, columns):
    tb._SYMBOLIC_SCOPE.value = True
    model = load_model(model_name)

    path = get_filepath('temp_files')
    files = os.listdir(path)

    if files:
        file = get_filepath(f'temp_files/{files[0]}')
    else:
        file = get_filepath("data/testing_dataset.csv")

    test = pd.read_csv(file)
    logger.warning(test.values)
    test.fillna(0, inplace=True)

    # ADDED: Verify columns exist
    missing_columns = [col for col in columns if col not in test.columns]
    if missing_columns:
        result = f"Format of test data is inappropriate. Missing required columns: {missing_columns}. Expected columns: {columns}"
        is_successful = False
        return result, is_successful

    temp = test.values

    # Encode only the columns that exist (excluding 'label')
    try:
        for i in range(len(columns)):
            if i < len(encode):
                test[columns[i]] = pd.Series(encode[i].transform(test[columns[i]].astype(str)))
    except KeyError as e:
        result = f"Format of test data is inappropriate. Column {e} not found. Expected columns: {columns}"
        is_successful = False
        return result, is_successful

    test = test.values

    if model_name == "cnn_model":
        test = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))
    elif model_name == "lstm_model":
        test = np.reshape(test, (test.shape[0], test.shape[1], 1))
    elif model_name == "autoencoder_gan_model":
        x = test[:, 0:test.shape[1] - 1]

    try:
        predicted_data = model.predict(test)
        processed_prediction = np.argmax(predicted_data, axis=1)
        processed_prediction = np.array(LABELS)[processed_prediction]
        result = np.concatenate((temp, processed_prediction[:, np.newaxis]), axis=1)
        is_successful = True
    except ValueError as e:
        result = f"Format of test data is inappropriate. Expected 10 columns (without 'label'): {columns}. Error: {str(e)}"
        is_successful = False
    return result, is_successful
