import numpy as np
import pandas as pd
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from backend.utils import get_filepath, save_to_file


def load_data(show_plot=False):
    csv_path = get_filepath('data/training_dataset.csv')
    training_dataset = pd.read_csv(csv_path)

    # fill the gaps
    training_dataset.fillna(0, inplace=True)
    values = training_dataset.values

    columns = training_dataset.columns  # column names
    labels = training_dataset.groupby(
        'label').size()  # count of rows with specific labels (MITM_UNALTERED, NORMAL, RESPONSE_ATTACK)

    # encode the dataset (simplify it using sklearn LabelEncoder)
    encoded_dataset = []
    for i in range(len(columns)):
        label_encoder = LabelEncoder()
        training_dataset[columns[i]] = pd.Series(label_encoder.fit_transform(training_dataset[columns[i]].astype(str)))
        # fit_transform() converts the categorical values into numerical labels
        encoded_dataset.append(label_encoder)
    # Label Encoding is a technique that is used to convert categorical columns into numerical ones so that they can be
    # fitted by machine learning models which only take numerical data. It is an important pre-processing step in a
    # machine-learning project.

    # prepare and shuffle data
    training_dataset = training_dataset.values  # only values without labels
    x = training_dataset[:, 0:training_dataset.shape[1] - 1]  # x = input dataset
    y = training_dataset[:, training_dataset.shape[1] - 1]  # y = output (values of data column)
    indexes = np.arange(x.shape[0])  # get indexes (from 0 to 3089)
    np.random.shuffle(indexes)  # shuffle indexes
    x = x[indexes]  # shuffle input dataset
    y = y[indexes]  # shuffle output dataset

    # convert class vector (integers) to binary class matrix
    y = to_categorical(y)  # (0 = [1 0 0], 1 = [0 1 0] ...)

    # split the dataset into training and testing sets using sklearn
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  # 20% (0.2) of data used for test

    processed_training_data = {
        'x': x,
        'y': y,
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }
    dataset = {
        'columns': columns,
        'values': values
    }
    # Saves files to SmartGridSecurity/data/
    save_to_file('data/processed_training_data.pckl', processed_training_data)
    save_to_file('data/encoded_dataset.pckl', encoded_dataset)
    save_to_file('data/dataset.pckl', dataset)

    if show_plot:
        labels.plot(kind="bar")
        plt.title("Various Attacks found in Modbus/TCP dataset")
        plt.show()
