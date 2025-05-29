import logging
import os
import pickle
from io import StringIO

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import sklearn.metrics as metrics
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from backend.constants import LABELS


logger = logging.getLogger(__name__)


def load_model(model_name):
    with open(get_filepath(f"model/{model_name}.json"), "r") as json_file:
        json_model = json_file.read()  # read file content
        training_model = model_from_json(json_model)  # load model with keras
    json_file.close()
    training_model.load_weights(get_filepath(f"model/{model_name}_weights.h5"))
    training_model._make_predict_function()
    return training_model


def prediction_check(model, x_test, y_test):
    # checks the model correctness
    predicted_data = model.predict(x_test)
    processed_prediction = np.argmax(predicted_data,
                                     axis=1)  # find the index of the maximum value (predicted output) with numpy
    processed_test_data = np.argmax(y_test, axis=1)  # find the index of the maximum value (desired output) with numpy
    # In classification tasks, the output of a model consists of predicted probabilities for each class
    # The class with the highest probability is considered the predicted class label for that instance
    # np.argmax() is used to find the index (or class label) corresponding to the maximum predicted value

    return processed_prediction, processed_test_data


def get_metrics(predicted_y, test_y):
    f1 = f1_score(test_y, predicted_y, average='macro')  # get f1 score using sklearn
    acc = accuracy_score(test_y, predicted_y)  # get accuracy score using sklearn
    all_fpr, all_tpr, threshold = metrics.roc_curve(test_y, predicted_y, pos_label=1)  # get metrics using sklearn
    average_fpr = (np.sum(all_fpr) / len(all_fpr))  # calculate average false positive rate
    average_tpr = (np.sum(all_tpr) / len(all_fpr))  # calculate average true positive rate
    return f1, acc, average_fpr, average_tpr


def get_filepath(filename):
    parent_path = "C:\\Users\\nefed\\Vitalii\\SmartGridSecurity"
    return os.path.join(parent_path, filename)


def save_to_file(filename, data):
    with open(get_filepath(filename), 'wb') as f:
        pickle.dump(data, f)


def read_from_file(filename):
    data = {}
    with open(get_filepath(filename), 'rb') as f:
        data = pickle.load(f)
    f.close()
    return data


def save_model_components_to_files(model, training_history, model_name):
    model.save_weights(get_filepath(f'model/{model_name}_weights.h5'))
    json_model = model.to_json()
    with open(get_filepath(f"model/{model_name}.json"), "w") as json_file:
        json_file.write(json_model)
    json_file.close()
    f = open(get_filepath(f'model/{model_name}.pckl'), 'wb')
    pickle.dump(training_history.history, f)
    f.close()


def show_confusion_matrix(predicted_y, test_y, title):

    conf_matrix = confusion_matrix(test_y, predicted_y)
    fig = plt.figure(figsize=(6, 6), facecolor='#32325d')
    ax = sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, cmap="magma", fmt="g", linecolor='#32325d', linewidth=2)
    ax.set_ylim([0, len(LABELS)])
    ax.set_title(title, color='white')
    ax.set_ylabel('True class', color='white')
    ax.set_xlabel('Predicted class', color='white')
    ax.tick_params(axis='x', colors='#adb5bd')
    ax.tick_params(axis='y', colors='#adb5bd')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors='#adb5bd')
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg', bbox_inches='tight', transparent=True)
    imgdata.seek(0)
    data = imgdata.getvalue()
    return data


def clear_folder(folder):
    path = get_filepath(folder)
    try:
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except OSError:
        logger.warning(f'Error occurred while deleting files.')
