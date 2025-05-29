from backend.detection import detect_anomalies
from backend.dataset import load_data
from backend.cnn import train_cnn
from backend.autoencoder_gan import train_autoencoder_gan
from backend.lstm import train_lstm
from backend.utils import read_from_file

# process dataset and prepare training data
load_data()
# data = read_from_file('data/processed_training_data.pckl')
# print(data)
# data = read_from_file('data/encoded_dataset.pckl')
# print(data)
data = read_from_file('data/dataset.pckl')
print(data)

# # train the model with GAN and get metrics
# train_autoencoder_gan(x, x_train, x_test, y_train, y_test, algorithm_titles, accuracy, f1, tpr, fpr)
# print(algorithm_titles, accuracy, f1, tpr, fpr)
#
# # train the model with CNN and get metrics
# train_cnn(x, y, algorithm_titles, accuracy, f1, tpr, fpr)
# print(algorithm_titles, accuracy, f1, tpr, fpr)
#
# # train the model with CNN and get metrics
# train_lstm(x, y, algorithm_titles, accuracy, f1, tpr, fpr)
# print(algorithm_titles, accuracy, f1, tpr, fpr)
#
# # predict with cnn
# model_name = "cnn_model"
# result = detect_anomalies(model_name, encode, columns)
# print(result)
#

