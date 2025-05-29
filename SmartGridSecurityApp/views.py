import logging
from django.shortcuts import render

from SmartGridSecurityApp.forms import DocumentForm
from SmartGridSecurityApp.models import Document
from backend.autoencoder_gan import train_autoencoder_gan
from backend.cnn import train_cnn
from backend.constants import METRICS
from backend.dataset import load_data
from backend.detection import detect_anomalies
from backend.lstm import train_lstm
from backend.utils import read_from_file, clear_folder

logger = logging.getLogger(__name__)


def home(request):
    load_data()
    return render(request, 'home.html')


def dataset(request):
    data = read_from_file('data/dataset.pckl')
    context = {'columns': data['columns'], 'data': data['values']}
    return render(request, 'dataset.html', context)


def training(request):

    processed_data = read_from_file('data/processed_training_data.pckl')
    #Loads preprocessed training data
    x, y = processed_data['x'], processed_data['y']
    # Extracting feature and label arrays, and training/testing splits
    x_train, x_test, y_train, y_test = processed_data['x_train'], processed_data['x_test'], processed_data['y_train'], processed_data['y_test']

    is_data = False
    training_metrics = []
    plot = ''
    if request.method == "POST":
        is_data = True
        current_algorithm = request.POST.get('algorithm_selected')
        accuracy, f1_score, tpr, fpr = 0, 0, 0, 0
        if current_algorithm == 'AutoEncoder-GAN':
            accuracy, f1_score, tpr, fpr, plot = train_autoencoder_gan(x, x_train, x_test, y_train, y_test, show_plot=True)
        elif current_algorithm == 'CNN':
            accuracy, f1_score, tpr, fpr, plot = train_cnn(x, y, show_plot=True)
        elif current_algorithm == 'LSTM':
            accuracy, f1_score, tpr, fpr, plot = train_lstm(x, y, show_plot=True)

        #Stores results in a list
        training_metrics = [current_algorithm, accuracy, f1_score, tpr, fpr]
        logger.warning(training_metrics)

    context = {"columns": METRICS, "data": [training_metrics], "is_data": is_data, "plot": plot}
    return render(request, 'training.html', context)


def detection(request):
    dataset = read_from_file('data/dataset.pckl')
    encoded_dataset = read_from_file('data/encoded_dataset.pckl')
    is_selected = False
    is_data = False
    is_successful = False
    analyzed_data = []
    if request.method == "POST":
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            selected_file = request.FILES['file']
            file_model = Document(file=selected_file) # creates a new unsaved instance of the Document model, with the file attached
            clear_folder('temp_files')
            file_model.save()
            is_selected = True
        if 'algorithm_selected' in request.POST:
            is_data = True
            is_selected = True
            current_algorithm = request.POST.get(
                'algorithm_selected')
            analyzed_data, is_successful = detect_anomalies(current_algorithm, encoded_dataset, dataset['columns'])

    context = {"is_selected": is_selected, "is_data": is_data, "is_successful": is_successful,
               "columns": dataset['columns'], "data": analyzed_data}
    return render(request, 'detection.html', context)
