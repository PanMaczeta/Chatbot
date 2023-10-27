import json
import string
import os

import numpy
import nltk
import tensorflow as tf
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def load_data(data_path):
    ''' Function that loads raw data from JSON file path.
        UTF-8 coding required.
    '''
    try:
        with open(data_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File '{data_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Decoding problem with JSON: {e} . Check if your file is coded UTF-8.")
        return None

# Preprocess data
def preprocess(data):
    '''Function used to preprocess raw data text.
       Tokenization, removing punctuation, 
       lowering cases and checks for stopwords in English.
    '''
    # Tokenize data
    tokens = nltk.word_tokenize(data)
    # Lowercase all words

    tokens = [word.lower() for word in tokens]

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens

def perplexity(y_true, y_pred):
    # Perplexity metric module for better LLM evaluation
    cross_entropy = tf.keras.backend.sparse_categorical_crossentropy(y_true, y_pred)
    perplexity = tf.exp(tf.reduce_mean(cross_entropy))

    return perplexity

# Metrics visualisation.
def plot_and_save_metric(history, metric_name, output_dir):
    ''' Function that creates visualisation of evaluation of model based on learning history.
        history - history form keras object,
        metric_name - name of metric form tensorflow keras documentation,
        output_dir - string type path for output files
    '''
    # Pobierz liczbę epok z historii treningu
    epochs = range(1, len(history.history[metric_name]) + 1)

    metric_values = history.history[metric_name]
    val_metric_name = 'val_' + metric_name
    val_metric_values = history.history[val_metric_name]

    # Tworzenie wykresu
    plt.figure()
    plt.plot(epochs, metric_values, 'b', label='Training ' + metric_name)
    plt.plot(epochs, val_metric_values, 'r', label='Validation ' + metric_name)
    plt.title('Training and Validation ' + metric_name)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()

    # Zapisywanie wykresu jako obraz
    output_path = os.path.join(output_dir, metric_name + '_plot.png')
    plt.savefig(output_path)
    plt.close()