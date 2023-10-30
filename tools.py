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

# Alternative architecture
# import tensorflow as tf
# 
# def transformer_model(vocab_size, d_model, n_heads, n_encoder_layers, n_decoder_layers, dff, input_max_len, target_max_len, rate=0.1):
#     inputs = tf.keras.layers.Input(shape=(input_max_len,))
#     dec_inputs = tf.keras.layers.Input(shape=(target_max_len,))
#     
#     # Embedding and positional encoding for the encoder
#     enc_padding_mask = tf.keras.layers.Lambda(create_padding_mask, input_shape=(1, None))(inputs)
#     embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
#     embeddings *= tf.keras.layers.Lambda(tf.sqrt, arguments={'x': d_model})(embeddings)
#     embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
#     
#     x = embeddings
#     for _ in range(n_encoder_layers):
#         x = encoder_layer(d_model, n_heads, dff, rate)([x, enc_padding_mask])
#     
#     # Embedding and positional encoding for the decoder
#     look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask, input_shape=(1, None), dtype=tf.float32)(dec_inputs)
#     dec_padding_mask = tf.keras.layers.Lambda(create_padding_mask, input_shape=(1, None))(inputs)
#     embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(dec_inputs)
#     embeddings *= tf.keras.layers.Lambda(tf.sqrt, arguments={'x': d_model})(embeddings)
#     embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
#     
#     x = embeddings
#     for _ in range(n_decoder_layers):
#         x = decoder_layer(d_model, n_heads, dff, rate)([x, enc_output, look_ahead_mask, dec_padding_mask])
#     
#     outputs = tf.keras.layers.Dense(vocab_size)(x)
#     
#     return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs)
# 