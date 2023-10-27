import configparser
import os

import tensorflow as tf

from tools import load_data, preprocess, perplexity, plot_and_save_metric

# Preprocessing data phase


data_path = ''
raw_data = load_data(data_path) # load raw data from path given (TODO: SQL integration)

preprocessed_data = [preprocess(qa) for qa in raw_data.split('\n')]

# Params from config
training_size = len(preprocessed_data)
config = configparser.ConfigParser()
config.read('config.ini')

vectorizer = tf.keras.layers.TextVectorization(max_tokens=config["DEFAULT"]["VocabSize"], output_mode='int')
vectorizer.adapt(preprocessed_data)

sequences = vectorizer(preprocessed_data)

padded_sequences = tf.keras.utils.pad_sequences(sequences, 
              maxlen=config['DEFAULT']['MaxLength'], 
              padding=config['DEFAULT']['PaddingType'], 
              truncating=config['DEFAULT']['TruncType'])

training_data = padded_sequences[:training_size // 2]       # Typical QA dataset
training_labels = padded_sequences[training_size // 2:]

# Architecture phase

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
              config['DEFAULT']['VocabSize'], 
              config['DEFAULT']['EmbeddingDim'], 
              input_length=config['DEFAULT']['MaxLength']),
    tf.keras.layers.Conv1D(64, 5),                          # linear activation
    tf.keras.layers.BatchNormalization(),                   # BaN for performance
    tf.keras.layers.Activation(activation='relu'),          # Activation after BaN
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.BatchNormalization(),                   # BaN for performance
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(config['DEFAULT']['VocabSize'], activation='softmax')
    ])


model.compile(loss='sparse_categorical_crossentropy', 
          optimizer='adam', 
          metrics=['accuracy', 'precision', 'recall', 'f1_score',perplexity]) 

early_stopping = tf.keras.callbacks.EarlyStopping(
          monitor='val_loss',
          patience=5,
          restore_best_weights=True
          )

# Learning phase

history = model.fit(                        # Using variable history to evalute model in the future
          training_data, training_labels, 
          epochs=10000,                     # Default high value controlled by early stopping
          verbose=2,                        # Using whole log information
          validation_split=0.15,            # 15% of validation data because of using potentially small dataset
          callbacks=[early_stopping]
          )


# Evaluation phase

output_directory = 'output_plots'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

default_metrics = ['accuracy', 'precision', 'recall', 'f1_score']

for metric_name in default_metrics:
    plot_and_save_metric(history, metric_name, output_directory)    # Will create plots for all metrics

model_output_directory = 'output_model'
if not os.path.exists(model_output_directory):
    os.makedirs(model_output_directory)

model.save("output_model/output_model.h5") # saving model in h5 format TODO: format is debatable
