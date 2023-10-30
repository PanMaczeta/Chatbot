import configparser
import os

import tensorflow as tf

from tools import load_data, preprocess, perplexity, plot_and_save_metric

# Preprocessing data phase
test_flag = True    # In case of lack of a data testing

if not test_flag:   #if you want to check compile
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
else:
    config = configparser.ConfigParser()
    config.read('config.ini')



# Architecture phase

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=config.getint('DEFAULT','VocabSize'), 
        output_dim=config.getint('DEFAULT','EmbeddingDim'), 
        input_length=config.getint('DEFAULT','MaxLength')
    ),
    tf.keras.layers.LSTM(units=512, return_sequences=True),
    tf.keras.layers.Conv1D(512, 5),
    tf.keras.layers.BatchNormalization(),              # BaN for performance
    tf.keras.layers.Activation(activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(512, return_sequences=True),  # Second LSTM layer
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(512, 5),
    tf.keras.layers.BatchNormalization(),              # BaN for performance
    tf.keras.layers.Activation(activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(512, return_sequences=True),  # Third LSTM layer
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(512, 5),
    tf.keras.layers.BatchNormalization(),              # BaN for performance
    tf.keras.layers.Activation(activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.2),                      # DO To prevent overfitting
    tf.keras.layers.LSTM(512, return_sequences=True),  # Fourth LSTM layer
    tf.keras.layers.BatchNormalization(),              # BaN for performance
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config.getint('DEFAULT','VocabSize'), activation='softmax'))
])


model.compile(loss='sparse_categorical_crossentropy', 
          optimizer='adam', 
          metrics=['accuracy', 'precision', 'recall', 'f1_score',perplexity])
model.summary()

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
