import configparser 
import tensorflow as tf

from tools import load_data, preprocess

data_path = ''
raw_data = load_data(data_path) # load raw data from path given

preprocessed_data = [preprocess(qa) for qa in raw_data.split('\n')]

#Params from config
training_size = len(preprocessed_data)
config = configparser.ConfigParser()
config.read('example.ini')

vectorizer = tf.keras.layers.TextVectorization(max_tokens=config["DEFAULT"]["VocabSize"], output_mode='int')
vectorizer.adapt(preprocessed_data)

sequences = vectorizer(preprocessed_data)

padded_sequences = tf.keras.utils.pad_sequences(sequences, 
              maxlen=config['DEFAULT']['MaxLength'], 
              padding=config['DEFAULT']['PaddingType'], 
              truncating=config['DEFAULT']['TruncType'])

training_data = padded_sequences[:training_size]
training_labels = padded_sequences[:training_size]

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
              config['DEFAULT']['VocabSize'], 
              config['DEFAULT']['EmbeddingDim'], 
              input_length=config['DEFAULT']['MaxLength']),
    tf.keras.layers.Conv1D(64, 5), #linear activation
    tf.keras.layers.BatchNormalization(), #BaN for performance
    tf.keras.layers.Activation(activation='relu'),   #Activation after BaN
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.BatchNormalization(), #BaN for performance
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(config['DEFAULT']['VocabSize'], activation='softmax')
    ])
model.compile(loss='sparse_categorical_crossentropy', 
          optimizer='adam', 
          metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
          monitor='val_loss',
          patience=5,
          restore_best_weights=True
          )

history = model.fit(
          training_data, training_labels, 
          epochs=100, 
          verbose=2,
          callbacks=[early_stopping]
          )