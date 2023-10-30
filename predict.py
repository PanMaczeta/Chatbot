import numpy as np

from tools import load_data
import tensorflow as tf

model = tf.keras.models.load_model("output_model.h5")  # Podaj ścieżkę do zapisanego modelu

# Prepare input data
test_data = np.random.rand(10, 64)  # example data

# Wykonaj inferencję na danych testowych
predictions = model.predict(test_data)

# Show predictions
print("Predictions:")
print(predictions)