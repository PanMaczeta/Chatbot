import numpy as np

from tools import load_data
import tensorflow as tf

model = tf.keras.models.load_model("output_model.h5")  # Input path to saved model

# Prepare input data
test_data = np.random.rand(10, 64)  # example data

# Do prediction
predictions = model.predict(test_data)

# Show predictions
print("Predictions:")
print(predictions)

# TODO: Add UI or communication for best live predictions