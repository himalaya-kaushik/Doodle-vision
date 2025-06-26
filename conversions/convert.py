import tensorflow as tf

# Load your native Keras model
model = tf.keras.models.load_model("best_hybrid_model_strokes_scaled.keras")

# Export it to SavedModel format (folder)
model.export("saved_model_format")  # This creates a directory


"""
# Python 3.9.12
pip install tf2onnx tensorflow onnxruntime
pip uninstall numpy -y
pip install numpy==1.26.4
python -m tf2onnx.convert --saved-model saved_model_format --output model.onnx

"""
