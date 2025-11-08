import tensorflow as tf
model = tf.keras.models.load_model("converted_xception.h5", compile=False)
model.save("converted_xception_savedmodel", save_format="tf")
    