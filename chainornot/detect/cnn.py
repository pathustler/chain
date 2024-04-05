import tensorflow as tf

def create_cnn_model():
    cnn = tf.keras.models.Sequential()
    return cnn

model = create_cnn_model()
model.load_weights('model.h5')