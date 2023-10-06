import tensorflow as tf
from tensorflow import keras


def build_base_model(input_shape=(128, 128, 3)):
    conv_base = tf.keras.applications.efficientnet.EfficientNetB7(include_top=False,
                                                                  weights='imagenet',
                                                                  input_shape=input_shape)
    conv_base.trainable = False

    model_input = keras.Input(shape=input_shape, name="Input")
    x = tf.keras.applications.efficientnet.preprocess_input(model_input)
    x = conv_base(x)
    model_output = keras.layers.GlobalMaxPooling2D()(x)

    EFNB7 = keras.Model(inputs=model_input, outputs=model_output)

    return EFNB7
