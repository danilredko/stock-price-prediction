import tensorflow as tf
from tensorflow.keras import layers


class ModelBuilder:
    def __init__(self, input_shape, model_type, dropout_rate):
        self.model_func = getattr(ModelBuilder, model_type)
        self.model = self.model_func(input_shape, dropout_rate)

    @staticmethod
    def fc_model(input_shape, dropout_rate):
        model = tf.keras.Sequential(
            [
                layers.Dense(500, input_shape=input_shape),
                layers.Dropout(dropout_rate),
                layers.BatchNormalization(),
                layers.Dense(40),
                layers.Dropout(dropout_rate),
                layers.BatchNormalization(),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        return model
