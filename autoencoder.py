import tensorflow as tf
import pandas as pd
import numpy as np


class AutoEncoder:
    def __init__(self, encoding_dim):
        self.encoding_dim = encoding_dim

    def build(self, input_shape, shape1, shape2, shape3, shape4, shape5, shape6):
        input = tf.keras.layers.Input(shape=(1, input_shape))
        dense1 = tf.keras.layers.Dense(
            units=shape1,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=tf.keras.regularizers.l2(l=0),
            kernel_constraint=None,
            bias_constraint=None
        )(input)
        dense2 = tf.keras.layers.Dense(
            units=shape2,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=tf.keras.regularizers.l2(l=0),
            kernel_constraint=None,
            bias_constraint=None
        )(dense1)
        dense3 = tf.keras.layers.Dense(
            units=shape3,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=tf.keras.regularizers.l2(l=0),
            kernel_constraint=None,
            bias_constraint=None
        )(dense2)
        dense4 = tf.keras.layers.Dense(
            units=shape4,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=tf.keras.regularizers.l2(l=0),
            kernel_constraint=None,
            bias_constraint=None
        )(dense3)
        dense5 = tf.keras.layers.Dense(
            units=shape5,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=tf.keras.regularizers.l2(l=0),
            kernel_constraint=None,
            bias_constraint=None
        )(dense4)
        dense6 = tf.keras.layers.Dense(
            units=shape6,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=tf.keras.regularizers.l2(l=0),
            kernel_constraint=None,
            bias_constraint=None
        )(dense5)

        encode_decode = tf.keras.Model(inputs=input, outputs=dense6)
        encode = tf.keras.Model(inputs=input, outputs=dense3)
        return (encode_decode, encode)

    def build_train_model(self, input_shape, encoded1_shape, encoded2_shape, decoded1_shape, decoded2_shape):

        (autoencoder, encoder) = self.build(input_shape, encoded1_shape, encoded2_shape, self.encoding_dim, decoded1_shape, decoded2_shape, input_shape)

        # Now train the model using data we already preprocessed
        autoencoder.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])

        train = pd.read_csv("preprocessing/rbm_train.csv", index_col=0)
        ntrain = np.array(train)
        train_data = np.reshape(ntrain, (len(ntrain), 1, input_shape))

        # print(train_data)
        # autoencoder.summary()
        autoencoder.fit(train_data, train_data, epochs=1000)

        encoder.save("models/encoder.h5")

        test = pd.read_csv("preprocessing/rbm_test.csv", index_col=0)
        ntest = np.array(test)
        test_data = np.reshape(ntest, (len(ntest), 1, 55))

        print(autoencoder.evaluate(test_data, test_data))
        # pred = np.reshape(ntest[1], (1, 1, 75))
        # print(encoder.predict(pred))

        log_train = pd.read_csv("preprocessing/log_train.csv", index_col=0)
        coded_train = []
        for i in range(len(log_train)):
            data = np.array(log_train.iloc[i, :])
            values = np.reshape(data, (1, 1, 55))
            coded = encoder.predict(values)
            shaped = np.reshape(coded, (20,))
            coded_train.append(shaped)

        train_coded = pd.DataFrame(coded_train)
        train_coded.to_csv("features/autoencoded_data.csv")


if __name__ == "__main__":
    autoencoder = AutoEncoder(20)
    autoencoder.build_train_model(55, 40, 30, 30, 40)
