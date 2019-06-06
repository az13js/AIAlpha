# -*- coding: UTF-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import output_file, figure, show


class NeuralNetwork:
    def __init__(self, input_shape, stock_or_return):
        self.input_shape = input_shape
        self.stock_or_return = stock_or_return

    def create_model(self, input_shape):
        # 这个必须是一个整数
        # input_shape = 20

        # 原始实现是用 Keras 框架做的，这里除了特殊指定之外，其余参数用 Keras 默认的参数配置
        model = tf.keras.models.Sequential()
        # 输入层
        model.add(tf.keras.layers.InputLayer(input_shape=(1, input_shape)))
        # 第一层 LSTM
        model.add(tf.keras.layers.LSTM(
            units=5, # 指定为5
            activation='tanh',
            recurrent_activation='hard_sigmoid',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            unit_forget_bias=True,
            kernel_regularizer=None,
            recurrent_regularizer=tf.keras.regularizers.l2(l=0), # 指定为0
            bias_regularizer=None,
            activity_regularizer=tf.keras.regularizers.l2(l=0.003), # 指定为0.003
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=0.2, # 指定为0.2
            recurrent_dropout=0.2, # 指定为0.2
            implementation=1,
            return_sequences=True, # 指定为False
            return_state=False,
            go_backwards=False,
            stateful=False,
            unroll=False
        ))
        # Dense 层
        model.add(tf.keras.layers.Dense(
            units=5, # 指定为5
            activation='sigmoid', # 指定为sigmoid
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=tf.keras.regularizers.l2(l=0.005), # 指定为0.005
            kernel_constraint=None,
            bias_constraint=None
        ))
        # 第二层 LSTM
        model.add(tf.keras.layers.LSTM(
            units=2, # 指定为2
            activation='tanh',
            recurrent_activation='hard_sigmoid',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            unit_forget_bias=True,
            kernel_regularizer=None,
            recurrent_regularizer=tf.keras.regularizers.l2(l=0.001), # 指定为0.001
            bias_regularizer=None,
            activity_regularizer=tf.keras.regularizers.l2(l=0.01), # 指定为0.003
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=0.2, # 论文指定为0.2
            recurrent_dropout=0.2, # 论文指定为0.2
            implementation=1,
            return_sequences=False,
            return_state=False,
            go_backwards=False,
            stateful=False,
            unroll=False
        ))
        # Dense 层
        model.add(tf.keras.layers.Dense(
            units=1, # 指定为1
            activation='sigmoid', # 指定为sigmoid
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=tf.keras.regularizers.l2(l=0.001), # 指定为0.005
            kernel_constraint=None,
            bias_constraint=None
        ))
        return model

    def make_train_model(self):
        model = self.create_model(self.input_shape)
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])

        # load data

        train = np.reshape(np.array(pd.read_csv("features/autoencoded_train_data.csv", index_col=0)),
                           (len(np.array(pd.read_csv("features/autoencoded_train_data.csv"))), 1, self.input_shape))
        train_y = np.array(pd.read_csv("features/autoencoded_train_y.csv", index_col=0))
        # train_stock = np.array(pd.read_csv("train_stock.csv"))

        # train model

        model.fit(train, train_y, epochs=2000)

        test_x = np.reshape(np.array(pd.read_csv("features/autoencoded_test_data.csv", index_col=0)),
                            (len(np.array(pd.read_csv("features/autoencoded_test_data.csv"))), 1, self.input_shape))
        test_y = np.array(pd.read_csv("features/autoencoded_test_y.csv", index_col=0))
        # test_stock = np.array(pd.read_csv("test_stock.csv"))

        stock_data_test = np.array(pd.read_csv("stock_data_test.csv", index_col=0))

        print(model.evaluate(test_x, test_y))
        prediction_data = []
        stock_data = []
        for i in range(len(test_y)):
            prediction = (model.predict(np.reshape(test_x[i], (1, 1, self.input_shape))))
            prediction_data.append(np.reshape(prediction, (1,)))
            std = np.std(prediction_data)
            if 0 == std:
                std = 0.0001
            prediction_corrected = (np.array(prediction_data) - np.mean(prediction_data)) * (1.0 / std)
            if 0 == i:
                stock_price = np.exp(np.reshape(prediction, (1,)))*stock_data_test[i]
            else:
                stock_price = np.exp(np.reshape(prediction, (1,)))*stock_data[i - 1]
            stock_data.append(stock_price[0])
        stock_data[:] = [i - (float(stock_data[0])-float(stock_data_test[0])) for i in stock_data]
        # stock_data = stock_data - stock_data[0]
        if self.stock_or_return:
            plt.plot(stock_data)
            plt.plot(stock_data_test)
            stock = pd.DataFrame(stock_data, index=None)
            stock_test = pd.DataFrame(stock_data_test, index=None)
            # print(stock_data)
            plt.show()
        else:
            # plt.plot(prediction_corrected)
            plt.plot(prediction_data)
            # print(prediction_data)
            plt.plot(test_y)
            plt.show()


if __name__ == "__main__":
    model = NeuralNetwork(20, True)
    model.make_train_model()
