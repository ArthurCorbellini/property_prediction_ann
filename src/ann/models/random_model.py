
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import random


class RandomModel:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = tf.constant(x_train)
        self.x_test = tf.constant(x_test)
        self.y_train = tf.constant(y_train)
        self.y_test = tf.constant(y_test)

        self.config = ""
        tf.random.set_seed(42)

        self.model = self._build_random_model()
        self._random_compile()
        self._random_fit()

        self.y_pred_train = tf.squeeze(self.model.predict(self.x_train))
        self.y_pred = tf.squeeze(self.model.predict(self.x_test))

    def _build_random_model(self):
        self._add_config("Architecture: ")
        model = tf.keras.models.Sequential()
        layers = random.randint(3, 5)
        self._add_config("  Hidden layers: " + str(layers))
        for i in range(layers):
            units = random.randint(10, 200)
            self._add_config("    Layer " + str(i) +
                             " > number of neurons: " + str(units))
            model.add(tf.keras.layers.Dense(
                units=units,
                activation="relu"),
            )
            # if (i + 1) < layers:
            #     dropout_rate = 0.2
            #     model.add(tf.keras.layers.Dropout(dropout_rate))
            #     self._add_config("      Dropout rate: " + str(dropout_rate))
        model.add(tf.keras.layers.Dense(units=1))

        return model

    def _random_compile(self):
        self._add_config("Compiler: ")
        if random.randint(0, 1) == 0:
            lr = random.choice([0.1, 0.01, 0.001, 0.0001, 0.00001])
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
            self._add_config("  Optimizer: Adam > lr " + str(lr))
        else:
            opt = tf.keras.optimizers.SGD()
            self._add_config("  Optimizer: SGD")

        self.model.compile(loss=tf.keras.losses.mae,
                           optimizer=opt,
                           metrics=['mae'])

    def _random_fit(self):
        self._add_config("Fit: ")
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=5)

        batch_size = random.choice([8, 16, 32, 64])
        self._add_config("  Batch size: " + str(batch_size))
        self.history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=batch_size,
            epochs=10000,
            callbacks=[callback],
            verbose=0,
        )
        self._add_config(
            "  Broke in " + str(self.history.epoch[-1] + 1) + " epochs")

    def compare_results(self):
        vertical_y_pred = self.y_pred.reshape(len(self.y_pred), 1)
        vertical_y_test = np.array(self.y_test).reshape(len(self.y_test), 1)

        np.set_printoptions(precision=2)
        # np.set_printoptions(precision=2, threshold=np.inf)
        return np.concatenate((vertical_y_pred, vertical_y_test), 1)

    def mae(self):
        return self._mae(y_true=self.y_test, y_pred=self.y_pred)

    def mae_train(self):
        return self._mae(y_true=self.y_train, y_pred=self.y_pred_train)

    def _mae(self, y_true, y_pred):
        return tf.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)

    def mse(self):
        return self._mse(y_true=self.y_test, y_pred=self.y_pred)

    def mse_train(self):
        return self._mse(y_true=self.y_train, y_pred=self.y_pred_train)

    def _mse(self, y_true, y_pred):
        return tf.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

    def plot_history(self):
        pd.DataFrame(self.history.history).plot()
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.show()

    def _add_config(self, text):
        self.config += "\n" + text
