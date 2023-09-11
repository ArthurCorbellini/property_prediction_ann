
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


class ModelOne:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = tf.constant(x_train)
        self.x_test = tf.constant(x_test)
        self.y_train = tf.constant(y_train)
        self.y_test = tf.constant(y_test)

        tf.random.set_seed(42)

        self.model = self._build_model()
        self._compile()
        self._fit()

        self.y_pred = tf.squeeze(self.model.predict(self.x_test))

    def compare_results(self):
        vertical_y_pred = self.y_pred.reshape(len(self.y_pred), 1)
        vertical_y_test = np.array(self.y_test).reshape(len(self.y_test), 1)

        np.set_printoptions(precision=2)
        # np.set_printoptions(precision=2, threshold=np.inf)
        return np.concatenate((vertical_y_pred, vertical_y_test), 1)

    def _fit(self):
        # vai parar o treinamento do modelo caso ele não tenha mais melhora durante a quantidade de epochs estipulada em "patiance"
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=5)

        # batch_size: tamanho dos lotes processados antes do backpropagation.
        self.history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=16,
            epochs=500,
            callbacks=[callback],
            # verbose=0,
        )

    def _compile(self):
        self.model.compile(loss=tf.keras.losses.mae,  # "mae" is the same as -> loss = mean(abs(y_true - y_pred), axis =-1)
                           optimizer=tf.keras.optimizers.Adam(
                               learning_rate=0.001),  # learning_rate is the most important hyper parameter for improve prediction
                           metrics=['mae'])

    def _build_model(self):
        return tf.keras.models.Sequential([
            # model.add(tf.keras.Input(shape=(104,))),  # quantidade de entradas
            tf.keras.layers.Dense(units=100, activation="relu"),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=100, activation="relu"),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=100, activation="relu"),
            # Output Layer -> quanto a função de ativação:
            # - Classificação com APENAS duas categorias -> é recomendado usar a "Sigmoid Function";
            # - Classificação com MAIS DE duas categorias -> é recomendado usar a "Soft Max Function";
            # - Regressão -> é recomendado não usar funções de ativação.
            tf.keras.layers.Dense(units=1)
        ])

    def mae(self):
        """
        MAE: mean absolute error, "on average, how wrong is each of my model's predictions";
        """
        return tf.metrics.mean_absolute_error(y_true=self.y_test,
                                              y_pred=self.y_pred)

    def mse(self):
        """
        MSE: mean square error, "square the average errors". When larger errors are more significant than smaller errors;
        """
        return tf.metrics.mean_squared_error(y_true=self.y_test,
                                             y_pred=self.y_pred)

    def plot_history(self):
        pd.DataFrame(self.history.history).plot()
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.show()

    # def plot_model(self):
    #     """
    #     Precisa instalar o graphviz para funcionar
    #     """
    #     tf.keras.utils.plot_model(model=self.model, show_shapes=True)

    # def plot_predictions(self):
    #     """
    #     Plots training and test data, and compares predictions to ground truth labels.
    #     """
    #     plt.figure(figsize=(10, 7))
    #     plt.scatter(self.x_train[:, 0], self.y_train,
    #                 c="b", label="Training data")
    #     plt.scatter(self.x_test[:, 0], self.y_test,
    #                 c="g", label="Testing data")
    #     plt.scatter(self.x_test[:, 0], self.y_pred, c="r", label="Predictions")
    #     plt.legend()
    #     plt.show()
