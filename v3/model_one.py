"""
    Criar uma logica para randomizar arquiteturas da rede.
    - Assim, é possível ver qual arquitetura performou melhor.
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"""
epochs: 100;
batch-size: 1;
units: 16 x 2
Epoch 57/100 - loss: 4968255.5000
Epoch 100/100 - loss: 4930904.5000
-----------------
epochs: 100;
batch-size: 2;
units: 256 x 3
Epoch 43/100 - loss: 5087550.0000
Epoch 100/100 - loss: 5073770.5000
-----------------
Após a primeira atualização do data_prep
-----------------
epochs: 100;
batch-size: 8;
units: 160 x 3
Epoch 50/100 - loss: 4118155.0000
Epoch 100/100 - loss: 3427015.2500
-----------------
Epoch 9998/10000
132/132 [==============================] - 0s 1ms/step - loss: 93.6174 - mae: 93.6174
Epoch 9999/10000
132/132 [==============================] - 0s 1ms/step - loss: 92.8698 - mae: 92.8698
Epoch 10000/10000
132/132 [==============================] - 0s 1ms/step - loss: 91.5106 - mae: 91.5106
34/34 [==============================] - 0s 649us/step
[[1417.33 1400.  ]
 [2484.53 2300.  ]
 [ 601.14 1000.  ]
 ...
 [1045.86 1300.  ]
 [2021.16 2000.  ]
 [1320.44 2300.  ]]

"""


class ModelOne:
    # 4763436.5000
    # 5058019.0000

    #   74677.1328 c/ 1000 epochs

    # c/ 100 epochs:
    # 1675329.1250
    # 1790547.7500
    # 1235155.6250 -> one hot nas self.categorical + self.discrete

    #   80538.4766 c/ 1000 epochs
    #   54164.6211
    #  190031.1094 s/outliers c/ 100 epochs
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = tf.constant(x_train)
        self.x_test = tf.constant(x_test)
        self.y_train = tf.constant(y_train)
        self.y_test = tf.constant(y_test)

        self.model = self._build_model()
        self._compile()
        self._fit()

        self.y_pred = tf.squeeze(self.model.predict(self.x_test))

    def results(self):
        vertical_y_pred = self.y_pred.reshape(len(self.y_pred), 1)
        vertical_y_test = np.array(self.y_test).reshape(len(self.y_test), 1)

        np.set_printoptions(precision=2)
        # np.set_printoptions(precision=2, threshold=np.inf)
        return np.concatenate((vertical_y_pred, vertical_y_test), 1)

    def _fit(self):
        # batch_size: tamanho dos lotes processados antes do backpropagation.
        self.model.fit(self.x_train, self.y_train, batch_size=32, epochs=10)

    def _compile(self):
        # optimizer="adam": é um dos mais famosos, tanto para classificação quanto para regrassão;
        # loss="mean_squared_error": função de perda usada para os modelos de regressão;
        #  - Para regressão, também pode ser usada a função de perda "root_mean_square_error".

        # self.model.compile(optimizer="adam", loss="mean_squared_error")
        self.model.compile(loss=tf.keras.losses.mae,  # "mae" is the same as -> loss = mean(abs(y_true - y_pred), axis =-1)
                           optimizer=tf.keras.optimizers.Adam(
                               learning_rate=0.01),  # learning_rate is the most important hyper parameter for improve prediction
                           metrics=['mae'])

    def _build_model(self):
        model = tf.keras.models.Sequential()
        # Inserindo camadas ocultas no modelo
        model.add(tf.keras.Input(shape=(107,)))  # quantidade de entradas
        model.add(tf.keras.layers.Dense(units=100, activation="relu"))
        model.add(tf.keras.layers.Dense(units=100, activation="relu"))
        model.add(tf.keras.layers.Dense(units=100, activation="relu"))
        model.add(tf.keras.layers.Dense(units=100, activation="relu"))
        # Output Layer -> atenção quanto a função de ativação:
        # - Classificação com APENAS duas categorias -> é recomendado usar a "Sigmoid Function";
        # - Classificação com MAIS DE duas categorias -> é recomendado usar a "Soft Max Function";
        # - Regressão -> é recomendado não usar funções de ativação.
        model.add(tf.keras.layers.Dense(units=1))

        return model

    def plot_model(self):
        """
        Precisa instalar o graphviz para funcionar
        """
        tf.keras.utils.plot_model(model=self.model, show_shapes=True)

    def plot_predictions(self):
        """
        Plots training and test data, and compares predictions to ground truth labels.
        """
        plt.figure(figsize=(10, 7))
        plt.scatter(self.x_train[:, 0], self.y_train,
                    c="b", label="Training data")
        plt.scatter(self.x_test[:, 0], self.y_test,
                    c="g", label="Testing data")
        plt.scatter(self.x_test[:, 0], self.y_pred, c="r", label="Predictions")
        plt.legend()
        plt.show()

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
