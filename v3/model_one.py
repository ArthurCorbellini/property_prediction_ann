"""
    Criar uma logica para randomizar arquiteturas da rede.
    - Assim, é possível ver qual arquitetura performou melhor.
"""
import numpy as np
import tensorflow as tf


class ModelOne:
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
    """
# 4763436.5000
# 5058019.0000

#   74677.1328 c/ 1000 epochs

# c/ 100 epochs:
# 1675329.1250
# 1790547.7500
# 1235155.6250 -> one hot nas self.categorical + self.discrete

#   80538.4766 c/ 1000 epochs
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.model = self._build_model()
        self._compile()
        self._fit()

    def predict(self):
        y_pred = self.model.predict(self.x_test)
        vertical_y_pred = y_pred.reshape(len(y_pred), 1)
        vertical_y_test = self.y_test.reshape(len(self.y_test), 1)

        np.set_printoptions(precision=2)
        return np.concatenate((vertical_y_pred, vertical_y_test), 1)

    def _fit(self):
        # batch_size: tamanho dos lotes processados antes do backpropagation.
        self.model.fit(self.x_train, self.y_train, batch_size=32, epochs=100)

    def _compile(self):
        # optimizer="adam": é um dos mais famosos, tanto para classificação quanto para regrassão;
        # loss="mean_squared_error": função de perda usada para os modelos de regressão;
        #  - Para regressão, também pode ser usada a função de perda "root_mean_square_error".
        self.model.compile(optimizer="adam", loss="mean_squared_error")

    def _build_model(self):
        model = tf.keras.models.Sequential()
        # Inserindo camadas ocultas no modelo
        model.add(tf.keras.layers.Dense(units=160, activation="relu"))
        model.add(tf.keras.layers.Dense(units=160, activation="relu"))
        model.add(tf.keras.layers.Dense(units=160, activation="relu"))
        # Output Layer -> atenção quanto a função de ativação:
        # - Classificação com APENAS duas categorias -> é recomendado usar a "Sigmoid Function";
        # - Classificação com MAIS DE duas categorias -> é recomendado usar a "Soft Max Function";
        # - Regressão -> é recomendado não usar funções de ativação.
        model.add(tf.keras.layers.Dense(units=1))

        return model
