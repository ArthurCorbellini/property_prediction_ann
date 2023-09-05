from data_prep import DataPrep
from model_one import ModelOne

# "sale" ou "rent"
dp = DataPrep("rent")
# print(dp.data_frame)
# print(dp.x_train)

m_one = ModelOne(dp.x_train, dp.x_test, dp.y_train, dp.y_test)

# print(m_one.results())
m_one.model.summary()
m_one.model.evaluate(m_one.x_test, m_one.y_test)
# For regression, whe have two main metrics:
# - MAE: mean absolute error, "on average, how wrong is each of my model's predictions";
# - MSE: mean square error, "square the average errors". When larger errors are more significant than smaller errors;
# - Huber: combination of MSE and MAE. Less sensitive to outliers than MSE.
print("-- MAE")
print(m_one.mae())
print("-- MSE")
print(m_one.mse())

m_one.plot_history()

# m_one.plot_predictions()

# IMPORTANTE: TensorBoard e Weights & Biases são libs extras pra monitorar o desempenho dos experimentos/modelagens

# o modelo tá muito agressivo, e tá causando overfitting. tentar ser mais cauteloso nos parâmetros do modelo, talvez reduzindo
# seu tamanho e gradualmente ir aumentando.
# - seria interessante também setar uma seed padrão no modelo, pra sempre usar os mesmos parâmetros de teste

# https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/01_neural_network_regression_in_tensorflow.ipynb
