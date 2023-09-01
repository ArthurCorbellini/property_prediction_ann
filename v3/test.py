from data_prep import DataPrep
from model_one import ModelOne

# "sale" ou "rent"
dp = DataPrep("rent")
# print(dp.data_frame)
# print(dp.x_train)

m_one = ModelOne(dp.x_train, dp.x_test, dp.y_train, dp.y_test)

# print(m_one.results())
m_one.model.summary()

# For regression, whe have two main metrics:
# - MAE: mean absolute error, "on average, how wrong is each of my model's predictions";
# - MSE: mean square error, "square the average errors". When larger errors are more significant than smaller errors;
# - Huber: combination of MSE and MAE. Less sensitive to outliers than MSE.
print("-- MAE")
print(m_one.mae())
print("-- MSE")
print(m_one.mse())

# IMPORTANTE: TensorBoard e Weights & Biases são libs extras pra monitorar o desempenho dos experimentos/modelagens

# m_one.plot_predictions()
