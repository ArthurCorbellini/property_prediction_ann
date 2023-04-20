
"""
  Para rodar o código abaixo, as seguites libs são necessárias:
    pip install pandas
    pip install plotly
    pip install matplotlib
    pip install scikit-learn
"""

from data_prep import DataPrep
from random_forest_model import RandomForestModel
from neural_network_model import NeuralNetworkModel

# NEG_TYPE = "rent"
NEG_TYPE = "sale"

# Inicialização e visualização dos dados
dp = DataPrep(NEG_TYPE)
# dp.build_graph().show()
# dp.describe_data_frame()
# dp.build_histogram()

# Prepação dos dados
# x_train, x_test, y_train, y_test = dp.process_data_rfm()

# Treinamento do modelo
# rfm = RandomForestModel(x_train, x_test, y_train, y_test)
# rfm.print_score()
# rfm.describe_example()

# ---------------
# x_train, x_test, y_train, y_test = dp.process_data_nnm()
# inputs = ["Rooms", "Toilets", "Suites", "Parking", "Furnished"]
# x_train = x_train[inputs]

# print(x_train)
# print(y_train)
# nnm = NeuralNetworkModel(x_train, x_test, y_train, y_test)
# ---------------

# print(dp.data_frame.head(10))
dp._remove_unnecessary_columns()
# print(dp.data_frame.head(10))
dp._standardize_data()

inputs = ["Rooms", "Toilets", "Suites"]
output = ["Price"]
# print(dp.data_frame[inputs].head(10))
# print(dp.data_frame[output].head(10))

nnm = NeuralNetworkModel(
    dp.data_frame[inputs], dp.data_frame[inputs], dp.data_frame[output], dp.data_frame[output])


# print(dp.data_frame.head(10))
