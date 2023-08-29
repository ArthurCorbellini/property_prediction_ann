from data_prep import DataPrep
from model_one import ModelOne

# "sale" ou "rent"
dp = DataPrep("rent")
# print(dp.data_frame)
# print(dp.x_train)

m_one = ModelOne(dp.x_train, dp.x_test, dp.y_train, dp.y_test)

print(m_one.predict())
