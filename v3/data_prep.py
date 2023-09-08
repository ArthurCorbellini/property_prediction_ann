import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.encoding import OrdinalEncoder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer, make_column_transformer

import utils.graph_util as gu
import utils.outlier_util as ou

"""
------------------------- Data Types -------------------------
  Numerical are those of type int and float and categorical those of type object.
    Categorical variables: 1
    Numerical variables: 12

- target
price              int64 

- continuous
condo              int64 > outliers(ok) > standardized(ok)
size               int64 > outliers(ok) > standardized(ok)
latitude         float64 > outliers(ok) > standardized(ok)
longitude        float64 > outliers(ok) > standardized(ok)

- discrete
rooms              int64 > outliers(ok) > standardized(ok)
toilets            int64 > outliers(ok) > standardized(ok)
suites             int64 > outliers(ok) > standardized(ok)
parking            int64 > outliers(ok) > standardized(ok)
elevator           int64 > (ok)
furnished          int64 > (ok)
swimming_pool      int64 > (ok)

- categoric
district          object > onehoted(ok)

dtype: object

 - There are 7 discrete variables
rooms  -> values:  [2  1  3  4  5 10  6  7]
toilets  -> values:  [2 3 4 1 5 6 7 8]
suites  -> values:  [1 3 2 4 0 5]
parking  -> values:  [1 2 3 4 5 6 8 9 0 7]
elevator  -> values:  [0 1]
furnished  -> values:  [0 1]
swimming_pool  -> values:  [0 1]

 - There are 4 numerical and continuous variables
condo
size
latitude
longitude
"""


class DataPrep:

    def __init__(self, neg_type):
        self.data_set = pd.read_csv(
            "datasets/sao-paulo-properties-april-2019.csv")

        self._prepare_data(neg_type)
        self._build_train_test_data()
        # self._outlier_handling()
        self._feature_scaling()

    def _prepare_data(self, neg_type):
        # Seta o tipo de negociação (rent ou sale) e remove as colunas desnecessárias:
        # - "property_type" só tem um tipo;
        # - "negotiation_type" só tem um tipo agora;
        # - "new" possui poucas amostras;
        df = self.data_set[self.data_set["negotiation_type"] == neg_type]
        self.data_frame = df.drop(
            ['property_type', 'negotiation_type', 'new'], axis=1)
        self._distinguish_variables()

    def _distinguish_variables(self):
        # Identificação dos tipos de variáveis, se são numéricas ou categóricas
        # - Numéricas são as int64 e float64;
        # - Categóricas são as object;
        self.numerical = [
            var for var in self.data_frame.columns if self.data_frame[var].dtype != 'O']
        self.categorical = [
            var for var in self.data_frame.columns if self.data_frame[var].dtype == 'O']
        # Identificação das variáveis discretas:
        # - Se tiver MENOS que 20 registros, é considerado uma variável DISCRETA;
        self.discrete = []
        for var in self.numerical:
            if len(self.data_frame[var].unique()) < 20:
                self.discrete.append(var)
        # Identificação das variáveis contínuas:
        # - Se tiver MAIS que 20 registros, é considerado uma variável CONTÍNUA;
        self.continuous = [
            var for var in self.numerical if var not in self.discrete and var not in ['price']]

    def _build_train_test_data(self):
        X = self.data_frame.loc[:, self.data_frame.columns != "price"]
        Y = self.data_frame["price"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X, Y, test_size=0.2, random_state=0)

# ---------------------------------------------------------------------------------------------

    def _outlier_handling(self):
        self.x_train, self.x_test, self.y_train, self.y_test = ou.trimmer_normal_gaussian(
            variables=['latitude', 'longitude'],
            x_train=self.x_train,
            x_test=self.x_test,
            y_train=self.y_train,
            y_test=self.y_test,
        )
        self.x_train, self.x_test, self.y_train, self.y_test = ou.trimmer_skewed_iqr(
            variables=['condo', 'size'],
            x_train=self.x_train,
            x_test=self.x_test,
            y_train=self.y_train,
            y_test=self.y_test,
        )
        self.x_train, self.x_test, self.y_train, self.y_test = ou.trimmer_normal_quantile(
            variables=['rooms', 'toilets', 'suites', 'parking'],
            x_train=self.x_train,
            x_test=self.x_test,
            y_train=self.y_train,
            y_test=self.y_test,
        )

    def _feature_scaling(self):
        ct = make_column_transformer(
            (MinMaxScaler(),  # ou StandardScaler(), dependendo de qual performa melhor
             ["condo", "size", "rooms", "toilets", "suites", "parking", "latitude", "longitude"]),
            (OneHotEncoder(categories="auto",
                           # to return k-1 (drop=false to return k dummies)
                           drop="first",
                           sparse_output=False,
                           handle_unknown="error"),
             ["district"]),
            remainder="passthrough"
        )

        ct.fit(self.x_train)
        self.x_train = ct.transform(self.x_train)
        self.x_test = ct.transform(self.x_test)


dp = DataPrep("rent")
# pd.set_option('display.max_columns', None)
# print(dp.x_train)
