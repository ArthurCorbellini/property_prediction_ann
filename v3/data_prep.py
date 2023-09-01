import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.encoding import OrdinalEncoder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

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
condo              int64 > outliers(todo) > zscored(ok)
size               int64 > outliers(todo) > zscored(ok)
latitude         float64 > outliers(todo) > zscored(ok)
longitude        float64 > outliers(todo) > zscored(ok)

- discrete
rooms              int64 > outliers(todo) > onehoted(ok)
toilets            int64 > outliers(todo) > onehoted(ok)
suites             int64 > outliers(todo) > onehoted(ok)
parking            int64 > outliers(todo) > onehoted(ok)
elevator           int64 > onehoted(ok)
furnished          int64 > onehoted(ok)
swimming_pool      int64 > onehoted(ok)

- categoric
district          object > onehoted(ok)

dtype: object

 - There are 7 discrete variables
rooms  -> values:  [ 2  1  3  4  5 10  6  7]
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

        # self._feature_engine()
        self._feature_engine_two()

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

    def _feature_engine_two(self):
        self._outlier_handling()
        # self._min_max_scaling() # zscore perfoma melhor que o min_max
        self._zscore_scaling()
        self._one_hot_encoding()

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
        # gu.plot_boxplot_and_hist(self.x_train, 'longitude')
        # gu.plot_boxplot_and_hist(self.x_train, 'latitude')

    def _zscore_scaling(self):
        """
            Aplicação do zscore nas variáveis que SÃO continuous
        """
        columns_index = [self.x_train.columns.get_loc(
            col) for col in self.continuous]
        sc = StandardScaler()
        sc.set_output(transform="pandas")
        self.x_train.iloc[:, columns_index] = sc.fit_transform(
            self.x_train.iloc[:, columns_index])
        self.x_test.iloc[:, columns_index] = sc.transform(
            self.x_test.iloc[:, columns_index])

    def _min_max_scaling(self):
        sc = MinMaxScaler()
        sc.set_output(transform="pandas")
        self.x_train = sc.fit_transform(self.x_train)
        self.x_test = sc.transform(self.x_test)

    def _one_hot_encoding(self):
        """
        Dependendo do modelo utilizado, há a necessidade de ajustar os dados categóricos:
        - Transformar a categoria (string) em um valor numérico (int ou bit);
        - Nesse caso, há apenas o campo "district" categórico;
        """
        encoder = OneHotEncoder(
            categories="auto",
            drop="first",  # to return k-1 (drop=false to return k dummies)
            sparse_output=False,
            handle_unknown="error"
        )
        ct = ColumnTransformer(
            [("ohe", encoder, self.categorical + self.discrete)],
            remainder="passthrough"
        )
        ct.set_output(transform="pandas")
        self.x_train = ct.fit_transform(self.x_train)
        self.x_test = ct.transform(self.x_test)

# ---------------------------------------------------------------------------------------------

    def _feature_engine(self):
        """
            Missing values: OK
            Outliers: 
            - variáveis contínuas:
              - condo e size -> Discretisation;
              - lat e lng -> remover outliers;
            - variáveis discretas:
        """
        self._build_train_test_data()
        self._discretize_continuous_variables()

    def _discretize_continuous_variables(self):
        disc = EqualFrequencyDiscretiser(
            q=100, variables=["condo", "size"], return_object=True)
        disc.fit(self.x_train)
        self.x_train = disc.transform(self.x_train)
        self.x_test = disc.transform(self.x_test)
        # pd.concat([self.x_train, self.y_train], axis=1).groupby(
        #     'condo')['price'].mean().plot()
        # plt.ylabel('teste')
        # plt.show()
        # pd.concat([self.x_train, self.y_train], axis=1).groupby(
        #     'size')['price'].mean().plot()
        # plt.ylabel('teste')
        # plt.show()
        enc = OrdinalEncoder(encoding_method='ordered')
        enc.fit(self.x_train, self.y_train)
        self.x_train = enc.transform(self.x_train)
        self.x_test = enc.transform(self.x_test)
        # pd.concat([self.x_train, self.y_train], axis=1).groupby(
        #    'condo')['price'].mean().plot()
        # plt.ylabel('teste')
        # plt.show()
        # pd.concat([self.x_train, self.y_train], axis=1).groupby(
        #     'size')['price'].mean().plot()
        # plt.ylabel('teste')
        # plt.show()


# ---------------------------------------------------------------------------------------------
"""
    def _feature_scaling(self, x_train, x_test):
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        return x_train, x_test
"""
dp = DataPrep("rent")
# pd.set_option('display.max_columns', None)
# print(dp.x_train)
