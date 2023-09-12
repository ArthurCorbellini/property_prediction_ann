import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _about_variables():
    print('\n\n\n')
    print('------------------------- Data Types -------------------------')
    print('\n')
    print('- Numerical are those of type int and float and categorical those of type object.')
    print(' - Categorical variables: {}'.format(len(categorical)))
    print(' - Numerical variables: {}'.format(len(numerical)))
    print(data.dtypes)
    print('\n')
    print(' - There are {} discrete variables'.format(len(discrete)))
    for var in discrete:
        print(var, ' -> values: ', data[var].unique())
    print('\n')
    print(' - There are {} numerical and continuous variables'.format(len(continuous)))
    for var in continuous:
        print(var)


def _about_missing_values():
    print('\n\n\n')
    print('----------------------- Missing Values -----------------------')
    print('\n')
    print(' - Number of missing values:')
    print(data.isnull().sum())


def _about_outliers_continuous():
    print('\n\n\n')
    print('--------------------- Outliers in continuous ---------------------')
    plt.figure(figsize=(16, 8))
    position = 0
    for var in continuous:
        position += 1
        plt.subplot(2, 4, position)
        fig = data.boxplot(column=var)
        fig.set_title('')
        fig.set_ylabel(var)

        position += 1
        plt.subplot(2, 4, position)
        fig = data[var].hist(bins=20)
        fig.set_ylabel('Number of houses')
        fig.set_xlabel(var)

    plt.tight_layout()
    plt.show()


def _about_outliers_discrete():
    """ 
      I will call outliers those values that are present in less than 5% of the houses.
      This is exactly the same as finding rare labels in categorical variables.
    """
    print('---------------------- Outliers in discrete ----------------------')
    plt.figure(figsize=(18, 9))
    position = 0
    for var in discrete:
        position += 1
        plt.subplot(2, 7, position)
        fig = (data.groupby(var)[var].count() / len(data)).plot.bar()
        fig.set_title(var)
        fig.set_ylabel('Percentage of observations per label')
        fig.axhline(y=0.01, color='red')

        position += 1
        plt.subplot(2, 7, position)
        fig = data.groupby(var)['price'].median().plot()
        fig.set_ylabel('Median house Price per label')
        fig.set_title(var)
        fig.ticklabel_format(style='plain')

    plt.tight_layout()
    plt.show()


def _about_cardinality():
    print('-------------------------- Cardinality ---------------------------')
    data[categorical].nunique().plot.bar(figsize=(10, 6))
    plt.title('CARDINALITY: Number of categories in categorical variables')
    plt.xlabel('Categorical variables')
    plt.ylabel('Number of different categories')

    plt.tight_layout()
    plt.show()
# ---------------------------------------------------------------------------


NEG_TYPE = 'rent'

pd.set_option('display.max_columns', None)
data_set = pd.read_csv('datasets/sao-paulo-properties-april-2019.csv')

data = data_set[data_set['negotiation_type'] == NEG_TYPE]

# Removendo colunas desnecessárias:
# - "property_type" só tem um tipo;
# - "negotiation_type" só tem um tipo;
# - "new" possui poucas amostras;
data = data.drop(['property_type', 'negotiation_type', 'new'], axis=1)

# Identificação dos tipos de variáveis, se são numéricas ou categóricas
# - Numéricas são as int64 e float64;
# - Categóricas são as object;
numerical = [var for var in data.columns if data[var].dtype != 'O']
categorical = [var for var in data.columns if data[var].dtype == 'O']

# Identificação das variáveis discretas:
# - Se tiver MENOS que 20 registros, é considerado uma variável DISCRETA;
discrete = []
for var in numerical:
    if len(data[var].unique()) < 20:
        discrete.append(var)

# Identificação das variáveis contínuas:
# - Se tiver MAIS que 20 registros, é considerado uma variável CONTÍNUA;
continuous = [
    var for var in numerical if var not in discrete and var not in ['price']]

_about_variables()
_about_missing_values()
_about_outliers_continuous()
_about_outliers_discrete()
_about_cardinality()

"""
    Missing values: OK
    Outliers: 
     - variáveis contínuas: será aplicado o método de Discretisation nelas; ----- CONTINUAR DAQUI -----
     - variáveis discretas:
"""

# https://github.com/solegalli/feature-engineering-for-machine-learning/blob/main/Section-19-Putting-it-altogether/02-Regression-house-prices.ipynb
