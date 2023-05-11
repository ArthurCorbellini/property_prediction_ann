import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from utils.outlier_utils import OutlierUtils

from feature_engine import imputation as mdi
from feature_engine import discretisation as dsc
from feature_engine import encoding as ce
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from feature_engine.outliers import OutlierTrimmer


def _about_variables():
    print('------------------------------------------------------------------')
    print('Data Types -------------------------------------------------------')
    print('------------------------------------------------------------------')
    print(' - Numerical are those of type int and float and categorical those of type object.')
    print('  - Categorical variables: {}'.format(len(categorical)))
    print('  - Numerical variables: {}'.format(len(numerical)))
    print(data.dtypes)
    print('\r')
    print(' - There are {} discrete variables'.format(len(discrete)))
    for var in discrete:
        print(var, ' -> values: ', data[var].unique())
    print('\r')
    print(' - There are {} numerical and continuous variables'.format(len(continuous)))
    for var in continuous:
        print(var)


def _about_missing_values():
    print('------------------------------------------------------------------')
    print('Missing Values ---------------------------------------------------')
    print('------------------------------------------------------------------')
    print(' - Number of missing values:')
    print(data.isnull().sum())


def _about_outliers_continuous():
    print('------------------------------------------------------------------')
    print('Outliers in continuous -------------------------------------------')
    print('------------------------------------------------------------------')
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
    print('------------------------------------------------------------------')
    print('Outliers in discrete ---------------------------------------------')
    print('------------------------------------------------------------------')
    plt.figure(figsize=(18, 9))
    position = 0
    for var in discrete:
        position += 1
        plt.subplot(2, 4, position)
        fig = (data.groupby(var)[var].count() / len(data)).plot.bar()
        fig.set_title(var)
        fig.set_ylabel('Percentage of observations per label')
        fig.axhline(y=0.05, color='red')

        position += 1
        plt.subplot(2, 4, position)
        fig = data.groupby(var)['price'].median().plot()
        fig.set_ylabel('Median house Price per label')
        fig.set_title(var)
        fig.ticklabel_format(style='plain')

    plt.tight_layout()
    plt.show()


def _about_cardinality():
    print('------------------------------------------------------------------')
    print('Cardinality: -----------------------------------------------------')
    print('------------------------------------------------------------------')
    data[categorical].nunique().plot.bar(figsize=(10, 6))
    plt.title('CARDINALITY: Number of categories in categorical variables')
    plt.xlabel('Categorical variables')
    plt.ylabel('Number of different categories')

    plt.tight_layout()
    plt.show()


def about():
    print('------------------------------------------------------------------')
    print('Overview: --------------------------------------------------------')
    print('------------------------------------------------------------------')
    print(data.head())

    print('\r')
    _about_variables()
    print('\r')
    _about_missing_values()
    print('\r')
    _about_outliers_continuous()
    print('\r')
    _about_outliers_discrete()
    print('\r')
    _about_cardinality()


# --------------------------------------------------------------------------
# PRE-PROCESSING -----------------------------------------------------------
# --------------------------------------------------------------------------

NEG_TYPE = 'rent'

pd.set_option('display.max_columns', None)
data_set = pd.read_csv('datasets/sao-paulo-properties-april-2019.csv')

data = data_set[data_set['negotiation_type'] == NEG_TYPE]
# Removendo colunas desnecessárias
data = data.drop(['property_type', 'negotiation_type', 'new'], axis=1)

# Transformando as variáveis "true/false" em categóricas.
# for var in ['elevator', 'furnished', 'swimming_pool', 'new']:
#     data[var] = data[var].astype('O')

numerical = [var for var in data.columns if data[var].dtype != 'O']
categorical = [var for var in data.columns if data[var].dtype == 'O']

discrete = []
for var in numerical:
    if len(data[var].unique()) < 20:
        discrete.append(var)

continuous = [
    var for var in numerical if var not in discrete and var not in ['price']]

# --------------------------------------------------------------------------
# FEATURE-ENG --------------------------------------------------------------
# --------------------------------------------------------------------------


def plot_boxplot_and_hist(data, variable):

    # creating a figure composed of two matplotlib.Axes
    # objects (ax_box and ax_hist)

    f, (ax_box, ax_hist) = plt.subplots(
        2, sharex=True, gridspec_kw={"height_ratios": (0.50, 0.85)}
    )

    # assigning a graph to each ax
    sns.boxplot(x=data[variable], ax=ax_box)
    sns.histplot(data=data, x=variable, ax=ax_hist)

    # Remove x axis name for the boxplot
    ax_box.set(xlabel="")
    plt.title(variable)
    plt.show()


# --------------------------------------------------------------------------
# about()
x_train, x_test, y_train, y_test = train_test_split(
    data.drop('price', axis=1),
    data['price'],
    test_size=0.2,
    random_state=0)

trimmer = OutlierUtils.trimmer_iqr(variables=['condo', 'size'])

pipe = Pipeline([
    ('outliers', OutlierUtils.trimmer_iqr(variables=['condo', 'size'])),
])

pipe.fit(x_train)

# print(pipe.left_tail_caps_)
# print(pipe.right_tail_caps_)
plot_boxplot_and_hist(x_train, "condo")
x_train = pipe.transform(x_train)  # aplica todos os transforms em sequência
plot_boxplot_and_hist(x_train, "condo")

# --------------------------------------------------------------------------


pipe_v0 = Pipeline([
    # Desconsiderar:
    # - variável latitude;
    # - variável longitude.

    # Tratamento dos outliers:
    # - variável condo;
    # - variável size.

    # Tratamento das rare labels:
    # - variável rooms;
    # - variável toilets;
    # - variável suites;
    # - variável parking.

    # One-hot encoding:
    # - variável district.

    # Standardize em todas as variáveis;

    # ('rare_label_enc', ce.RareLabelEncoder(
    #     tol=0.05, n_categories=1, variables=categorical+discrete)),

    # ('categorical_enc', ce.OrdinalEncoder(
    #     encoding_method='ordered', variables=categorical+discrete)),

    # ('discretisation', dsc.EqualFrequencyDiscretiser(
    #     q=5, return_object=True, variables=numerical)),

    # ('encoding', ce.OrdinalEncoder(encoding_method='ordered', variables=numerical)),

    # ('scaler', StandardScaler())
])

# x_train[discrete] = x_train[discrete].astype('O')
# x_test[discrete] = x_test[discrete].astype('O')

# pipe.fit(x_train, y_train)

# data = x_train
# about()
