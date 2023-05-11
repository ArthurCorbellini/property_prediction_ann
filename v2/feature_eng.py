import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


# Parte destinada à Engenharia de Requisitos:

print('\r\n-------------------------------------------\r\n')
print('- Overview do dataset')
pd.set_option('display.max_columns', None)
data = pd.read_csv('datasets/sao-paulo-properties-april-2019.csv')
print(data.head())

# -------------------------------------------------------------------------------------------
# Valores nulos
print('\r\n-------------------------------------------\r\n')
print('- Verificação se há valores nulos')
print(data.isnull().sum())
# Não foram encontrados valores nulos no dataset,
# logo, não há necessidade para tratamento de valores nulos.

# -------------------------------------------------------------------------------------------
# Cardinalidade e labels raras
# - Variáveis categóricas:
#   - district -> alta cardinalidade;
#   - negotiation_type -> baixa cardinalidade;
#   - property_type -> baixa cardinalidade (um valor apenas, pode ser ignorada);
print('\r\n-------------------------------------------\r\n')
print('- Quantidade de categorias por variável')
cat_cols = ['district', 'negotiation_type']
for col in cat_cols:
    print(col, '-', data[col].nunique())
print('\r\n')

total_houses = len(data)
print('total houses: ', total_houses)

# Let's plot the category frequency.
# That is, the percentage of houses with each label.

# For each categorical variable
for col in cat_cols:
    # Count the number of houses per category
    # and divide by total houses.
    # That is, the percentage of houses per category.

    temp_df = pd.Series(data[col].value_counts() / total_houses)

    # Make plot with these percentages.
    fig = temp_df.sort_values(ascending=False).plot.bar()
    fig.set_xlabel(col)

    # Add a line at 5 % to flag the threshold for rare categories.
    fig.axhline(y=0.05, color='red')
    fig.set_ylabel('Percentage of houses')
    plt.show()


# --/-/-/-/--/-/


def calculate_mean_target_per_category(df, var):

    # total number of houses
    total_houses = len(df)

    # percentage of houses per category
    temp_df = pd.Series(df[var].value_counts() / total_houses).reset_index()
    temp_df.columns = [var, 'perc_houses']

    # add the mean SalePrice
    temp_df = temp_df.merge(df.groupby([var])['price'].mean().reset_index(),
                            on=var,
                            how='left')

    return temp_df


temp_df = calculate_mean_target_per_category(data, 'district')


def plot_categories(df, var):

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.xticks(df.index, df[var], rotation=90)

    ax2 = ax.twinx()
    ax.bar(df.index, df["perc_houses"], color='lightgrey')
    ax2.plot(df.index, df["price"], color='green', label='Seconds')
    ax.axhline(y=0.05, color='red')
    ax.set_ylabel('percentage of houses per category')
    ax.set_xlabel(var)
    ax2.set_ylabel('Average Sale Price per category')
    plt.show()


plot_categories(temp_df, 'district')


# -------------------------------------------------------------------------------------------

# 13000 registros no total, sendo metade "venda" e metade "aluguel".
# Desses + -7000, metade são em bairros de "rare label". Se as rares labels forem removidas, sobram + -3500.

# continuar na parte dos rarelabels
# # https://github.com/solegalli/feature-engineering-for-machine-learning/blob/main/Section-04-Variable-Characteristics/3-Rare-Labels.ipynb

# -------------------------------------------------------------------------------------------
# Outliers

# Function to create a histogram, a Q-Q plot and
# a boxplot.
def diagnostic_plots(df, variable):
    # The function takes a dataframe (df) and
    # the variable of interest as arguments.

    # Define figure size.
    plt.figure(figsize=(16, 4))

    # histogram
    plt.subplot(1, 3, 1)
    sns.histplot(df[variable], bins=30)
    plt.title('Histogram')

    # Q-Q plot
    plt.subplot(1, 3, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.ylabel('RM quantiles')

    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')

    plt.show()
