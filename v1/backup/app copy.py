
"""
  Para rodar o código abaixo, as seguites libs são necessárias:
    pip install pandas
    pip install plotly
    pip install matplotlib
    pip install scikit-learn
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# import seaborn as sns

# ------------------------------------------------------------------------
# CARREGAMENTO E VISUALIZAÇÃO DOS DADOS ----------------------------------
# ------------------------------------------------------------------------

# carrega o csv
all_data = pd.read_csv("webscrap/datasets/sao-paulo-properties-april-2019.csv")

# divide o data_set em dois distintos, de vendas e aluguel
NEG_TYPE = "rent"
# NEG_TYPE = "sale"
data_frame = all_data[all_data["Negotiation Type"] == NEG_TYPE]

# carrega o token de acesso do mapbox.
#   - O token em questão está dentro do arquivo "mapbox_access_token";
#   - Gerado ao criar uma conta no mapbox;
px.set_mapbox_access_token(
    open("../private/mapbox_access_token", encoding="utf-8").read())

# cria o gráfico
img = px.scatter_mapbox(data_frame,
                        lat="Latitude",
                        lon="Longitude",
                        color="Price",
                        size="Size",
                        color_continuous_scale=px.colors.cyclical.IceFire,
                        size_max=15,
                        zoom=10,
                        opacity=0.3)

# ajusta a escala de cores do gráfico, para visualização
img.update_coloraxes(
    colorscale=[
        [0, 'rgb(166,206, 227, 0.5)'],
        [0.02, 'rgb(31,120,180,0.5)'],
        [0.05, 'rgb(178,233,138,0.5)'],
        [0.10, 'rgb(51,160,44,0.5)'],
        [0.15, 'rgb(251,154,153,0.5)'],
        [1, 'rgb(227,26,28,0.5)']
    ]
)

# centralização do gráfico e muda o template
img.update_layout(template="plotly_dark",
                  mapbox=dict(
                      center=go.layout.mapbox.Center(
                          lat=-23.543138,
                          lon=-46.69486)))

# mostra o gráfico
img.show()

# descreve dados gerais, médias, máximo, mínimo, desvio padrão....
print(" -- Dados gerais:")
print(data_frame)
print(data_frame.describe())
print(" -- Mediana:")
print(data_frame["Price"].median())
print(" -- Contagem por tipo:")
print(data_frame["Property Type"].value_counts())

# histograma
data_frame.hist(bins=30, figsize=(30, 15))
plt.show()

# ------------------------------------------------------------------------
# PRÉ-PROCESSAMENTO DOS DADOS --------------------------------------------
# ------------------------------------------------------------------------

# removendo colunas desnecessárias:
#   - New: há poucos registros "New = 1", o que pode confundir o modelo;
#   - Property Type: só há um tipo neste dataSet (apartment);
#   - Negotiation Type: já foi tratado anteriormente.
data_frame = data_frame.drop(
    ["New", "Property Type", "Negotiation Type"], axis=1)

# Dependendo do modelo utilizado, há a necessidade de ajustar os dados categóricos:
#   - Transformar a categoria (string) em um valor numérico (int ou bit);
#   - Nesse caso, há apenas o campo "District" é categórico;
#   - Pode ser tatado de duas formas, Integer Encoding ou One-Hot Encoding;
#   - Abaixo, será utilizado o one-hot encoding.
one_hot = pd.get_dummies(data_frame["District"])
data_frame = data_frame.drop("District", axis=1).join(one_hot)

# Separadação da coluna "Price" dos demais dados da tabela
#   - X é o data frame de parâmetros;
#   - Y é o data frame de respostas;
#   - O modelo se guia pelo index da tabela, que permanece o mesmo.
X = data_frame.loc[:, data_frame.columns != "Price"]
Y = data_frame["Price"]

# Separação dos dados de treino e de teste que o modelo irá se basear
#   - 30% dos dados serão utilizados para teste;
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# ------------------------------------------------------------------------
# TREINAMENTO DO MODELO --------------------------------------------------
# ------------------------------------------------------------------------

# Modelo usado foi o RandomForestRegressor
# Ha outros modelos q podem ser utilizados, como:
#   - DecisionTree Regressor;
#   - XGBoost Regressor;
#   - Redes neurais;
#   - Regressões logísticas;
rf_reg = RandomForestRegressor()
rf_reg.fit(x_train, y_train)

# ------------------------------------------------------------------------
# VALIDAÇÃO DO MODELO ----------------------------------------------------
# ------------------------------------------------------------------------

# Pega um valor aleatório dos data_sets de teste para validação do modelo
N = 0
print("------------------------------------------------------------------------")
print("Imóvel em questão: ")
print(x_test.iloc[N].head(10))
print("------------------------------------------------------------------------")
print("Valor sugerido (justo): ")
print(rf_reg.predict(x_test.iloc[N].values.reshape(1, -1)))
print("------------------------------------------------------------------------")
print("Valor real: ")
print(y_test.iloc[N])
print("------------------------------------------------------------------------")


print(rf_reg.score(x_test, y_test))
