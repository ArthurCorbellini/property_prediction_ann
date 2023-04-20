import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPrep:

    def __init__(self, neg_type):
        self.all_data = pd.read_csv(
            "webscrap/datasets/sao-paulo-properties-april-2019.csv")
        self.data_frame = self.all_data[self.all_data["Negotiation Type"] == neg_type]
        px.set_mapbox_access_token(
            open("../private/mapbox_access_token", encoding="utf-8").read())

    def build_graph(self):
        img = px.scatter_mapbox(self.data_frame,
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

        return img

    def build_histogram(self):
        self.data_frame.hist(bins=30, figsize=(30, 15))
        plt.show()

    def describe_data_frame(self):
        """
            Descreve dados gerais do conjunto, médias, máximo, mínimo, desvio padrão....
        """
        print(" -- Dados gerais:")
        print(self.data_frame)
        print(self.data_frame.describe())
        print(" -- Mediana:")
        print(self.data_frame["Price"].median())
        print(" -- Contagem por tipo:")
        print(self.data_frame["Property Type"].value_counts())

    def process_data_rfm(self):
        self._remove_unnecessary_columns()
        self._fit_categorical_data()
        return self._build_train_test_data()

    def process_data_nnm(self):
        self._remove_unnecessary_columns()
        self._standardize_data()
        return self._build_train_test_data()

    def _standardize_data(self):
        """ 
           Deixa os dados em uma escala menor, maior ou menor que 0
        """
        # for column in self.data_frame.columns:
        #     self.data_frame[column] = (self.data_frame[column] - self.data_frame[column].mean())/self.data_frame[column].std()
        columns = self.data_frame.columns
        self.data_frame = pd.DataFrame(
            StandardScaler().fit_transform(self.data_frame), columns=columns)

    def _remove_unnecessary_columns(self):
        """
            Removendo colunas desnecessárias:
            - New: há poucos registros "New = 1", o que pode confundir o modelo;
            - Property Type: só há um tipo neste dataSet (apartment);
            - Negotiation Type: já foi tratado anteriormente.
        """
        self.data_frame = self.data_frame.drop(
            ["New", "Property Type", "Negotiation Type", "District", "Latitude", "Longitude"], axis=1)

    def _fit_categorical_data(self):
        """
            Dependendo do modelo utilizado, há a necessidade de ajustar os dados categóricos:
            - Transformar a categoria (string) em um valor numérico (int ou bit);
            - Nesse caso, há apenas o campo "District" é categórico;
            - Pode ser tatado de duas formas, Integer Encoding ou One-Hot Encoding;
            - Abaixo, será utilizado o one-hot encoding.
        """
        one_hot = pd.get_dummies(self.data_frame["District"])
        self.data_frame = self.data_frame.drop(
            "District", axis=1).join(one_hot)

    def _build_train_test_data(self):
        """
            Separadação da coluna "Price" dos demais dados da tabela
            - X é o data frame de parâmetros;
            - Y é o data frame de respostas;
            - O modelo se guia pelo index da tabela, que permanece o mesmo.

            Separação dos dados de treino e de teste que o modelo irá se basear
            - 30% dos dados serão utilizados para teste;
        """
        X = self.data_frame.loc[:, self.data_frame.columns != "Price"]
        Y = self.data_frame["Price"]

        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.3)

        return x_train, x_test, y_train, y_test
