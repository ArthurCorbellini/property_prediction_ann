
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import utils.outlier_util as ou


def build_houses_to_rent():
    data = pd.read_csv("src/data/datasets/houses_to_rent_v2.csv")

    data.rename({"parking spaces": "parking_spaces",
                 "hoa (R$)": "hoa",
                 "rent amount (R$)": "rent",
                 "property tax (R$)": "property_tax",
                 "fire insurance (R$)": "fire_insurance",
                 "total (R$)": "total"}, axis="columns", inplace=True)

    # pré-processamento
    df = data.drop(["total", "floor"], axis=1)
    df.drop_duplicates(inplace=True)

    # remoção de outliers
    df = ou.trimmer_skewed_iqr(
        variables=['area', 'hoa', 'property_tax', 'fire_insurance'],
        data_frame=df,
    )
    df = ou.trimmer_normal_quantile(
        variables=['rooms', 'bathroom', 'parking_spaces'],
        data_frame=df,
    )

    # separação dos dados de treino e teste
    X = df.loc[:, df.columns != "rent"]
    Y = df["rent"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=1)

    # feature engineering
    ct = make_column_transformer(
        (MinMaxScaler(),  # ou StandardScaler(), dependendo de qual performa melhor
         ['area', 'rooms', 'bathroom', 'parking_spaces', 'hoa', 'property_tax', 'fire_insurance']),
        (OneHotEncoder(categories="auto",
                       # to return k-1 (drop=false to return k dummies)
                       drop="first",
                       sparse_output=False,
                       handle_unknown="error"),
         ['city', 'animal', 'furniture']),
        remainder="passthrough"
    )

    ct.fit(X_train)
    X_train = ct.transform(X_train)
    X_test = ct.transform(X_test)

    return X_train, X_test, y_train, y_test
