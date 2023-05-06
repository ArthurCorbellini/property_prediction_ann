import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPrep:

    def __init__(self, neg_type):
        self.data_set = pd.read_csv(
            "datasets/sao-paulo-properties-april-2019.csv")
        self.data_frame = self._prepare_data(neg_type)
        self.x_train, self.x_test, self.y_train, self.y_test = self._build_train_test_data()

    def _prepare_data(self, neg_type):
        df = self.data_set[self.data_set["Negotiation Type"] == neg_type]
        df = self._remove_unnecessary_columns(df)
        return df

    def _remove_unnecessary_columns(self, df):
        drop = ["New", "Property Type", "Negotiation Type",
                "District", "Latitude", "Longitude"]
        return df.drop(drop, axis=1)

    def _build_train_test_data(self):
        X = self.data_frame.loc[:, self.data_frame.columns != "Price"].values
        Y = self.data_frame["Price"].values

        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=0)
        x_train, x_test = self._feature_scaling(x_train, x_test)

        return x_train, x_test, y_train, y_test

    def _feature_scaling(self, x_train, x_test):
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        return x_train, x_test
