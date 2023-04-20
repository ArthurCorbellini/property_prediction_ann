from sklearn.ensemble import RandomForestRegressor


class RandomForestModel:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = RandomForestRegressor()
        self._train()

    def _train(self):
        self.model = self.model.fit(self.x_train, self.y_train)

    def print_score(self):
        print("------------------------------------------------------------------------")
        print("Score: " + str(self.model.score(self.x_test, self.y_test)))
        print("------------------------------------------------------------------------")

    def describe_example(self):
        N = 0
        print("------------------------------------------------------------------------")
        print("Imóvel em questão: ")
        print(self.x_test.iloc[N].head(10))
        print("------------------------------------------------------------------------")
        print("Valor sugerido (justo): ")
        print(self.model.predict(self.x_test.iloc[N].values.reshape(1, -1)))
        print("------------------------------------------------------------------------")
        print("Valor real: ")
        print(self.y_test.iloc[N])
        print("------------------------------------------------------------------------")
