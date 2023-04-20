import torch
import torch.optim as optim
from torch import nn
# from torch.nn.modules.activation import ReLU


class NeuralNetworkModel:

    def __init__(self, x_train, x_test, y_train, y_test):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.x_train = self._prepare_data_frame(x_train)
        self.x_test = self._prepare_data_frame(x_test)
        self.y_train = self._prepare_data_frame(y_train)
        self.y_test = self._prepare_data_frame(y_test)

        self.model = self._build_model()
        self._train()

    def _prepare_data_frame(self, data_frame):
        # pylint: disable=no-member
        # return torch.from_numpy(data_frame.values).float().to(self.device)
        return torch.tensor(data_frame.values, dtype=torch.float, device=self.device)
        # pylint: enable=no-member

    def _train(self):
        # Variável que mede o quão bem a rede está prevendo os resultados
        criterion = nn.MSELoss()
        # Variável que atualiza os pesos da rede conforme ela vai aprendendo
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        # Treinamento da rede, interando pelo dataset 5 vezes
        for epoch in range(5):
            total_loss = 0
            for i in range(len(self.x_train)):
                # Faz predição
                y_pred = self.model(self.x_train[i])
                # Mede a perda (medido vs esperado)
                loss = criterion(y_pred, self.y_train[i])
                # Mede o quão bem foi a medição
                total_loss += loss.item()

                # Atualiza o modelo conforme o que foi aprendido
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # O valor da perda precisa diminuir a cada iteração (indica que o modelo está aprendendo bem)
            print("Total Loss: ", total_loss)

    def _build_model(self):
        """
          Número de inputs: 5
          Número de hidden units: 100
          Número de hidden layers: 1
          Número de saídas: 1
          Função de ativação: ReLU
        """
        input_layer = 3
        output_layer = 1
        hidden_layer = 256
        f_atv = nn.ReLU()

        model = nn.Sequential(nn.Linear(input_layer, hidden_layer), f_atv,
                              nn.Linear(hidden_layer, hidden_layer), f_atv,
                              nn.Linear(hidden_layer, hidden_layer), f_atv,
                              nn.Linear(hidden_layer, hidden_layer), f_atv,
                              nn.Linear(hidden_layer, hidden_layer), f_atv,
                              nn.Linear(hidden_layer, hidden_layer), f_atv,
                              nn.Linear(hidden_layer, hidden_layer), f_atv,
                              nn.Linear(hidden_layer, hidden_layer), f_atv,
                              nn.Linear(hidden_layer, output_layer))
        model.to(self.device)
        return model
