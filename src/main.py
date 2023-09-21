from data.data_prep import build_houses_to_rent
from models.random_model import RandomModel
from datetime import datetime

for i in range(100):
    with open("src/logs/log_1.txt", "a") as file:
        init = datetime.now()
        X_train, X_test, y_train, y_test = build_houses_to_rent()
        rm = RandomModel(X_train, X_test, y_train, y_test)

        print("-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-", file=file)
        print(rm.config, file=file)
        print("Results:", file=file)
        print("  Training time: " + str(datetime.now() - init), file=file)
        print("  MAE (train) -> " + str(rm.mae_train()), file=file)
        print("  MAE ---------> " + str(rm.mae()), file=file)
        print("  MSE (train) -> " + str(rm.mse_train()), file=file)
        print("  MSE ---------> " + str(rm.mse()), file=file)
        print("\n", file=file)

        file.close()
