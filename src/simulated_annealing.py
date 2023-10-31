import tensorflow as tf
from data.data_prep import build_houses_to_rent
import random
import math
from datetime import datetime
import copy

X_train, X_test, y_train, y_test = build_houses_to_rent()
X_train = tf.constant(X_train)
X_test = tf.constant(X_test)
y_train = tf.constant(y_train)
y_test = tf.constant(y_test)


def build_model_and_predict(architecture):
    model = tf.keras.models.Sequential()
    for layer, units in architecture['units_per_layer'].items():
        model.add(tf.keras.layers.Dense(
            units=units,
            activation="relu"),
        )

    model.add(tf.keras.layers.Dense(units=1))

    model.compile(
        loss=tf.keras.losses.mae,
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=architecture['learning_rate']),
        metrics=['mae'],
    )

    model.fit(
        X_train,
        y_train,
        batch_size=int(architecture['batch_size']),
        epochs=10000,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
        )],
        verbose=0,
    )

    y_pred = tf.squeeze(model.predict(X_test))
    return tf.metrics.mean_absolute_error(y_true=y_test, y_pred=y_pred).numpy()


def accept_neighbor(current_mae, new_mae, temperature):
    if new_mae < current_mae:
        return True
    else:
        probability = math.exp((current_mae - new_mae) / temperature)
        return random.random() < probability


def build_arch(current_arch):
    new_arch = copy.deepcopy(current_arch)

    def change_choise(): return random.choice([-1, 0, 1])

    last_layer = max(new_arch['units_per_layer'].keys())
    choise = change_choise()
    if choise > 0 and last_layer < 10:
        new_arch['units_per_layer'][last_layer + 1] = 1
    if choise < 0 and last_layer > 1:
        new_arch['units_per_layer'].pop(last_layer)

    for layer, units in new_arch['units_per_layer'].items():
        choise = change_choise()
        if choise != 0:
            new_units = units + random.randint(-50, 50)
            if 1 < new_units < 1000:
                new_arch['units_per_layer'][layer] = new_units

    choise = change_choise()
    if choise > 0 and new_arch['learning_rate'] != 0.1:
        new_arch['learning_rate'] = new_arch['learning_rate'] * 10
    if choise < 0 and new_arch['learning_rate'] != 0.00001:
        new_arch['learning_rate'] = new_arch['learning_rate'] / 10

    choise = change_choise()
    if choise > 0 and new_arch['batch_size'] < 4096.0:
        new_arch['batch_size'] = new_arch['batch_size'] * 2
    if choise < 0 and new_arch['batch_size'] > 4.0:
        new_arch['batch_size'] = new_arch['batch_size'] / 2

    return new_arch

# -------------------------------------------------------------------


temperature = 100
cooling_rate = 0.99

current_arch = {
    'units_per_layer': {
        1: 1,
    },
    'learning_rate': 0.1,
    'batch_size': 32,
    'mae': None,
}
current_arch['mae'] = build_model_and_predict(current_arch)
best_arch = copy.deepcopy(current_arch)

i = 0
while temperature > 1:
    with open("src/logs/log_01.txt", "a") as file:
        init = datetime.now()
        i += 1
        print("-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-", file=file)
        print("looping: ", str(i), file=file)
        print("temp: " + str(temperature), file=file)
        print("cooling_rate: " + str(cooling_rate), file=file)

        new_arch = build_arch(current_arch)
        new_arch['mae'] = build_model_and_predict(new_arch)

        new_become_current = False
        if accept_neighbor(current_arch['mae'], new_arch['mae'], temperature):
            new_become_current = True
            current_arch = copy.deepcopy(new_arch)
        print("new_become_current: " + str(new_become_current), file=file)

        current_become_best = False
        if round(current_arch['mae'], 6) < round(best_arch['mae'], 6):
            current_become_best = True
            best_arch = copy.deepcopy(current_arch)
        print("current_become_best: " + str(current_become_best), file=file)

        temperature *= cooling_rate

        print("training_time: " + str(datetime.now() - init), file=file)
        print("new: ", file=file)
        print(" - units_per_layer: " +
              str(new_arch['units_per_layer']), file=file)
        print(" - learning_rate: " + str(new_arch['learning_rate']), file=file)
        print(" - batch_size: " + str(new_arch['batch_size']), file=file)
        print(" - mae: " + str(new_arch['mae']), file=file)

        print("current: ", file=file)
        print(" - units_per_layer: " +
              str(current_arch['units_per_layer']), file=file)
        print(" - learning_rate: " +
              str(current_arch['learning_rate']), file=file)
        print(" - batch_size: " + str(current_arch['batch_size']), file=file)
        print(" - mae: " + str(current_arch['mae']), file=file)

        print("best: ", file=file)
        print(" - units_per_layer: " +
              str(best_arch['units_per_layer']), file=file)
        print(" - learning_rate: " +
              str(best_arch['learning_rate']), file=file)
        print(" - batch_size: " + str(best_arch['batch_size']), file=file)
        print(" - mae: " + str(best_arch['mae']), file=file)

        file.close()
