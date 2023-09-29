import tensorflow as tf
from data.data_prep import build_houses_to_rent
import numpy as np
import pandas as pd


X_train, X_test, y_train, y_test = build_houses_to_rent()
X_train = tf.constant(X_train)
X_test = tf.constant(X_test)
y_train = tf.constant(y_train)
y_test = tf.constant(y_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=169, activation="relu"),
    tf.keras.layers.Dense(units=11, activation="relu"),
    tf.keras.layers.Dense(units=159, activation="relu"),
    tf.keras.layers.Dense(units=28, activation="relu"),
    tf.keras.layers.Dense(units=88, activation="relu"),
    tf.keras.layers.Dense(units=1)
])

model.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['mae'],
)

model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=55,
    # callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)],
    # verbose=0,
)

y_pred_train = tf.squeeze(model.predict(X_train))
y_pred = tf.squeeze(model.predict(X_test))

print("TRAIN: ")
print(" MAE: " + str(tf.metrics.mean_absolute_error(y_true=y_train, y_pred=y_pred_train)))
print(" MSE: " + str(tf.metrics.mean_squared_error(y_true=y_train, y_pred=y_pred_train)))
print("TEST: ")
print(" MAE: " + str(tf.metrics.mean_absolute_error(y_true=y_test, y_pred=y_pred)))
print(" MSE: " + str(tf.metrics.mean_squared_error(y_true=y_test, y_pred=y_pred)))

y_pred = y_pred.numpy()
y_test = y_test.numpy()
print(pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}).to_string())


# vertical_y_pred = y_pred.reshape(len(y_pred), 1)
# vertical_y_test = np.array(y_test).reshape(len(y_test), 1)
# np.set_printoptions(precision=2)
# # np.set_printoptions(precision=2, threshold=np.inf)
# print(np.concatenate((vertical_y_pred, vertical_y_test), 1))
