import tensorflow as tf
from data.data_prep import build_houses_to_rent
import numpy as np
import pandas as pd


X_train, X_test, y_train, y_test = build_houses_to_rent()
X_train = tf.constant(X_train)
X_test = tf.constant(X_test)
y_train = tf.constant(y_train)
y_test = tf.constant(y_test)

initializer = kernel_initializer = tf.keras.initializers.HeNormal(seed=1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=360,
                          activation="relu",
                          kernel_initializer=initializer),
    tf.keras.layers.Dense(units=168,
                          activation="relu",
                          kernel_initializer=initializer),
    tf.keras.layers.Dense(units=9,
                          activation="relu",
                          kernel_initializer=initializer),
    tf.keras.layers.Dense(units=1)
])

model.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['mae'],
)

model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=100,
    # callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)],
    verbose=0,
)

model.save("exports/neural_model_v1.h5")

y_pred_train = tf.squeeze(model.predict(X_train))
y_pred = tf.squeeze(model.predict(X_test))

print("TRAIN: ")
print(" MAE: " + str(tf.metrics.mean_absolute_error(y_true=y_train, y_pred=y_pred_train)))
print(" MSE: " + str(tf.metrics.mean_squared_error(y_true=y_train, y_pred=y_pred_train)))
print("TEST: ")
print(" MAE: " + str(tf.metrics.mean_absolute_error(y_true=y_test, y_pred=y_pred)))
print(" MSE: " + str(tf.metrics.mean_squared_error(y_true=y_test, y_pred=y_pred)))

# y_pred = y_pred.numpy()
# y_test = y_test.numpy()
# print(pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}).to_string())
