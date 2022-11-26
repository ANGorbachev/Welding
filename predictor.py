import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

model = keras.models.load_model('./model/best_model.hdf5')


def value_scaler(iw_val, if_val, vw_val, fp_val):
    X_input = np.array([[iw_val, if_val, vw_val, fp_val]])
    scaler = StandardScaler()
    scaler.scale_, scaler.mean_, scaler.var_ = (np.array([1.78815062, 5.86387096, 0.88755868, 24.902588]),
                                                np.array([45.72916667, 141.10416667, 9.4375, 80.83333333]),
                                                np.array([3.19748264, 34.38498264, 0.78776042, 620.13888889]))
    return scaler.transform(X_input).reshape(-1, 4)


def predict_welding(X_input):
    return model.predict(X_input)


if __name__ == "__main__":
    print(predict_welding(value_scaler([43, 146, 9.0, 60])))
