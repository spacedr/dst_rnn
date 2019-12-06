import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as Scaler
import tensorflow as tf


def read_data():
    return pd.read_csv('data/omni.csv', index_col='ts', parse_dates=True).interpolate()


def bias(target, pred):
    return np.mean(pred-target)


def rmse(target, pred):
    return np.sqrt(np.mean((pred-target)**2))


def corr(target, pred):
    return pd.concat([target, pred], axis=1).corr().values[0, 1]


def compute_stats_per_year(target, pred):
    stats = []
    years = set(target.index.year)
    for y in years:
        t = target[target.index.year == y]
        p = pred[pred.index.year == y]
        stats.append([bias(t, p), rmse(t, p), corr(t, p)])
    return pd.DataFrame(stats, index=years, columns=['BIAS', 'RMSE', 'CORR'])


def plot_mse(res):
    return pd.DataFrame(res).plot(logy=True)


def create_model(network, num_in, num_hidden, tau, activation='tanh', learning_rate=1e-3):
    tf.keras.backend.clear_session()
    m = tf.keras.Sequential()
    m.add(tf.keras.layers.Input(shape=(tau, num_in), name='input'))
    m.add(network(num_hidden, activation=activation, unroll=True, name='hidden'))
    m.add(tf.keras.layers.Dense(1, name='output'))
    m.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return m


#################
# Delay operator.
#################

def create_delayed(x, tau):
    assert tau > 0
    m, n = x.shape
    y = np.nan * np.zeros((m, tau, n))  # (number of samples, number of time steps, number of inputs)
    for t in range(tau):
        y[tau - 1:, t] = x[t:m - tau + 1 + t]
    return y


def select_data_for_years(data, years):
    return pd.concat([data[data.index.year == y] for y in years])


def compute_scalers(data, inputs, target):
    return Scaler().fit(data[inputs]), Scaler().fit(data[target])


def create_input(data, inputs, scaler_input, tau):
    return create_delayed(scaler_input.transform(data[inputs]), tau)


def create_input_target(data, inputs, targets, scaler_input, scaler_target, tau, train_years, val_years):
    assert set(train_years).isdisjoint(set(val_years))
    x = create_delayed(scaler_input.transform(data[inputs]), tau)
    y = scaler_target.transform(data[targets])
    i = np.isfinite(x).all(axis=-1).all(axis=-1)
    x = x[i]
    y = y[i]
    years = data.index.year[i]
    x_train = np.vstack([x[years == year] for year in train_years])
    y_train = np.vstack([y[years == year] for year in train_years])
    x_val = np.vstack([x[years == year] for year in val_years])
    y_val = np.vstack([y[years == year] for year in val_years])
    return x_train, y_train, x_val, y_val
