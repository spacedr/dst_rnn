import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as Scaler


def read_data():
    return pd.read_csv('data/omni.csv', index_col='ts', parse_dates=True).interpolate()


def bias(x):
    return np.mean(x['dst_pred']-x['dst'])


def rmse(x):
    return np.sqrt(np.mean((x['dst']-x['dst_pred'])**2))


def corr(x):
    return x.corr()['dst']['dst_pred']


def compute_stats_per_year(z):
    stats = []
    years = set(z.index.year)
    for y in years:
        z1 = z[z.index.year==y]
        stats.append([bias(z1), rmse(z1), corr(z1)])
    return pd.DataFrame(stats, index=years, columns=['BIAS', 'RMSE', 'CORR'])


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
    return pd.concat([data[data.index.year==y] for y in years])


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
    x_train = np.vstack([x[years==year] for year in train_years])
    y_train = np.vstack([y[years==year] for year in train_years])
    x_val = np.vstack([x[years==year] for year in val_years])
    y_val = np.vstack([y[years==year] for year in val_years])
    return x_train, y_train, x_val, y_val