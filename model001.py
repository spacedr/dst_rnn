from util import read_data, compute_stats_per_year
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import m_p


def pressure(x):
    return (m_p * x['n'] * 1e6 * (x['v'] * 1e3) ** 2 * 1e9).rename('p')


def vbs(x):
    v_bs = -x['bz']*x['v']*1e-3
    v_bs[v_bs < 0] = 0
    return v_bs.rename('vbs')


def dst_star(q, tau):
    if np.isscalar(tau):
        tau = tau * np.ones_like(q)
    dst_s = np.zeros(len(q))
    dst0 = 0
    for i in range(len(q)):
        if np.isnan(dst0):
            dst0 = 0
        dst_s[i] = dst0 + q[i] - dst0 / tau[i]
        dst0 = dst_s[i]
    return pd.Series(dst_s, q.index).rename('dst_star')


def obrien_ak1(x, bias=0.0):
    q = -2.47 * vbs(x)
    tau = 17.0
    b = 8.74
    c = 11.5
    return (dst_star(q, tau) + b * np.sqrt(pressure(x)) - c + bias).rename('dst_pred')


if __name__ == '__main__':

    BIAS_CORRECTION = 0.0  # Change e.g. to 8.6.

    DATA = read_data()
    Z = pd.concat([DATA['dst'], obrien_ak1(DATA, BIAS_CORRECTION)], axis=1)
    Z.to_csv('model001.csv')

    print(compute_stats_per_year(Z['dst'], Z['dst_pred']))

    Z.plot()

    plt.show()
