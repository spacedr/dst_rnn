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


def obrien_ak1(x):
    p = pressure(x)
    v = vbs(x)
    dst_star = np.zeros(len(x) + 1)
    for i in range(len(x)):
        dst0 = dst_star[i]
        if np.isnan(dst0):
            dst0 = 0
        dst_star[i + 1] = dst0 - 2.47*v.iloc[i] - dst0/17
    dst_star = pd.Series(dst_star[1:], x.index)
    return (dst_star + 8.74*np.sqrt(p) - 11.5).rename('dst_pred')


if __name__ == '__main__':

    data = read_data()
    z = pd.concat([data['dst'], obrien_ak1(data)], axis=1)
    z.to_csv('model001.csv')

    print(compute_stats_per_year(z['dst'], z['dst_pred']))

    z.plot()

    plt.show()
