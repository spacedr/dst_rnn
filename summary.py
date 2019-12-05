from util import bias, rmse, corr
from glob import glob
import pandas as pd


def read_model_output(filename):
    return pd.read_csv(filename, index_col='ts', parse_dates=True).rename(columns={'dst_pred': filename.rstrip('.csv')})


def read_model_outputs():
    files = sorted(glob('model*.csv'))
    z = [read_model_output(files[0])]
    for file in files[1:]:
        z.append(read_model_output(file).iloc[:, -1:])
    return pd.concat(z, axis=1)


def compute_stat_per_year(z, stat_fun):
    target_col = 'dst'
    pred_cols = z.columns.drop(target_col)
    years = set(z.index.year)
    result = []
    for year in years:
        x = z[z.index.year == year]
        result.append([stat_fun(x[target_col], x[col]) for col in pred_cols])
    return pd.DataFrame(result, index=years, columns=pred_cols)


if __name__ == '__main__':

    res = read_model_outputs()

    for title, f in [('BIAS', bias), ('RMSE', rmse), ('CORR', corr)]:
        print(title)
        print(compute_stat_per_year(res, f))
        print(20*'-')
