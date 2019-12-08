from util import read_data, select_data_for_years, compute_scalers, create_input, create_input_target, \
    compute_stats_per_year
from model001 import pressure, vbs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def create_model(tau, learning_rate=1e-3):
    tf.keras.backend.clear_session()
    m = tf.keras.Sequential()
    m.add(tf.keras.layers.Input(shape=(tau, 1), name='input'))
    m.add(tf.keras.layers.SimpleRNN(1, activation='linear', unroll=True, name='output'))
    m.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return m


if __name__ == '__main__':

    model_filename = 'model006.hdf5'
    output_filename = 'model006.csv'

    inputs = ['vbs']
    target = ['dst_star']

    train_years, val_years, test_years = [1998, 1999, 2000], [2002], [2001]

    tau = 48

    # First create dst_star by correcting for sqrt_pressure using coefficient from O'Brien.
    # Then train on dst_star.

    data = read_data()
    data['sqrtp'] = np.sqrt(pressure(data))
    data['vbs'] = vbs(data)
    data['dst_star'] = data['dst'] - 8.74 * data['sqrtp']

    scaler_input, scaler_target = compute_scalers(select_data_for_years(data, train_years), inputs, target)
    x_train, y_train, x_val, y_val = create_input_target(data,
                                                         inputs, target,
                                                         scaler_input, scaler_target,
                                                         tau,
                                                         train_years, val_years)

    model = create_model(tau, 1e-2)

    res = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    callbacks=[tf.keras.callbacks.ModelCheckpoint(model_filename, save_best_only=True)],
                    batch_size=1000,
                    epochs=500)
    pd.DataFrame(res.history).plot(logy=True)

    model_best = tf.keras.models.load_model(model_filename)

    x = create_input(data, inputs, scaler_input, tau)

    z = pd.concat([data['dst'],
                   pd.Series(scaler_target.inverse_transform(model.predict(x)).ravel() + 8.74 * data['sqrtp'],
                             index=data.index, name='dst_pred')], axis=1)

    z_best = pd.concat([data['dst'],
                        pd.Series(scaler_target.inverse_transform(model_best.predict(x)).ravel() + 8.74 * data['sqrtp'],
                                  index=data.index, name='dst_pred')], axis=1)
    z_best.to_csv(output_filename)

    print('Final model:')
    print(compute_stats_per_year(z['dst'], z['dst_pred']))

    print('Best model:')
    print(compute_stats_per_year(z_best['dst'], z_best['dst_pred']))

    print('Recurrent weight gives tau = {}'.format(1 / (1 - model_best.get_weights()[1][0, 0])))
    print('Re-normalized input weights = {}'.format(model_best.get_weights()[0][:, 0] * scaler_target.scale_[0] / scaler_input.scale_ ))

    z.plot(title='Prediction using final model')

    z_best.plot(title='Prediction using best model')

    plt.show()
