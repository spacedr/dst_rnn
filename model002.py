from util import read_data, select_data_for_years, compute_scalers, create_input, create_input_target, compute_stats_per_year
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


def create_model(tau, ni, nh):
    tf.keras.backend.clear_session()
    m = tf.keras.Sequential()
    m.add(tf.keras.layers.Input(shape=(tau, ni), name='input'))
    m.add(tf.keras.layers.SimpleRNN(nh, activation='tanh', name='hidden'))
    m.add(tf.keras.layers.Dense(1, name='output'))
    m.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    return m


if __name__ == '__main__':

    model_filename = 'model002.hdf5'
    output_filename = 'model002.csv'

    inputs = ['bz', 'n', 'v']
    target = ['dst']

    train_years, val_years, test_years = [2000], [2002], [2001]

    num_hidden = 4
    tau = 48

    data = read_data()
    scaler_input, scaler_target = compute_scalers(select_data_for_years(data, train_years), inputs, target)
    x_train, y_train, x_val, y_val = create_input_target(data,
                                                         inputs, target,
                                                         scaler_input, scaler_target,
                                                         tau,
                                                         train_years, val_years)

    model = create_model(tau, x_train.shape[-1], num_hidden)

    res = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    callbacks=[tf.keras.callbacks.ModelCheckpoint(model_filename, save_best_only=True)],
                    batch_size=1000,
                    epochs=1000)
    pd.DataFrame(res.history).plot(logy=True)

    model_best = tf.keras.models.load_model(model_filename)

    x = create_input(data, inputs, scaler_input, tau)

    z = pd.concat([data['dst'],
                   pd.Series(scaler_target.inverse_transform(model.predict(x)).ravel(),
                             index=data.index, name='dst_pred')], axis=1)

    z_best = pd.concat([data['dst'],
                        pd.Series(scaler_target.inverse_transform(model_best.predict(x)).ravel(),
                                  index=data.index, name='dst_pred')], axis=1)
    z_best.to_csv(output_filename)

    print('Final model:')
    print(compute_stats_per_year(z['dst'], z['dst_pred']))

    print('Best model:')
    print(compute_stats_per_year(z['dst'], z['dst_pred']))

    z.plot(title='Prediction using final model')

    z_best.plot(title='Prediction using best model')

    plt.show()
