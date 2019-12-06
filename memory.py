from util import create_delayed, plot_mse, create_model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def create_input(n, m):
    x = -1 * np.ones((n, 1))
    x[np.random.permutation(n)[:m]] = 1
    return x


def create_target(x, tau):
    y = -1 * np.ones_like(x)
    for i in range(len(y)):
        if np.any(x[max(0, i - tau + 1):i+1] > 0):
            y[i] = 1
    return y


if __name__ == '__main__':

    # Create input and target data.
    TAU_DATA = 5    # The memory of the process generating the data.
    TAU_MODEL = 10  # The memory of the model.
    X_series = create_input(1000, 50)
    X_series[:TAU_DATA] = -1
    Y = create_target(X_series, TAU_DATA)
    X = create_delayed(X_series, TAU_MODEL)
    X[np.isnan(X).any(axis=-1).any(axis=-1)] = -1   # The first records will be nan, given by TAU_MODEL.
    # Y[np.isnan(X).any(axis=-1).any(axis=-1)] = -1   # Also set target to -1 for those records.

    M = create_model(tf.keras.layers.SimpleRNN, 1, 5, TAU_MODEL, learning_rate=1e-2)
    R = M.fit(X, Y, batch_size=100, epochs=200)
    Z = M.predict(X)

    print('Shapes of time series:')
    print('\tX_series.shape = {}'.format(X_series.shape))
    print('\tX.shape = {}'.format(X.shape))
    print('\tY.shape = {}'.format(Y.shape))

    # Find first input sample with input > 0 and plot a few input samples and targets.
    i = np.where(X[:, -1, 0] > 0)[0][0]
    js = [-1, 0, 1, 2]
    f, axs = plt.subplots(len(js), 1, sharex=True)
    for j, ax in zip(js, axs):
        ax.plot(X[i+j], 's-', label='target = {}'.format(Y[i+j]))
        ax.set_ylim([-1.1, 1.1])
        ax.legend(loc='upper left')

    plot_mse(R.history)

    plt.figure()
    plt.plot(Y)
    plt.plot(Z)

    plt.show()