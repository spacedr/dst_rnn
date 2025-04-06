# Exploring prediction models for Dst

In this tutorial we will explore different neural networks for the prediction of the Dst index. This problem has been 
addressed in numerous studies and therefore serves as a good example.

The Dst index is a measure of the strength of the ring current, a current typically located around the Earth equatorial
plane at about 3 to 8 Earth radii. The current causes a weakening of the magnetic field at the Earth surface. The Dst 
index is computed from magnetic field measurements at four near-equatorial locations. When the ring current increases in
strength the Dst index becomes more negative. The main driver for the strengthening of the ring current is the transfer
of energy from the solar wind into Earth's magnetosphere, and the rate of energy transfer is dependent on the conditions
in the solar wind. A key parameter is the direction of the solar wind magnetic field, when it points southward the
energy transfer is more efficient. Another process, that is not related to the ring current, that also affects the Dst 
index is the compression of the day-side magnetosphere which causes an increase in Dst. When the Dst index becomes
strongly negative we label the event as a geomagnetic storm. After the peak of the storm the Dst index decays back
towards zero with a time constant of 10-20 hours.

We will study models that take solar wind data as input and predicts the Dst index, thus past Dst values will **not** be
used as inputs. For this purpose we will use solar wind data and Dst index from the OMNI (https://omniweb.gsfc.nasa.gov)
low resolution (hourly) database.

We will explore three types of neural networks using the TensorFlow (https://www.tensorflow.org) package: SimpleRNN,
GRU, and LSTM. We refer to the TensorFlow documentation for a description of the networks.

## Setup

The code in this tutorial uses Python 3.12. Several packages also needs to be installed, like TensorFlow and Pandas. The
file `requirements.txt` lists all required packages. They can easily be installed using pip

    pip install -r requirements.txt

It is also highly recommended to use IPython (https://ipython.org) for any interactive work. It is listed in the
requirements file and will be install with the pip command above. When IPython is launched it will display something
like

    $ ipython
    Python 3.12.1 (main, Dec 16 2023, 15:11:56) [Clang 15.0.0 (clang-1500.1.0.2.5)]
    Type 'copyright', 'credits' or 'license' for more information
    IPython 9.0.2 -- An enhanced Interactive Python. Type '?' for help.
    Tip: You can use latex or unicode completion, `\alpha<tab>` will insert the α symbol.
    
    In [1]: 

## Data set

OMNI hourly solar wind data and Dst index for the years 1998, 1999, 2000, 2001, and 2002 have been collected and stored
into the CSV file at data/omni.csv. This period contains many large storms. There are also very few data gaps: 26 hours
in solar wind magnetic field; 469 hours in density; 84 hours in speed. To simplify the analysis the data gaps are filled
in using linear interpolation.

The `read_data` function in the `util.py` module reads the data into a Pandas DataFrame:

    In [1]: from util import read_data                                                                                                                            
    
    In [2]: data = read_data()                                                                                                                                    
    
    In [3]: data.head()                                                                                                                                           
    Out[3]: 
                                 b   bz    n      v   f10  dst
    ts                                                        
    1998-01-01 00:00:00+00:00  3.1  2.2  7.7  366.0  98.3   -9
    1998-01-01 01:00:00+00:00  4.1  3.3  8.3  367.0  98.3   -8
    1998-01-01 02:00:00+00:00  4.4 -0.9  8.2  359.0  98.3   -9
    1998-01-01 03:00:00+00:00  4.7 -1.1  8.8  364.0  98.3   -9
    1998-01-01 04:00:00+00:00  5.4 -1.2  8.5  362.0  98.3  -13
    
Here we see the timestamps at a 1 hour cadence with the solar wind magnetic field strength `b` and z-component `bz` in
nT, the solar wind proton density `n` (cm^-3), and solar wind speed `v` (km/s). The last column (`dst`) is the Dst index
(nT). We also include the daily solar F10.7 flux.

## Data pre-processing

Before the data can be used for training the network a couple of data processing steps must be applied.


### Normalisation

The first step is the normalisation of the data so that the numerical ranges of each parameter are similar. We use the 
scaling classes available in the `sklearn.preprocessing` module. The `StandardScaler` class normalises the data
according to

    x_norm = (x - m) / s
    
where `s` is the standard deviation and `m` is the mean of the variable `x`. For example

    In [1]: from util import read_data, Scaler                                                                                                                    
    
    In [2]: data = read_data()                                                                                                                                    
    
    In [3]: s = Scaler().fit(data)                                                                                                                                
    
    In [4]: s.scale_                                                                                                                                              
    Out[4]: 
    array([ 3.38097518,  3.72765003,  5.41132324, 92.05810749, 39.56462204,
           26.51648161])
    
    In [5]: s.mean_                                                                                                                                               
    Out[5]: 
    array([ 6.41241329e+00, -3.21182457e-02,  6.42924311e+00,  4.32507085e+02,
            1.62516320e+02, -1.75803213e+01])
    
    In [6]: s.transform(data)                                                                                                                                     
    Out[6]: 
    array([[-0.97972127,  0.59880038,  0.23483293, -0.7224468 , -1.62307426,
             0.32358446],
           [-0.68394861,  0.89389246,  0.34571154, -0.7115841 , -1.62307426,
             0.36129685],
           [-0.59521682, -0.23282276,  0.32723177, -0.79848573, -1.62307426,
             0.32358446],
           ...,
           [-0.03324878, -0.04503689, -0.31956012, -0.17931158, -1.29449789,
            -0.12896427],
           [ 0.11463755, -0.01821033, -0.33803989, -0.10327265, -1.29449789,
            -0.09125188],
           [-0.06282604,  0.8134128 , -0.19020174, -0.24448781, -1.29449789,
             0.0595977 ]])

The `scale_` and `mean_` correspond to the standard deviations and means for each of the six data columns, respectively.
The scaling objects are determined separately for the input data (solar wind) and target data (Dst) in the function
`compute_scalers`.

### Tap delay line

The normal approach to train a recurrent neural network (RNN) with TensorFlow/Keras is to transform the input data using
tap delay lines (TDL). This is exactly the same approach taken with non-recurrent time delayed neural networks (TDNN),
but the two network types are different. In the TensorFlow documentation (for example
https://www.tensorflow.org/guide/keras/rnn) the tap-delay-line axis is called *timesteps*. In principle, the RNN has
unlimited memory but in practice the maximum trainable memory-length is limited by the length of the TDL. The
`create_delayed` function in the `util.py` module creates a TDL from a matrix.

    In [1]: from util import create_delayed                                                                                                                                                                                     
    
    In [2]: import numpy as np                                                                                                                                                                                                  
    
    In [3]: x = np.array([[0, 1], [2, 3], [4, 5], [6, 7]]) # 4 samples, 2 inputs                                                                                                                                                                     
    
    In [4]: x.shape                                                                                                                                                                                                             
    Out[4]: (4, 2)
    
    In [5]: y = create_delayed(x, 3) # Creates 4 samples with 2 inputs with 3 tapped delays                                                                                                                                                                                           
    
    In [6]: y.shape                                                                                                                                                                                                             
    Out[6]: (4, 3, 2)
    
    In [7]: y                                                                                                                                                                                                                   
    Out[7]: 
    array([[[nan, nan],
            [nan, nan],
            [nan, nan]],
    
           [[nan, nan],
            [nan, nan],
            [nan, nan]],
    
           [[ 0.,  1.],
            [ 2.,  3.],
            [ 4.,  5.]],
    
           [[ 2.,  3.],
            [ 4.,  5.],
            [ 6.,  7.]]])

### Delay line length

The length of the TDL sets a limit to how far back in time the RNN can model correlated structures. If the length is
shorter than the memory of the system modelled the RNN will fail, although the RNN in itself does not impose such a
limit, it is only an effect of how the data is presented. The `memory.py` program can be used to study the effects of
different TDL lengths. The input data series contain pulses (+1) at random times. The output series will also turn
positive when the input is positive and stay positive for `TAU_DATA` time steps. The length of the TDL is `TAU_MODEL`.
Running the program will train the RNN and show some samples of input-target pairs, show the training MSE, and target
and predicted values. Setting `TAU_MODEL < TAU_DATA` shows what happens when the memory of the model is too short. 

## Models

### Baseline model (model001.py)

It is always useful to have a baseline model against which the neural networks can be compared. Here we will use the AK1
model by O'Brien and McPherron (https://www.sciencedirect.com/science/article/pii/S1364682600000729). The model is
implemented in the file `model001.py` and can be run as

    python model001.py  # Note: assumes Python 3.

This should open a window with target Dst and predicted Dst. Summary statistics should also be printed on the console.
Alternatively an IPython session could be started and the program be run

    In [1]: %matplotlib                                                                                                                                                                              
    Using matplotlib backend: MacOSX

    In [2]: %run model001.py                                                                                                                                                                         
              BIAS       RMSE      CORR
    1998 -7.718673  14.615768  0.861214
    1999 -9.765002  16.503797  0.796858
    2000 -8.195001  16.997045  0.849961
    2001 -7.358677  19.238412  0.836700
    2002 -1.521446  14.104872  0.832688

As we will use the data before 2001 for training the neural networks we may apply a bias correction to the model using
the computed bias for the years 1998-2000: 8.6. This will improve RMSE but not CORR.

### Elman network (model002.py)

We will implement the simplest kind of recurrent network, also known as the Elman network or SimpleRNN in TensorFlow.
This was the RNN implemented in the paper by Lundstedt et al. 2002 
(https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2002GL016151).

![Dst Elman network](https://github.com/spacedr/dst_rnn/blob/master/grl16387-fig-0001.png)

The network takes the solar wind Bz, density n, and speed V as inputs and predicts Dst one hour ahead. The network has
only 4 hidden units and uses the tanh activation function.

In this implementation we will actually predict Dst with **no** lead time, that is, with latest solar wind input at time
*t* we will predict Dst at time *t*. The reason we do this is that the pressure related increase in Dst (initial phase)
is directly controlled by the solar wind with no lead time. This is also how the baseline model (model001) is
implemented.

We set the maximum memory to 48 hours (`tau = 48`). Data for year 2000 are used for training and 2002 for validation.
Running the model in an IPython session will print the training progress and summary statistics similar to below:

    In [1]: %matplotlib                                                                                                                                                                                                         
    Using matplotlib backend: MacOSX
    
    In [2]: %run model002.py                                                                                                                                                                                                    
    2019-12-06 16:35:00.312605: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
    2019-12-06 16:35:00.344495: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x1407625d0 executing computations on platform Host. Devices:
    2019-12-06 16:35:00.344515: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
    Train on 8784 samples, validate on 8760 samples
    Epoch 1/1000
    8784/8784 [==============================] - 2s 257us/sample - loss: 0.8866 - val_loss: 0.6981
    Epoch 2/1000
    8784/8784 [==============================] - 0s 9us/sample - loss: 0.8686 - val_loss: 0.6785

    ...
    
    Epoch 999/1000
    8784/8784 [==============================] - 0s 7us/sample - loss: 0.1441 - val_loss: 0.2960
    Epoch 1000/1000
    8784/8784 [==============================] - 0s 7us/sample - loss: 0.1441 - val_loss: 0.2931
    Final model:
              BIAS       RMSE      CORR
    1998  3.565568  13.560642  0.875774
    1999 -0.512858  12.105195  0.836200
    2000  0.254705  10.670174  0.924413
    2001  1.777083  15.659932  0.877618
    2002  7.376343  15.142497  0.853722
    Best model:
              BIAS       RMSE      CORR
    1998  0.701294  13.980712  0.867888
    1999 -2.858034  12.585170  0.831248
    2000 -2.174544  10.990695  0.922874
    2001 -0.858191  15.329488  0.881886
    2002  4.860753  14.180491  0.850721

Three plots will also appear showing the training and validation loss (MSE), the predictions using the final network,
and the predictions using the network with the smallest validation loss. A model checkpoint object

    tf.keras.callbacks.ModelCheckpoint(model_filename, save_best_only=True)
    
is used to store the best (smallest validation MSE) to file. The best network may be different from the final network.

Files `model003.py` and `model004.py` are variations on the Elman network (`model002.py`) but with more training data
and longer memory, respectively.

## Further experimentation

### A very simple RNN

In `model005.py` a very simple RNN consisting of only one linear unit is used to demonstrate the connection to the Dst
differential equation previously discussed. The RNN can be expressed as

$$y_{t+1} = a x_t + b y_t + c$$

were $x_t$ and $y_t$ are the input and output, respectively, at at time step $t$. During training the input weight $a$, the recurrent weight $b$, and the bias $c$ are adjusted to minimise loss. The `use_bias` argument to `SimpleRNN` in [model005.py](https://github.com/spacedr/dst_rnn/blob/6eb135706c217000281bd4727439825bf691f6ea/model005.py#L16) control whether the bias $c$ is included or not. This equation can be compared with the differential equation from by O'Brien and McPherron (https://www.sciencedirect.com/science/article/pii/S1364682600000729)

<<<<<<< HEAD
$$
\frac{d \mathrm{Dst}^*(t)}{dt} = Q(t) - \frac{\mathrm{Dst}^*(t)}{\tau}
$$
=======
$\frac{d \mathrm{Dst}^*(t)}{dt} = Q(t) - \frac{\mathrm{Dst}^*(t)}{\tau}$
>>>>>>> 1cf4106 (fix equation display on github)

were $\mathrm{Dst}^*$ is the pressure corrected $\mathrm{Dst}$, $Q$ the energy injection rate, and $\tau$ the ring current decay time. Here we assume that $Q$ is proportional to the dawn-dusk solar wind electric field $Q = q V B_s$, were $q$ is an empirical coefficient, $V$ is speed, and $B_s = 0$ when $B_z > 0$, and $B_s = -B_z$ when $B_z < 0$. Rewriting the differential equation into discrete form and setting the timestep $dt \approx \Delta t = 1$ hour we get

$$\mathrm{Dst}^*(t+1) = Q(t) + \left( 1 - \frac{1}{\tau} \right) \mathrm{Dst}^*(t)$$

which is equivalent to the RNN with $c = 0$.

To be able to interpret the weights $a$ and $b$ we must take into account the normalisation that have been applied, i.e

$$x = \frac{V B_s - m_x}{s_x}$$

and

$$y = \frac{\mathrm{Dst}^* - m_y}{s_y}$$

Inserting this into the RNN equation without the bias $c$, and after rearranging the terms, we get

$$\mathrm{Dst}^*(t+1) = a \frac{s_y}{s_x} V(t) B_s(t) + b \mathrm{Dst}^*(t) - a \frac{s_y}{s_x} m_x - b m_y$$

We now see that the coefficient in the differential equation can be found as

$$q = a \frac{s_y}{s_x}$$

and

$$\left( 1 - \frac{1}{\tau} \right) = b$$

After training one should obtain $q \approx -2.5 \; \mathrm{km}^{-1}$ and $\tau \approx 15 \; \mathrm{hours}$.

### GRU and LSTM

The previous Elman (SimpleRNN) models (files) can be easily copied and edited to implement the GRU and LSTM networks.

## Summary

The above RNNs illustrate how to set up the models using Keras and Tensorflow. The datasets and models are kept small in
order to make the training/validation in a short time. The various hyper-parameters have also been set to some
reasonable values. However, to achieve really good accuracy more data should be used and a thorough hyper-parameter
search should be done.
