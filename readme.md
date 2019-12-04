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

## Data set

OMNI hourly solar wind data and Dst index for the years 1998, 1999, 2000, 2001, and 2002 have been collected and stored
into the CSV file at data/omni.csv. This period contains many large storms. There are also very few data gaps: 26 hours
in solar wind magnetic field; 469 hours in density; 84 hours in speed. To simplify the analysis the data gaps are filled
in using linear interpolation.

## Models

### Baseline model

It is always useful to have a baseline model against which the neural networks can be compared. Here we will use the AK1
model by O'Brien and McPherron (https://www.sciencedirect.com/science/article/pii/S1364682600000729). The model is
implemented in the file `model001.py` and can be run as

    python model001.py  # Note: assumes Python 3.

This should open a window with target Dst and predicted Dst. Summary statistics should also be printed on the console.
Alternatively an IPython session could be started and the program be run

    $ ipython
    Python 3.6.5 (v3.6.5:f59c0932b4, Mar 28 2018, 03:03:55) 
    Type 'copyright', 'credits' or 'license' for more information
    IPython 7.10.1 -- An enhanced Interactive Python. Type '?' for help.

    In [1]: %matplotlib                                                                                                                                                                              
    Using matplotlib backend: MacOSX

    In [2]: %run model001.py                                                                                                                                                                         
              BIAS       RMSE      CORR
    1998 -7.718673  14.615768  0.861214
    1999 -9.765002  16.503797  0.796858
    2000 -8.195001  16.997045  0.849961
    2001 -7.358677  19.238412  0.836700
    2002 -1.521446  14.104872  0.832688

### Data pre-processing

Preparing data for recurrent networks.

Normalization.

### Elman network

### GRU network

### LSTM network

### Simple network simulating the AK1 model.

## Summary