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

The code in this tutorial uses Python 3.6. Several packages also needs to be installed, like TensorFlow and Pandas. The
file `requirements.txt` lists all required packages. They can easily be installed using pip

    pip install -r requirements.txt

It is also highly recommended to use IPython (https://ipython.org) for any interactive work. It is listed in the
requirements file and will be install with the pip command above. When IPython is launched it will display something
like

    $ ipython
    Python 3.6.5 (v3.6.5:f59c0932b4, Mar 28 2018, 03:03:55) 
    Type 'copyright', 'credits' or 'license' for more information
    IPython 7.10.1 -- An enhanced Interactive Python. Type '?' for help.

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

## Models

### Baseline model

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

### Elman network

### GRU network

### LSTM network

### Simple network simulating the AK1 model.

## Summary