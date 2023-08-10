# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:04:55 2023

@author: saeid
"""

import os.path
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from refnx.analysis import Parameter, Model
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from time import time
import random


import refnx
from refnx.dataset import ReflectDataset, Data1D
from refnx.analysis import Transform, CurveFitter, Objective, Model, Parameter
from refnx.reflect import SLD, Slab, ReflectModel

import keras

import os.path
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from time import time
import random
from keras.models import load_model

import keras

plt.rcParams['figure.dpi'] = 900
plt.rcParams['savefig.dpi'] = 900

#############Force to run on GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.compat.v1.Session(config=config)
'''
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
'''
###########################################

no_q_point=1024
q_values = np.linspace(0.01, 0.6 , no_q_point)

def check_condition(randomized_roughnesses,randomized_thicknesses):
    for i in range(len(randomized_roughnesses)):
        if randomized_roughnesses[i] >= randomized_thicknesses[i]/2:
            return True
    return False

def make_training_input(n_layer):
    min_thickness, max_thickness = [20]*n_layer, [400]*n_layer
    randomized_thicknesses = randomize_inputs(min_thickness, max_thickness)
    min_roughness, max_roughness = [0]*n_layer, [60.0]*n_layer
    randomized_roughnesses = randomize_inputsr(min_roughness, max_roughness)
    min_scattering_length_density_real, max_scattering_length_density_real = [0]*n_layer, [300]*n_layer
    min_scattering_length_density_img, max_scattering_length_density_img = [-20]*n_layer, [0]*n_layer
    while check_condition(randomized_roughnesses,randomized_thicknesses):
        randomized_roughnesses = randomize_inputsr(min_roughness, max_roughness)
    randomized_SLDs_real = randomize_inputs(min_scattering_length_density_real, max_scattering_length_density_real)
    randomized_SLDs_img = randomize_inputs(min_scattering_length_density_img, max_scattering_length_density_img)
    return randomized_thicknesses, randomized_roughnesses, randomized_SLDs_real, randomized_SLDs_img

def randomize_inputsr(min_value, max_value):
    min_value = np.asarray(min_value)
    max_value = np.asarray(max_value)
    number_of_layers = len(min_value)
    randomized_inputs = np.zeros([number_of_layers])
    for layer in range(number_of_layers):
        randomized_inputs[layer] = np.array(tf.random.uniform(shape=(1,),minval=min_value[layer], maxval=max_value[layer],dtype=tf.float64))
    return randomized_inputs

def randomize_inputs(min_value, max_value):
    min_value = np.asarray(min_value)
    max_value = np.asarray(max_value)
    if np.all(np.isreal(min_value)) and np.all(np.isreal(max_value)):
        number_of_layers = len(min_value)
        randomized_inputs = np.zeros([number_of_layers])
        for layer in range(number_of_layers):
            randomized_inputs[layer] = np.array(tf.random.uniform(shape=(1,),minval=min_value[layer], maxval=max_value[layer],dtype=tf.float64))
        return randomized_inputs
    else:
        real_min_value = min_value.real
        real_max_value = max_value.real
        imag_min_value = min_value.imag
        imag_max_value = max_value.imag
        real_randomized_inputs = randomize_inputs(real_min_value, real_max_value,1)
        imag_randomized_inputs = randomize_inputs(imag_min_value, imag_max_value,1)
        complex_randomized_inputs = real_randomized_inputs + 1j * imag_randomized_inputs
        return complex_randomized_inputs

def make_reflectivity_curves(
    q_values, thicknesses, roughnesses, SLDs_real,SLDs_img
):
    air = SLD(0-0j, name='first')
    s= air(0,1)
    structure=s
    for i in range(len(thicknesses)):
        labl=f"Layer_{i}"
        k=SLDs_real[i]+SLDs_img[i]*1j
        w=SLD(k, name=labl)
        s=w(thicknesses[i],roughnesses[i])
        structure=structure|s
    si = SLD(20-0.1j, name='last')
    s = si(0,0)
    structure=structure|s
    model = ReflectModel(structure, bkg=1e-9, dq=1.0)
    reflectivity = model(q_values)
    reflectivity_noisy = apply_shot_noise(reflectivity)
    reflectivity_curve = reflectivity_noisy

    return reflectivity_curve, [[0,thicknesses[0],0],[1,roughnesses[0],0],[0-0j,(SLDs_real+SLDs_img*1j)[0],20-0.1j]]


def make_reflectivity_curves2(
    q_values, thicknesses, roughnesses, SLDs_real,SLDs_img
):
    air = SLD(0-0j, name='first')
    s= air(0,1)
    structure=s
    for i in range(len(thicknesses)):
        labl=f"Layer_{i}"
        k=SLDs_real[i]+SLDs_img[i]*1j
        w=SLD(k, name=labl)
        s=w(thicknesses[i],roughnesses[i])
        structure=structure|s
    si = SLD(20-0.1j, name='last')
    s = si(0,0)
    structure=structure|s
    model = ReflectModel(structure, bkg=1e-9, dq=1.0)
    reflectivity = model(q_values)
    reflectivity_noisy = apply_shot_noise(reflectivity)
    reflectivity_curve = reflectivity_noisy
    if len(reflectivity_noisy) ==1:
        return [-1]
    else:
        return reflectivity_curve, [thicknesses,roughnesses,SLDs_real,SLDs_img]

def apply_shot_noise(reflectivity_curve):
    try:
        result = np.random.poisson(
            reflectivity_curve*1e6
        )*1e-6  # Call the function with the current curve
        noisy_reflectivity = np.clip(result,1e-6,None,)
        return noisy_reflectivity
    except ValueError:
        return [-1]


def make_training_data(n_layers):   #part of generate training data
    Tmp = True
    while Tmp:
        training_data_input = make_training_input(n_layers)
        [thicknesses, roughnesses, SLDs_real, SLDs_img] = training_data_input
        training_reflectivity = make_reflectivity_curves2(q_values, thicknesses, roughnesses, SLDs_real, SLDs_img)
        if len(training_reflectivity)!=1:
            Tmp=False
    return training_reflectivity

def data_generation_w(No_files, No_curves_in_a_file):
    for i in range(No_files):
        data=[]
        for k in range(No_curves_in_a_file):
            Z=make_training_data(2)
            X=Z[0]
            Y=Z[1]
            for j in range(4):
                X=np.append(X, Y[j][0])
            for j in range(4):
                X=np.append(X, Y[j][1])
            if k==0:
                data=[X]
            else:
                data=np.concatenate((data, [X]), axis=0)
            if (k+1)%1000==0 or k==0:
                print (f"{round((k+1)/100,1)}% is done! On {i}-Step!")
        path=f"/home/kowarik/Documents/Saeid/Data2layer/Sim_data_2layer_{i}.csv"
        pd.DataFrame(data).to_csv(path, sep=',',header=None, index=None)


data_generation_w(100,10000)

#########################Model
data_list = []

for i in range(100):
    filename = f"/home/kowarik/Documents/Saeid/Data2layer/Sim_data_2layer_{i}.csv"
    data = np.loadtxt(filename, delimiter=",")
    data_list.append(data)
    print(i)
    
concatenated_data = np.concatenate(data_list, axis=0)
dat=pd.DataFrame(concatenated_data)
X = dat.iloc[:,:].values
############Clean Incorrect simulated data
W = []
for i in range(len(X)):
    if np.max(X[i,:1024])<1.0:
        W.append(X[i,:])
X = np.array(W)
##############
#mi=[]
#ma=[]
#for i in range(len(X[0])-8):
#    mi.append(X[:,i].min())
#    ma.append(X[:,i].max())
#for i in range(len(X[0])-8):
#    X[:,i]=(X[:,i]-X[:,i].min())/(X[:,i].max()-X[:,i].min())


data1=pd.DataFrame(X)

#path = 'C:/Users/saeid/Desktop/Python code/XR/Sim_data_1024M_2layer.csv'
#pd.DataFrame(concatenated_data).to_csv(path, sep=',',header=None, index=None)

#dat = pd.read_csv(path,header=None)

model = Sequential()
model.add(Dense(no_q_point, input_dim=(no_q_point), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))

train, validate, test = np.split(data1.sample(frac=1, random_state=42),[int(.8*len(data1)), int(.9*len(data1))])
X_train=train.iloc[:,:-8].values
y_train=train.iloc[:,-8:].values
y_train[:,-1]=y_train[:,-1]*-1
y_train[:,-5]=y_train[:,-5]*-1
X_valid=validate.iloc[:,:-8].values
y_valid=validate.iloc[:,-8:].values
y_valid[:,-1]=y_valid[:,-1]*-1
y_valid[:,-5]=y_valid[:,-5]*-1
X_test=test.iloc[:,:-8].values
y_test=test.iloc[:,-8:].values
y_test[:,-1]=y_test[:,-1]*-1
y_test[:,-5]=y_test[:,-5]*-1


opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt,loss="mse",metrics="mae")
model=tf.keras.models.load_model('/home/kowarik/Documents/Saeid/Data2layer/Models/NN_model/trained_model_all_parameters_1024_2layer.h5')

maxepochs=1000
history=model.fit(X_train,y_train,epochs=maxepochs,validation_data=(X_valid,y_valid),batch_size=64)
filename="trained_model_all_parameters_1024_2layer.h5"
model.save('/home/kowarik/Documents/Saeid/Data2layer/Models/NN_model/'+filename)
#################Autoencoder
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

data_list = []

for i in range(100):
    filename = f"/home/kowarik/Documents/Saeid/Data2layer/Sim_data_2layer_{i}.csv"
    data = np.loadtxt(filename, delimiter=",")
    data_list.append(data)
    print(i)
    
concatenated_data = np.concatenate(data_list, axis=0)
dat=pd.DataFrame(concatenated_data)
X = dat.iloc[:,:].values

############Clean Incorrect simulated data
W = []
for i in range(len(X)):
    if np.max(X[i,:1024])<1.0:
        W.append(X[i,:])
X = np.array(W)
##############

#mi=[]
#ma=[]
#for i in range(len(X[0])-8):
#    mi.append(X[:,i].min())
#    ma.append(X[:,i].max())
#for i in range(len(X[0])-8):
#    X[:,i]=(X[:,i]-X[:,i].min())/(X[:,i].max()-X[:,i].min())


data1=pd.DataFrame(X[:,:-8])
X_train, X_validate, X_test = np.split(data1.sample(frac=1, random_state=42),[int(.8*len(data1)), int(.9*len(data1))])

input_layer = tf.keras.layers.Input(shape=(no_q_point,))
hidden_layer = tf.keras.layers.Dense(512, activation='relu')(input_layer)
hidden_layer = tf.keras.layers.Dense(256, activation='relu')(input_layer)
hidden_layer = tf.keras.layers.Dense(128, activation='relu')(hidden_layer)
hidden_layer = tf.keras.layers.Dense(64, activation='relu')(hidden_layer)
hidden_layer = tf.keras.layers.Dense(32, activation='relu')(hidden_layer)
hidden_layer = tf.keras.layers.Dense(16, activation='relu')(hidden_layer)
hidden_layer = tf.keras.layers.Dense(32, activation='relu')(hidden_layer)
hidden_layer = tf.keras.layers.Dense(64, activation='relu')(hidden_layer)
output_layer = tf.keras.layers.Dense(128, activation='linear')(hidden_layer)
output_layer = tf.keras.layers.Dense(256, activation='linear')(hidden_layer)
output_layer = tf.keras.layers.Dense(512, activation='linear')(hidden_layer)
output_layer = tf.keras.layers.Dense(no_q_point, activation='linear')(hidden_layer)
autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder=tf.keras.models.load_model('/home/kowarik/Documents/Saeid/Data2layer/Models/AutoEncoder/Autoencoder_fit_1024_2layer.h5')
epochs = 100
batch_size = 32
autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_validate, X_validate))

filename="Autoencoder_fit_1024_2layer.h5"
autoencoder.save('/home/kowarik/Documents/Saeid/Data2layer/Models/AutoEncoder/'+filename)

from sklearn.ensemble import RandomForestRegressor
from random import sample
import joblib

# Define the RandomForestRegressor model with hyperparameters
model = RandomForestRegressor(n_estimators=500, max_depth=16, random_state=42, n_jobs=12)

sample1 = sample(range(len(X)),int(len(X)/2))
X_train = data1.iloc[sample1,:-8].values
y_train = data1.iloc[sample1,-8:].values
model.fit(X_train, y_train)


# Save the trained model to disk

joblib.dump(model, '/home/kowarik/Documents/Saeid/Data2layer/Models/RandomForestReg/random_forest_regressor_500t_2layers.joblib')

