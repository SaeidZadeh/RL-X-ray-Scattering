# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 10:34:08 2023

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
from scipy.optimize import minimize

import refnx
from refnx.dataset import ReflectDataset, Data1D
from refnx.analysis import Transform, CurveFitter, Objective, Model, Parameter
from refnx.reflect import SLD, Slab, ReflectModel

import keras

plt.rcParams['figure.dpi'] = 900
plt.rcParams['savefig.dpi'] = 900


q_values = np.linspace(0.01, 0.6 , 1024)

import tensorflow as tf
import numpy as np
from random import sample
import pandas as pd
import matplotlib.pyplot as plt
import numbers

import joblib
import pickle


modelf = joblib.load('/home/kowarik/Documents/Saeid/Data2layer/Models/RandomForestReg/random_forest_regressor_500t_2layers.joblib')

autoencoder_model=tf.keras.models.load_model('/home/kowarik/Documents/Saeid/Data2layer/Models/AutoEncoder/Autoencoder_fit_1024_2layer.h5')
parameters_model=tf.keras.models.load_model('/home/kowarik/Documents/Saeid/Data2layer/Models/NN_model/trained_model_all_parameters_1024_2layer.h5')



def Fit_Thick(X,q_values,Z):
    result=[]
    non_zero_idx=np.where(X!=0)
    Y=np.concatenate(([q_values[non_zero_idx]], [X[non_zero_idx]]), axis=0)
    air = SLD(0-0j, name='first')
    s= air(0,1)
    structure=s
    labl="Layer1"
    tmp_var=1j
    k=Z[-2]+Z[-1]*tmp_var
    w=SLD(k, name=labl)
    midl_layer1=w(Z[-4],Z[-3])
    structure=structure|midl_layer1
    labl="Layer2"
    tmp_var=1j
    k=Z[-6]+Z[-5]*tmp_var
    w=SLD(k, name=labl)
    midl_layer2=w(Z[-8],Z[-7])
    structure=structure|midl_layer2
    
    si = SLD(20-0.1j, name='last')
    s = si(0,0)
    structure=structure|s
    model = ReflectModel(structure, bkg=3e-9, dq=0)
    midl_layer1.sld.real.setp(bounds=(1e-3,300), vary=True)
    midl_layer1.sld.imag.setp(bounds=(-20, -1e-3), vary=True)
    midl_layer1.thick.setp(bounds=(20,400), vary=True)
    midl_layer1.rough.setp(bounds=(1e-3,60), vary=True)
    midl_layer2.sld.real.setp(bounds=(1e-3,300), vary=True)
    midl_layer2.sld.imag.setp(bounds=(-20, -1e-3), vary=True)
    midl_layer2.thick.setp(bounds=(20,400), vary=True)
    midl_layer2.rough.setp(bounds=(1e-3,60), vary=True)
    model.bkg.setp(bounds=(1e-9, 9e-6), vary=True)
    #model.dq.setp(bounds=(1e-3,5),vary=True)
    objective = Objective(model, Y, transform=Transform('logY'))

    fitter = CurveFitter(objective)
    fitter.fit('differential_evolution');
    result.append(objective.parameters['Structure - ']['Layer1']['Layer1 - thick'].value)
    result.append(objective.parameters['Structure - ']['Layer1']['Layer1 - rough'].value)
    result.append(objective.parameters['Structure - ']['Layer1']['Layer1']['Layer1 - sld'].value)
    result.append(objective.parameters['Structure - ']['Layer1']['Layer1']['Layer1 - isld'].value)
    result.append(objective.parameters['Structure - ']['Layer2']['Layer2 - thick'].value)
    result.append(objective.parameters['Structure - ']['Layer2']['Layer2 - rough'].value)
    result.append(objective.parameters['Structure - ']['Layer2']['Layer2']['Layer2 - sld'].value)
    result.append(objective.parameters['Structure - ']['Layer2']['Layer2']['Layer2 - isld'].value)
    return result

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
    if isinstance(thicknesses, numbers.Number):
        thicknesses, roughnesses, SLDs_real,SLDs_img = [thicknesses], [roughnesses], [SLDs_real], [SLDs_img]
    for i in range(len(thicknesses)):
        labl=f"Layer_{i}"
        k=SLDs_real[i]-SLDs_img[i]*1j
        w=SLD(k, name=labl)
        s=w(thicknesses[i],roughnesses[i])
        structure=structure|s
    si = SLD(20-0.1j, name='last')
    s = si(0,0)
    structure=structure|s
    model = ReflectModel(structure, bkg=1e-7, dq=0.0)
    reflectivity = model(q_values)
    if np.count_nonzero(np.isnan(reflectivity)) < len(reflectivity) and len(np.isnan(reflectivity)) > 0:
        reflectivity[np.isnan(reflectivity)] = next(x for x in reflectivity if not np.isnan(x))
    if np.count_nonzero(np.isnan(reflectivity)) == len(reflectivity):
        reflectivity[np.isnan(reflectivity)] = (q_values-0.6)/0.59
    #reflectivity_noisy = apply_shot_noise(reflectivity)
    reflectivity_curve = reflectivity#_noisy

    return reflectivity_curve


def apply_shot_noise(reflectivity_curve):
    noisy_reflectivity = np.clip(
        np.random.poisson(
            reflectivity_curve*1e6
        )*1e-6,
        1e-6,
        None,
    )

    return noisy_reflectivity

def make_training_data(n_layers):   #part of generate training data
    training_data_input = make_training_input(n_layers)
    [thicknesses, roughnesses, SLDs_real, SLDs_img] = training_data_input
 
    training_reflectivity = make_reflectivity_curves(q_values, thicknesses, roughnesses, SLDs_real, SLDs_img)
    return training_reflectivity

# Define the predicted curve using four parameters (a, b, c, d)
def predicted_curve(q_values, thicknesses, roughnesses, SLDs_real, SLDs_img):
    return make_reflectivity_curves(q_values, thicknesses, roughnesses, SLDs_real, SLDs_img)

# Define the mean square error function
def mean_square_error(y_pred,y_true, q_values):
    y_actual = y_true
    thicknesses1, roughnesses1, SLDs_real1, SLDs_img1 = y_pred
    y_predicted = predicted_curve(q_values, thicknesses1, roughnesses1, SLDs_real1, SLDs_img1)
    mse = np.mean((y_actual - y_predicted) ** 2)
    return mse

X=list(range(1024))
importances = modelf.feature_importances_
data=pd.DataFrame()
data["parameter"]=[str(x) for x in X]
data["weights"]=importances
data=data.sort_values(by=['weights'], ascending=False)

data_list = []

for i in range(100):
    filename = f"/home/kowarik/Documents/Saeid/Data2layer/Sim_data_2layer_{i}.csv"
    datac = np.loadtxt(filename, delimiter=",")
    data_list.append(datac)
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

data1=pd.DataFrame(X)

train, validate, test = np.split(data1.sample(frac=1, random_state=42),[int(.8*len(data1)), int(.9*len(data1))])
X_train=train.iloc[:,:-8].values
y_train=train.iloc[:,-8:].values
y_train[:,-1]=y_train[:,-1]*-1
X_valid=validate.iloc[:,:-8].values
y_valid=validate.iloc[:,-8:].values
y_valid[:,-1]=y_valid[:,-1]*-1
X_test=test.iloc[:,:-8].values
y_test=test.iloc[:,-8:].values
y_test[:,-1]=y_test[:,-1]*-1

c=sample(range(len(y_test)),1000)
Tmp=X_test[c,:].copy()
New_X=np.array(Tmp)#(np.array(Tmp)-np.array(mi))/(np.array(ma)-np.array(mi))

Error_RFR=[]
Error_RFRNN=[]
Error_RFRWO=[]
Error_Rand=[]
Error_RandNN=[]
Error_RandWO=[]
Error_Eq=[]
Error_EqNN=[]
Error_EqWO=[]
Acc_RFR=[]
Acc_RFRNN=[]
Acc_RFRWO=[]
Acc_Rand=[]
Acc_RandNN=[]
Acc_RandWO=[]
Acc_Eq=[]
Acc_EqNN=[]
Acc_EqWO=[]

Ground_truth=y_test[c,:].copy()

Tmp=y_test[c,:].copy()
Ground_truth1=Tmp
Ground_truth1[:,-1]=Ground_truth1[:,-1]*-1
Ground_truth[:,-1]=Ground_truth[:,-1]*-1

List_to_eval=sample(range(1000),500)

flag2=0

for test_data_to_fit in List_to_eval[0:500]:
    if test_data_to_fit == List_to_eval[0]:
        flag2 = 0
    else:
        flag2=flag2+1
    ErrorRand=[]
    ErrorRandNN=[]
    ErrorRandWO=[]
    ErrorRFR=[]
    ErrorRFRNN=[]
    ErrorRFRWO=[]
    ErrorEq=[]
    ErrorEqNN=[]
    ErrorEqWO=[]


    tree_index=0
    tree = modelf.estimators_[tree_index]
    res = [eval(i) for i in data["parameter"]]
    for j in range(256):
        tree_index = 0
        node_index = 0
        next_node=0
        flag=0
        sol_index=list(map(int,list(np.arange(0,1024,1024/(4*(j+1))))))
        New_X=X_test[c[test_data_to_fit],:].copy()
        New_X[list(set(range(1024)) - set(sol_index))]=math.nan
        New_X=np.array(New_X)#(np.array(New_X)-np.array(mi))/(np.array(ma)-np.array(mi))
        New_X[[math.isnan(x) for x in New_X]] = 0 # replace NaN values with zeros
        predicted_list = autoencoder_model.predict(New_X.reshape(1, -1))
        predicted_list[predicted_list < 0] = 0 # replace negative values with zeros
        predicted_list = predicted_list.flatten()
        Tmp=Fit_Thick(predicted_list,q_values,Ground_truth1[test_data_to_fit,:])
        ErrorEq.append(abs(np.array(Ground_truth1[test_data_to_fit])-np.array(Tmp)))
        Tmp=parameters_model.predict([predicted_list.reshape(1, 1024)])
        initial_params = Tmp[0]
        y_true = X_test[c[test_data_to_fit],sol_index].copy()
        q_val = q_values[sol_index]
        def mean_square_error(y_pred):
            thicknesses1, roughnesses1, SLDs_real1, SLDs_img1, thicknesses2, roughnesses2, SLDs_real2, SLDs_img2 = y_pred
            y_predicted = predicted_curve(q_val, [thicknesses1,thicknesses2], [roughnesses1,roughnesses2], [SLDs_real1,SLDs_real2], [SLDs_img1,SLDs_img2])
            mse = np.mean((y_true - y_predicted) ** 2)
            return mse
        bounds = [(20,400), (0,60), (0,300), (0,20), (20,400), (0,60), (0,300), (0,20)]
        Tmp = minimize(mean_square_error, initial_params, method='L-BFGS-B', bounds = bounds)
        Tmp = np.array(Tmp.x)
        ErrorEqNN.append(abs(np.array(Ground_truth1[test_data_to_fit])-np.array(Tmp)))
        New_X=X_test[c[test_data_to_fit],:].copy()
        New_X[list(set(range(1024)) - set(sol_index))]=math.nan
        New_X[[math.isnan(x) for x in New_X]] = 0 # replace NaN values with zeros
        Tmp=Fit_Thick(New_X,q_values,Ground_truth1[test_data_to_fit,:])
        ErrorEqWO.append(abs(np.array(Ground_truth1[test_data_to_fit])-np.array(Tmp)))
        print(f"{j} from {flag2} is done!!...")
    print(f"{c[test_data_to_fit]}--Eq-dist {flag2} is Done!!...")
    
    for j in range(256):
        tree_index = 0
        node_index = 0
        next_node=0
        flag=0
        sol_index=[res[counter] for counter in range(4*(j+1))]
        New_X=X_test[c[test_data_to_fit],:].copy()
        New_X[list(set(range(1024)) - set(sol_index))]=math.nan
        New_X=np.array(New_X)#(np.array(New_X)-np.array(mi))/(np.array(ma)-np.array(mi))
        New_X[[math.isnan(x) for x in New_X]] = 0 # replace NaN values with zeros
        predicted_list = autoencoder_model.predict(New_X.reshape(1, -1))
        predicted_list[predicted_list < 0] = 0 # replace negative values with zeros
        predicted_list = predicted_list.flatten()
        Tmp=Fit_Thick(predicted_list,q_values,Ground_truth1[test_data_to_fit,:])
        ErrorRFR.append(abs(np.array(Ground_truth1[test_data_to_fit])-np.array(Tmp)))
        Tmp=parameters_model.predict([predicted_list.reshape(1, 1024)])
        initial_params = Tmp[0]
        y_true = X_test[c[test_data_to_fit],sol_index].copy()
        q_val = q_values[sol_index]
        def mean_square_error(y_pred):
            thicknesses1, roughnesses1, SLDs_real1, SLDs_img1, thicknesses2, roughnesses2, SLDs_real2, SLDs_img2 = y_pred
            y_predicted = predicted_curve(q_val, [thicknesses1,thicknesses2], [roughnesses1,roughnesses2], [SLDs_real1,SLDs_real2], [SLDs_img1,SLDs_img2])
            mse = np.mean((y_true - y_predicted) ** 2)
            return mse
        bounds = [(20,400), (0,60), (0,300), (0,20), (20,400), (0,60), (0,300), (0,20)]
        Tmp = minimize(mean_square_error, initial_params, method='L-BFGS-B', bounds = bounds)
        Tmp = np.array(Tmp.x)
        ErrorRFRNN.append(abs(np.array(Ground_truth1[test_data_to_fit])-np.array(Tmp)))
        New_X=X_test[c[test_data_to_fit],:].copy()
        New_X[list(set(range(1024)) - set(sol_index))]=math.nan
        New_X[[math.isnan(x) for x in New_X]] = 0 # replace NaN values with zeros
        Tmp=Fit_Thick(New_X,q_values,Ground_truth1[test_data_to_fit,:])
        ErrorRFRWO.append(abs(np.array(Ground_truth1[test_data_to_fit])-np.array(Tmp)))
        print(f"{j} from {flag2} is done!!...")
    print(f"{c[test_data_to_fit]}--RFR {flag2} is Done!!...")
    
    for j in range(256):
        tree_index = 0
        node_index = 0
        next_node=0
        flag=0
        sol_index=sample(list(range(1024)),4*(j+1))
        New_X=X_test[c[test_data_to_fit],:].copy()
        New_X[list(set(range(1024)) - set(sol_index))]=math.nan
        New_X=np.array(New_X)#(np.array(New_X)-np.array(mi))/(np.array(ma)-np.array(mi))
        New_X[[math.isnan(x) for x in New_X]] = 0 # replace NaN values with zeros
        predicted_list = autoencoder_model.predict(New_X.reshape(1, -1))
        predicted_list[predicted_list < 0] = 0 # replace negative values with zeros
        predicted_list = predicted_list.flatten()
        Tmp=Fit_Thick(predicted_list,q_values,Ground_truth1[test_data_to_fit,:])
        ErrorRand.append(abs(np.array(Ground_truth1[test_data_to_fit])-np.array(Tmp)))
        Tmp=parameters_model.predict([predicted_list.reshape(1, 1024)])
        initial_params = Tmp[0]
        y_true = X_test[c[test_data_to_fit],sol_index].copy()
        q_val = q_values[sol_index]
        def mean_square_error(y_pred):
            thicknesses1, roughnesses1, SLDs_real1, SLDs_img1, thicknesses2, roughnesses2, SLDs_real2, SLDs_img2 = y_pred
            y_predicted = predicted_curve(q_val, [thicknesses1,thicknesses2], [roughnesses1,roughnesses2], [SLDs_real1,SLDs_real2], [SLDs_img1,SLDs_img2])
            mse = np.mean((y_true - y_predicted) ** 2)
            return mse
        bounds = [(20,400), (0,60), (0,300), (0,20), (20,400), (0,60), (0,300), (0,20)]
        Tmp = minimize(mean_square_error, initial_params, method='L-BFGS-B', bounds = bounds)
        Tmp = np.array(Tmp.x)
        ErrorRandNN.append(abs(np.array(Ground_truth1[test_data_to_fit])-np.array(Tmp)))
        New_X=X_test[c[test_data_to_fit],:].copy()
        New_X[list(set(range(1024)) - set(sol_index))]=math.nan
        New_X[[math.isnan(x) for x in New_X]] = 0 # replace NaN values with zeros
        Tmp=Fit_Thick(New_X,q_values,Ground_truth1[test_data_to_fit,:])
        ErrorRandWO.append(abs(np.array(Ground_truth1[test_data_to_fit])-np.array(Tmp)))
        print(f"{j} from {flag2} is done!!...")
    print(f"{c[test_data_to_fit]}--Random {flag2} is Done!!...")
    
    p2=np.array(ErrorRand.copy())
    pNN2=np.array(ErrorRandNN.copy())
    pWO2=np.array(ErrorRandWO.copy())
    np2=p2.copy()
    p1=np.array(ErrorRFR.copy())
    pNN1=np.array(ErrorRFRNN.copy())
    pWO1=np.array(ErrorRFRWO.copy())
    p3=np.array(ErrorEq.copy())
    pNN3=np.array(ErrorEqNN.copy())
    pWO3=np.array(ErrorEqWO.copy())
    np1=p1.copy()
    for i in range(4):
        max_t=max(np1[:,i])
        np1[:,i]=np1[:,i]/max_t
    npNN1=pNN1.copy()
    for i in range(4):
        max_t=max(npNN1[:,i])
        npNN1[:,i]=npNN1[:,i]/max_t
    npWO1=pWO1.copy()
    for i in range(4):
        max_t=max(npWO1[:,i])
        npWO1[:,i]=npWO1[:,i]/max_t
    np3=p3.copy()
    for i in range(4):
        max_t=max(p3[:,i])
        np3[:,i]=np3[:,i]/max_t
    npNN3=pNN3.copy()
    for i in range(4):
        max_t=max(pNN3[:,i])
        npNN3[:,i]=npNN3[:,i]/max_t
    npWO3=pWO3.copy()
    for i in range(4):
        max_t=max(pWO3[:,i])
        npWO3[:,i]=npWO3[:,i]/max_t
    for i in range(4):
        max_t=max(p2[:,i])
        np2[:,i]=np2[:,i]/max_t
    npNN2=pNN2.copy()
    for i in range(4):
        max_t=max(pNN2[:,i])
        npNN2[:,i]=npNN2[:,i]/max_t
    npWO2=pWO2.copy()
    for i in range(4):
        max_t=max(pWO2[:,i])
        npWO2[:,i]=npWO2[:,i]/max_t
    rnp1=1-np.mean(np.power(np1, 2), axis=1)
    rnpNN1=1-np.mean(np.power(npNN1, 2), axis=1)
    rnpWO1=1-np.mean(np.power(npWO1, 2), axis=1)
    rnp3=1-np.mean(np.power(np3, 2), axis=1)
    rnpNN3=1-np.mean(np.power(npNN3, 2), axis=1)
    rnpWO3=1-np.mean(np.power(npWO3, 2), axis=1)

    rnp2=1-np.mean(np.power(np2, 2), axis=1)
    rnpNN2=1-np.mean(np.power(npNN2, 2), axis=1)
    rnpWO2=1-np.mean(np.power(npWO2, 2), axis=1)
    if flag2 == 0:
        Error_Rand=ErrorRand.copy()
        Error_RandNN=ErrorRandNN.copy()
        Error_RandWO=ErrorRandWO.copy()
        Acc_Rand=rnp2.copy()
        Acc_RandNN=rnpNN2.copy()
        Acc_RandWO=rnpWO2.copy()
        Error_RFR=ErrorRFR.copy()
        Error_RFRNN=ErrorRFRNN.copy()
        Error_RFRWO=ErrorRFRWO.copy()
        Error_Eq=ErrorEq.copy()
        Error_EqNN=ErrorEqNN.copy()
        Error_EqWO=ErrorEqWO.copy()
        Acc_RFR=rnp1.copy()
        Acc_RFRNN=rnpNN1.copy()
        Acc_RFRWO=rnpWO1.copy()
        Acc_Eq=rnp3.copy()
        Acc_EqNN=rnpNN3.copy()
        Acc_EqWO=rnpWO3.copy()
    else:
        Error_Rand=np.array(Error_Rand)+np.array(ErrorRand.copy())
        Error_RandNN=np.array(Error_RandNN)+np.array(ErrorRandNN.copy())
        Error_RandWO=np.array(Error_RandWO)+np.array(ErrorRandWO.copy())
        Acc_Rand=np.array(Acc_Rand)+np.array(rnp2.copy())
        Acc_RandNN=np.array(Acc_RandNN)+np.array(rnpNN2.copy())
        Acc_RandWO=np.array(Acc_RandWO)+np.array(rnpWO2.copy())
        Error_Eq=np.array(Error_Eq)+np.array(ErrorEq.copy())
        Error_EqNN=np.array(Error_EqNN)+np.array(ErrorEqNN.copy())
        Error_EqWO=np.array(Error_EqWO)+np.array(ErrorEqWO.copy())
        Acc_Eq=np.array(Acc_Eq)+np.array(rnp3.copy())
        Acc_EqNN=np.array(Acc_EqNN)+np.array(rnpNN3.copy())
        Acc_EqWO=np.array(Acc_EqWO)+np.array(rnpWO3.copy())
        Error_RFR=np.array(Error_RFR)+np.array(ErrorRFR.copy())
        Error_RFRNN=np.array(Error_RFRNN)+np.array(ErrorRFRNN.copy())
        Error_RFRWO=np.array(Error_RFRWO)+np.array(ErrorRFRWO.copy())
        Acc_RFR=np.array(Acc_RFR)+np.array(rnp1.copy())
        Acc_RFRNN=np.array(Acc_RFRNN)+np.array(rnpNN1.copy())
        Acc_RFRWO=np.array(Acc_RFRWO)+np.array(rnpWO1.copy())
    print(f"{c[test_data_to_fit]} is {flag2}th and done!!!...")
    np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Error_1024_RFR.csv', Error_RFR, delimiter=',')
    np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Error_1024_RFRNN.csv', Error_RFRNN, delimiter=',')
    np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Error_1024_RFRWO.csv', Error_RFRWO, delimiter=',')
    np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Error_1024_Rand.csv', Error_Rand, delimiter=',')
    np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Error_1024_RandNN.csv', Error_RandNN, delimiter=',')
    np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Error_1024_RandWO.csv', Error_RandWO, delimiter=',')
    np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Error_1024_Eq.csv', Error_Eq, delimiter=',')
    np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Error_1024_EqNN.csv', Error_EqNN, delimiter=',')
    np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Error_1024_EqWO.csv', Error_EqWO, delimiter=',')

    np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Acc_1024_RFR.csv', Acc_RFR, delimiter=',')
    np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Acc_1024_RFRNN.csv', Acc_RFRNN, delimiter=',')
    np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Acc_1024_RFRWO.csv', Acc_RFRWO, delimiter=',')
    np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Acc_1024_Rand.csv', Acc_Rand, delimiter=',')
    np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Acc_1024_RandNN.csv', Acc_RandNN, delimiter=',')
    np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Acc_1024_RandWO.csv', Acc_RandWO, delimiter=',')
    np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Acc_1024_Eq.csv', Acc_Eq, delimiter=',')
    np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Acc_1024_EqNN.csv', Acc_EqNN, delimiter=',')
    np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Acc_1024_EqWO.csv', Acc_EqWO, delimiter=',')







Error_Rand=Error_Rand/500
Error_RandNN=Error_RandNN/500
Error_RandWO=Error_RandWO/500

Acc_Rand=Acc_Rand/500
Acc_RandNN=Acc_RandNN/500
Acc_RandWO=Acc_RandWO/500

Error_RFR=Error_RFR/500
Error_RFRNN=Error_RFRNN/500
Error_RFRWO=Error_RFRWO/500

Acc_Eq=Acc_Eq/500
Acc_EqNN=Acc_EqNN/500
Acc_EqWO=Acc_EqWO/500

Error_Eq=Error_Eq/500
Error_EqNN=Error_EqNN/500
Error_EqWO=Error_EqWO/500

Acc_RFR=Acc_RFR/500
Acc_RFRNN=Acc_RFRNN/500
Acc_RFRWO=Acc_RFRWO/500

np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Error_1024_RFR.csv', Error_RFR, delimiter=',')
np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Error_1024_RFRNN.csv', Error_RFRNN, delimiter=',')
np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Error_1024_RFRWO.csv', Error_RFRWO, delimiter=',')
np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Error_1024_Rand.csv', Error_Rand, delimiter=',')
np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Error_1024_RandNN.csv', Error_RandNN, delimiter=',')
np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Error_1024_RandWO.csv', Error_RandWO, delimiter=',')
np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Error_1024_Eq.csv', Error_Eq, delimiter=',')
np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Error_1024_EqNN.csv', Error_EqNN, delimiter=',')
np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Error_1024_EqWO.csv', Error_EqWO, delimiter=',')

np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Acc_1024_RFR.csv', Acc_RFR, delimiter=',')
np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Acc_1024_RFRNN.csv', Acc_RFRNN, delimiter=',')
np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Acc_1024_RFRWO.csv', Acc_RFRWO, delimiter=',')
np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Acc_1024_Rand.csv', Acc_Rand, delimiter=',')
np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Acc_1024_RandNN.csv', Acc_RandNN, delimiter=',')
np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Acc_1024_RandWO.csv', Acc_RandWO, delimiter=',')
np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Acc_1024_Eq.csv', Acc_Eq, delimiter=',')
np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Acc_1024_EqNN.csv', Acc_EqNN, delimiter=',')
np.savetxt('/home/kowarik/Documents/Saeid/Data2layer/Models/Results/Acc_1024_EqWO.csv', Acc_EqWO, delimiter=',')

test_data_to_fit = 0

p1 = np.array(ErrorEqNN)
p2 = np.array(ErrorEqWO)
p3 = np.array(ErrorRFRNN)
p4 = np.array(ErrorRFRWO)
plt.plot(range(30),p1[:,test_data_to_fit])
plt.plot(range(30),p2[:,test_data_to_fit])
plt.plot(range(30),p3[:,test_data_to_fit])
plt.plot(range(30),p4[:,test_data_to_fit])
plt.legend(["NN","Refnx WO","RFRNN","RFRRefnx"])

