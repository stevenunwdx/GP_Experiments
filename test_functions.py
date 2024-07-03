import numpy as np
#import math
#import pandas as pd
#import torch
#Test Function 0

def test_function_0(X,Y):
    value_array = X**2 -Y**2  #no meshgrid in function implementation.
    return value_array

#Test Function 1

def modified_ackley(X,Y):
    value_array = - 5 * np.exp(-0.2* np.sqrt(0.5*((0.5*(X-20))**2 + (0.5*(Y-20))**2) ) ) - np.exp(0.5* (np.cos(0.125*(X-20)) + np.cos(0.125*(Y-20))  ) ) +np.exp(1)
    return value_array

def step_function(X,Y):
    value_array = 0.5*( np.exp(-0.5*(np.floor(0.2*X))) + np.exp(-0.5*(np.floor(0.2*Y))) ) - 1
    return value_array

def indicator_square(X,Y):
    if (len(list(X.shape))==1):
        X=np.expand_dims(X,axis=0)
    if (len(list(Y.shape))==1):
        Y=np.expand_dims(Y,axis=0)
    xdim = X.shape[0]
    ydim = X.shape[1]
    value_array = np.zeros((xdim,ydim))
    for idx1 in range(xdim):
        for  idx2 in range(ydim):
                    if ( ((X[idx1][idx2] >= 10) & (X[idx1][idx2]<=30)) & ((Y[idx1][idx2] >=10)& (Y[idx1][idx2]<=30)) ):
                        value_array[idx1,idx2] = 1
    return value_array

def indicator_edges(X,Y):
    if (len(list(X.shape))==1):
        X=np.expand_dims(X,axis=0)
    if (len(list(Y.shape))==1):
        Y=np.expand_dims(Y,axis=0)
    xdim = X.shape[0]
    ydim = X.shape[1]
    value_array = np.ones((xdim,ydim))
    for idx1 in range(xdim):
        for  idx2 in range(ydim):
            if ( ((X[idx1][idx2] >= 10) & (X[idx1][idx2]<=30)) & ((Y[idx1][idx2] >=10)& (Y[idx1][idx2]<=30)) ):
                value_array[idx1,idx2] = 0
    return value_array

def test_function_1(X,Y):
    value_array = modified_ackley(X,Y) * indicator_square(X,Y) + step_function(X,Y) * indicator_edges(X,Y)
    return value_array


#Test Function 2

def test_function_2(X,Y):
    R_squared = X**2+ Y**2
    value_array = (1/(X**2-9)) * np.exp(-0.1*(R_squared))*np.sin(R_squared)
    return value_array

#Test Function 3

def test_function_3(X,Y,a=1,b=1):
    if (len(list(X.shape))==1):
        X=np.expand_dims(X,axis=0)
    if (len(list(Y.shape))==1):
        Y=np.expand_dims(Y,axis=0)
    ydim = X.shape[0]
    xdim = X.shape[1]
    value_array = np.zeros((ydim,xdim))
    for idx1 in range(ydim):
        for idx2 in range(xdim):
            x = X[idx1][idx2]
            y = Y[idx1][idx2]
            if abs(x-y)<1:
                value_array[idx1][idx2] += a*np.exp(-1/( 1 - (x-y)**2 )  )
            if abs(-x-y)<1:
                value_array[idx1][idx2] += b* np.exp(-1/( 1 - (-x-y)**2 )  ) 
    return value_array

#Test Function 4

def test_function_4(X1,X2,X3,X4):
    Rsquared = X1**2 + X2**2 + X3**2 + X4**2
    value_array = 100/( (X1**2 + X2**2) * (X3**2 + X4**2) ) * (X1**2) * X2 * np.exp(-0.1*Rsquared)
    return value_array
#Some sections of Test Function 4
def test_function_4_x1x2_section(x1,x2,X3,X4):
    value_array = test_function_4(x1,x2,X3,X4)  
    value_array = value_array[0,0,:,:]
    return value_array

def test_function_4_x3x4_section(X1,X2,x3,x4):
    value_array = test_function_4(X1,X2,x3,x4)
    value_array = value_array[:,:,0,0]
    return value_array