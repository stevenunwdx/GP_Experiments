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
    xdim = X.shape[0]
    ydim = X.shape[1]
    value_array = np.zeros((xdim,ydim))
    for idx1 in range(xdim):
        for  idx2 in range(ydim):
                    if ( ((X[idx1][idx2] >= 10) & (X[idx1][idx2]<=30)) & ((Y[idx1][idx2] >=10)& (Y[idx1][idx2]<=30)) ):
                        value_array[idx1,idx2] = 1
    return value_array

def indicator_edges(X,Y):
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


#
# # Auxilliary Functions Generate to Data from each test function.
# # We can evaluate the functions wherever their defining formulas are valid, not only where defined in the Test Function Catalog

# #Data generation for functions with 2-dimensional input domain: Gaussian noisy observations, mean zero fixed variance noise:
# def generate_data_2d_problem(test_function, xmin, xmax, ymin, ymax, data_set_size,noise_var=0,save=True,data_file_name=None): 
#     data_input_x_coords = np.random.uniform(xmin,xmax,size=data_set_size).reshape(-1,1) #
#     data_input_y_coords = np.random.uniform(ymin,ymax,size=data_set_size).reshape(-1,1) #or other

#     data_input_coords = np.concatenate((data_input_x_coords,data_input_y_coords),axis=1)
#     if (noise_var==0):
#         data_output_values = test_function(data_input_x_coords,data_input_y_coords)
#     else:
#         data_output_values = test_function(data_input_x_coords,data_input_y_coords)+ math.sqrt(noise_var)*np.randn(data_set_size)
#     if(save):
#         dataframe = pd.concat([pd.DataFrame(data_input_coords),pd.DataFrame(data_output_values)],axis=1)
#         dataframe.to_csv(data_file_name,header=False,index=False)

#     data_input_coords = torch.Tensor(data_input_coords)  
#     #data_input_coords is a torch.Tensor of size (data_set_size, d) ,where d is the dimension of input space
#     data_output_values = torch.Tensor(data_output_values).squeeze()
#     #data_output_values is a torch.Tensor of size (data_set_size,)  
#     #the objects returned by this function are meant to be passed into constructors and forward methods for PyTorch/GPyTorch modules
#     #Single (scalar) output
#     return data_input_coords, data_output_values

# #Data generation for functions with 4-dimensional input domain:
# def generate_data_4d_problem(test_function, x1min, x1max, x2min, x2max, x3min, x3max, x4min, x4max, data_set_size,noise_var=0,save=True,data_file_name=None):
#     x1_coords = np.random.uniform(x1min,x1max,size=data_set_size).reshape(-1,1)
#     x2_coords = np.random.uniform(x2min,x2max,size=data_set_size).reshape(-1,1)
#     x3_coords = np.random.uniform(x3min,x3max,size=data_set_size).reshape(-1,1)
#     x4_coords = np.random.uniform(x4min,x4max,size=data_set_size).reshape(-1,1)

#     data_input_coords = np.concatenate((x1_coords, x2_coords, x3_coords, x4_coords),axis=1)
#     #data_input_coords: shape is (data_set_size ,d) each entry is a d-tuple where d is input space dimension (d=4 here)

#     if (noise_var==0):
#         data_output_values = test_function(x1_coords, x2_coords, x3_coords, x4_coords)
#         #data_output_values: shape is (data_set_size,1) each entry is a (1,) array
#     else:
#         data_output_values = test_function(x1_coords,x2_coords,x3_coords,x4_coords) + math.sqrt(noise_var)*np.randn(data_set_size)
#     if(save):
#         dataframe = pd.concat([pd.DataFrame(data_input_coords),pd.DataFrame(data_output_values)],axis=1)
#         dataframe.to_csv(data_file_name,header=False,index=False)

#     data_input_coords = torch.Tensor(data_input_coords)  
#     #data_input_coords is a torch.Tensor of size (data_set_size, d) ,where d is the dimension of input space
#     data_output_values = torch.Tensor(data_output_values).squeeze()
#     #data_output_values is a torch.Tensor of size (data_set_size,)  
#     #the objects returned by this function are meant to be passed into constructors and forward methods for PyTorch/GPyTorch modules
#     #Single (scalar) output
#     return data_input_coords, data_output_values





