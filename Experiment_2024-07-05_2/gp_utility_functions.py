import gp_models

import math
import numpy as np
import pandas as pd
import torch
import warnings
import torch
import gpytorch
import sklearn.metrics


#Model initialization, training, and saving for a single instance of a model that subclasses gpytorch.models.ExactGP in the file gp_models.py 
def single_instance_model_training(modelclass,modelname,train_input,train_output,n_epochs,likelihood=gpytorch.likelihoods.GaussianLikelihood()):
    likelihood = likelihood
    model = modelclass(train_input,train_output,likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood,model)
    for i in range(n_epochs):
        optimizer.zero_grad()
        predictive_dist = model(train_input)
        loss = -mll(predictive_dist,train_output)
        loss.backward()
        optimizer.step()
    
    torch.save(model.state_dict(),modelname) 


def single_instance_model_testing(modelclass,state_dict_name,test_input,test_output,likelihood=gpytorch.likelihoods.GaussianLikelihood()):
    likelihood = likelihood
    model = modelclass(test_input,test_output,likelihood)
    state_dict = torch.load(state_dict_name)
    model.load_state_dict(state_dict)
    likelihood.eval()
    model.eval()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        predictive_dist = model(test_input)
    posterior_gp_mean = predictive_dist.mean

    r2_score = sklearn.metrics.r2_score(test_output.detach().numpy(), posterior_gp_mean.detach().numpy())

    mask = np.abs(test_output.detach().numpy())>=0.0001
    mape = sklearn.metrics.mean_absolute_percentage_error(test_output.detach().numpy()[mask],posterior_gp_mean.detach().numpy()[mask])

    max_res_error = sklearn.metrics.max_error(test_output.detach().numpy(),posterior_gp_mean.detach().numpy())
    rmse = sklearn.metrics.root_mean_squared_error(test_output.detach().numpy(),posterior_gp_mean.detach().numpy())
    return r2_score, mape, max_res_error, rmse

    


# Auxilliary Functions Generate to Data from each test function.

def generate_data_2d_problem(test_function, xmin, xmax, ymin, ymax, data_set_size,noise_var=0,save=True,data_file_name=None): 
    """Generate (noisy) random data with uniform distribution on rectangular grid from known function.  Used to generate training data.
    Returns a tuple:
    data_input_coords - 2-dim torch.Tensor of size (data_set_size,d),where d is the dimension of input space
    data_output_values - 1-dim torch.Tensor of size (data_set_size,)
    Returned data structures are suitable for immediate passing into PyTorch/GPyTorch modules"""
    data_input_x_coords = np.random.uniform(xmin,xmax,size=data_set_size).reshape(-1,1)
    data_input_y_coords = np.random.uniform(ymin,ymax,size=data_set_size).reshape(-1,1)

    data_input_coords = np.concatenate((data_input_x_coords,data_input_y_coords),axis=1)
    if (noise_var==0):
        data_output_values = test_function(data_input_x_coords,data_input_y_coords)
    else:
        data_output_values = test_function(data_input_x_coords,data_input_y_coords)+ math.sqrt(noise_var)*np.randn(data_set_size)
    if(save):
        dataframe = pd.concat([pd.DataFrame(data_input_coords),pd.DataFrame(data_output_values)],axis=1)
        dataframe.to_csv(data_file_name,header=False,index=False)

    data_input_coords = torch.Tensor(data_input_coords)  
    data_output_values = torch.Tensor(data_output_values).squeeze()

    return data_input_coords, data_output_values

def generate_data_4d_problem(test_function, x1min, x1max, x2min, x2max, x3min, x3max, x4min, x4max, data_set_size,noise_var=0,save=True,data_file_name=None):
    """Generate  (noisy) random data with uniform distribution on rectangular grid from known function. Used to generate training data.
    Returns a tuple:
    data_input_coords - 2-dim torch.Tensor of size (data_set_size,d),where d is the dimension of input space
    data_output_values - 1-dim torch.Tensor of size (data_set_size,)
    Returned data structures are suitable for immediate passing into PyTorch/GPyTorch modules"""
    x1_coords = np.random.uniform(x1min,x1max,size=data_set_size).reshape(-1,1)
    x2_coords = np.random.uniform(x2min,x2max,size=data_set_size).reshape(-1,1)
    x3_coords = np.random.uniform(x3min,x3max,size=data_set_size).reshape(-1,1)
    x4_coords = np.random.uniform(x4min,x4max,size=data_set_size).reshape(-1,1)

    data_input_coords = np.concatenate((x1_coords, x2_coords, x3_coords, x4_coords),axis=1)

    if (noise_var==0):
        data_output_values = test_function(x1_coords, x2_coords, x3_coords, x4_coords)
    else:
        data_output_values = test_function(x1_coords,x2_coords,x3_coords,x4_coords) + math.sqrt(noise_var)*np.randn(data_set_size)
    if(save):
        dataframe = pd.concat([pd.DataFrame(data_input_coords),pd.DataFrame(data_output_values)],axis=1)
        dataframe.to_csv(data_file_name,header=False,index=False)
    data_input_coords = torch.Tensor(data_input_coords)  
    data_output_values = torch.Tensor(data_output_values).squeeze()
    return data_input_coords, data_output_values

def generate_data_2d_grid(func,xmin,xmax,ymin,ymax,n_subintervals,noise_var=0):
    """Generate (noisy) test data from a known function on a uniform rectangular grid.
    n_subintervals is number of subintervals in each axis.\nReturns a tuple:\n
    input_points - 2-dim torch.Tensor of size ((n_subintervals)^d,2)
    ouput_values - 1-dim torch.Tensor of size ((n_subintervals)^d,)\n
    Objects returned are in format suitable for passing into PyTorch/GPyTorch modules  """
    
    x_coords = np.linspace(xmin,xmax,n_subintervals)
    y_coords = np.linspace(ymin,ymax,n_subintervals)
    x_coords,y_coords = np.meshgrid(x_coords,y_coords)
    input_points = np.concatenate(
    (x_coords.reshape(-1,1),
     y_coords.reshape(-1,1)
    ),
        axis=1)
    if (noise_var==0):
        output_values = func(input_points[:,0],input_points[:,1])
    else:
        print(input_points.shape[0])
        output_values = func(input_points[:,0],input_points[:,1]) + math.sqrt(noise_var)*np.random.randn(input_points.shape[0])
        
    input_points = torch.Tensor(input_points)
    output_values = torch.Tensor(output_values).squeeze()
    
    return input_points, output_values 


def generate_data_4d_grid(func,x1min,x1max,x2min,x2max,x3min,x3max,x4min,x4max,n_subintervals,noise_var=0):
    """Generate (noisy) test data from a known function on a uniform rectangular grid.
    n_subintervals is number of subintervals in each axis.\nReturns a tuple:\n
    input_points - 2-dim torch.Tensor of size ((n_subintervals)^d,2)
    ouput_values - 1-dim torch.Tensor of size ((n_subintervals)^d,)\n
    Objects returned are in format suitable for passing into PyTorch/GPyTorch modules  """
    
    x1_coords = np.linspace(x1min,x1max,n_subintervals)
    x2_coords = np.linspace(x2min,x2max,n_subintervals)
    x3_coords = np.linspace(x3min,x3max,n_subintervals)
    x4_coords = np.linspace(x4min,x4max,n_subintervals)
    x1_coords,x2_coords,x3_coords,x4_coords = np.meshgrid(x1_coords,x2_coords,x3_coords,x4_coords)
    input_points = np.concatenate(
    (x1_coords.reshape(-1,1),
    x2_coords.reshape(-1,1),
    x3_coords.reshape(-1,1),
    x4_coords.reshape(-1,1),
    ),
        axis=1)
    if (noise_var==0):
        output_values = func(input_points[:,0],input_points[:,1],input_points[:,2],input_points[:,3])
    else:
        output_values = func(input_points[:,0],input_points[:,1],input_points[:,2],input_points[:,3]) + math.sqrt(noise_var)*np.random.randn(input_points.shape[0])

    input_points = torch.Tensor(input_points)
    output_values = torch.Tensor(output_values)
    return input_points, output_values 