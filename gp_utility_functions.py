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
    #save the training data itself
    #training error vs epoch  (loss array into array)
    #or error vs hyperparameter tuning
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
    
    torch.save(model.state_dict(),modelname)  #but no structural information like number neurons, etc


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
    mape = sklearn.metrics.mean_absolute_percentage_error(test_output.detach().numpy(),posterior_gp_mean.detach().numpy())
    return r2_score, mape


    


# Auxilliary Functions Generate to Data from each test function.
# We can evaluate the functions wherever their defining formulas are valid, not only where defined in the Test Function Catalog

#Data generation for functions with 2-dimensional input domain: Gaussian noisy observations, mean zero fixed variance noise:
def generate_data_2d_problem(test_function, xmin, xmax, ymin, ymax, data_set_size,noise_var=0,save=True,data_file_name=None): 
    data_input_x_coords = np.random.uniform(xmin,xmax,size=data_set_size).reshape(-1,1) #
    data_input_y_coords = np.random.uniform(ymin,ymax,size=data_set_size).reshape(-1,1) #or other

    data_input_coords = np.concatenate((data_input_x_coords,data_input_y_coords),axis=1)
    if (noise_var==0):
        data_output_values = test_function(data_input_x_coords,data_input_y_coords)
    else:
        data_output_values = test_function(data_input_x_coords,data_input_y_coords)+ math.sqrt(noise_var)*np.randn(data_set_size)
    if(save):
        dataframe = pd.concat([pd.DataFrame(data_input_coords),pd.DataFrame(data_output_values)],axis=1)
        dataframe.to_csv(data_file_name,header=False,index=False)

    data_input_coords = torch.Tensor(data_input_coords)  
    #data_input_coords is a torch.Tensor of size (data_set_size, d) ,where d is the dimension of input space
    data_output_values = torch.Tensor(data_output_values).squeeze()
    #data_output_values is a torch.Tensor of size (data_set_size,)  
    #the objects returned by this function are meant to be passed into constructors and forward methods for PyTorch/GPyTorch modules
    #Single (scalar) output
    return data_input_coords, data_output_values

#Data generation for functions with 4-dimensional input domain:
def generate_data_4d_problem(test_function, x1min, x1max, x2min, x2max, x3min, x3max, x4min, x4max, data_set_size,noise_var=0,save=True,data_file_name=None):
    x1_coords = np.random.uniform(x1min,x1max,size=data_set_size).reshape(-1,1)
    x2_coords = np.random.uniform(x2min,x2max,size=data_set_size).reshape(-1,1)
    x3_coords = np.random.uniform(x3min,x3max,size=data_set_size).reshape(-1,1)
    x4_coords = np.random.uniform(x4min,x4max,size=data_set_size).reshape(-1,1)

    data_input_coords = np.concatenate((x1_coords, x2_coords, x3_coords, x4_coords),axis=1)
    #data_input_coords: shape is (data_set_size ,d) each entry is a d-tuple where d is input space dimension (d=4 here)

    if (noise_var==0):
        data_output_values = test_function(x1_coords, x2_coords, x3_coords, x4_coords)
        #data_output_values: shape is (data_set_size,1) each entry is a (1,) array
    else:
        data_output_values = test_function(x1_coords,x2_coords,x3_coords,x4_coords) + math.sqrt(noise_var)*np.randn(data_set_size)
    if(save):
        dataframe = pd.concat([pd.DataFrame(data_input_coords),pd.DataFrame(data_output_values)],axis=1)
        dataframe.to_csv(data_file_name,header=False,index=False)

    data_input_coords = torch.Tensor(data_input_coords)  
    #data_input_coords is a torch.Tensor of size (data_set_size, d) ,where d is the dimension of input space
    data_output_values = torch.Tensor(data_output_values).squeeze()
    #data_output_values is a torch.Tensor of size (data_set_size,)  
    #the objects returned by this function are meant to be passed into constructors and forward methods for PyTorch/GPyTorch modules
    #Single (scalar) output
    return data_input_coords, data_output_values





