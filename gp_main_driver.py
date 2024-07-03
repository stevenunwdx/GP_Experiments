import gp_models
import gp_utility_functions
import test_functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

print('Gaussian Process Experiment Driver')
print('Synclesis, Inc\n')

print('Experiment 1: RBF Kernel, Constant Mean GP on Each Test Function\n')

modelclass = gp_models.BasicGPRegression

model_list = [modelclass]
func_list = [test_functions.test_function_0,
             test_functions.test_function_1,
             test_functions.test_function_2,
             test_functions.test_function_3,
             test_functions.test_function_4]
numbers_samples = list(range(5,200,5))
#numbers_samples = [5,10]
print('Number of learning problems:',len(func_list))
print('Numbers of sample sizes: ',len(numbers_samples))
print('Number of models:',len(model_list),'\n')

print('Number of Experiments:',len(func_list)*len(numbers_samples)*len(model_list))

experiment_dict = {'Test Function':[],
                   'Model':[],
                   'Number of Samples':[],
                   'Coefficient of Determination':[],
                   'MAPE':[]}  #dictionary is a "3-d" array indexing experiments and outcomes

for k, func in enumerate(func_list):  #each learning problem
    for n in numbers_samples: #each number of samples
        if (k< len(func_list)-1):
            train_input, train_output = gp_utility_functions.generate_data_2d_problem(func,0,30,0,30,data_set_size=n,save=True)
            test_input, test_output = gp_utility_functions.generate_data_2d_problem(func,0,30,0,30,data_set_size=1000,save=False)
        else:
            train_input, train_output = gp_utility_functions.generate_data_4d_problem(func,1,6,1,6,1,6,1,6,data_set_size=n,save=True)
            test_input, test_output = gp_utility_functions.generate_data_4d_problem(func,1,6,1,6,1,6,1,6,data_set_size=1000,save=False)
        for i, modelclass in enumerate(model_list):
            modelname = 'testfunction{}_model{}_NSamples={}'.format(k,modelclass.__name__,n)
            gp_utility_functions.single_instance_model_training(modelclass,modelname,train_input,train_output,n_epochs=500)

            r_squared_score, mape = gp_utility_functions.single_instance_model_testing(modelclass,modelname,test_input,test_output)

            experiment_dict['Test Function'].append(k)
            experiment_dict['Model'].append(modelclass.__name__)
            experiment_dict['Number of Samples'].append(n)
            experiment_dict['Coefficient of Determination'].append(r_squared_score)
            experiment_dict['MAPE'].append(mape)

experiment_data_frame = pd.DataFrame(experiment_dict)
output_file_name = 'GP_Experiment_1_Data'
experiment_data_frame.to_csv(output_file_name,index =False)
experiment_data_frame.to_excel('GP_Experiment_1_Data.xlsx')

#plot array of subplots for R^2 Score vs. NSamples.

fig, ax = plt.subplots(len(model_list),len(func_list),figsize = (20,5))

for k in range(len(func_list)):
    for i, modelclass in enumerate(model_list):
        mask = ((experiment_data_frame['Test Function'] == k) & (experiment_data_frame['Model']== modelclass.__name__))
        df_test_function_k_model_i = experiment_data_frame[mask]
        ns_samples = df_test_function_k_model_i['Number of Samples'].to_numpy()
        r_squared_scores = df_test_function_k_model_i['Coefficient of Determination'].to_numpy()
        if (len(model_list)==1):
            ax[k].plot(ns_samples,r_squared_scores,'o-',linewidth=0.7, markersize=2,label = 'R^2 Score')
            ax[k].set_xlim(0,max(numbers_samples))
            ax[k].set_ylim(0,1.2)
            ax[k].hlines(y=0.99,xmin=0, xmax=max(numbers_samples),linestyle='--',linewidth=0.5,color ='r',label = 'R^2 Score = 0.99')
            ax[k].set_xlabel('Number of Training Samples')
            ax[k].set_ylabel('Coefficient of Determination')
            ax[k].set_title(f'Test Function {k}\nModel: {modelclass.__name__}')
            ax[k].legend()

        else: 
            print('else block executed')
            ax[i,k].plot(ns_samples,r_squared_scores,'o-',linewidth=0.7, markersize=3,label = 'R^2 Score')
            ax[i,k].set_xlim(0,max(numbers_samples))
            ax[i,k].set_ylim(0,1.2)
            ax[i,k].hlines(y=0.99,xmin=0, xmax=max(numbers_samples),linestyle='--',linewidth=0.5,color ='r',label = 'R^2 Score = 0.99')
            ax[i,k].set_xlabel('Number of Training Samples')
            ax[i,k].set_ylabel('Coefficient of Determination')
            ax[i,k].set_title(f'Test Function {k}\nModel: {modelclass.__name__}')
            ax[i,k].legend()

fig.savefig('GP_Experiment_1_R2_Graph')

#plot array of subplots for MAPE vs. NSamples.

fig, ax = plt.subplots(len(model_list),len(func_list),figsize = (20,5))

for k in range(len(func_list)):
    for i, modelclass in enumerate(model_list):
        mask = ((experiment_data_frame['Test Function'] == k) & (experiment_data_frame['Model']== modelclass.__name__))
        df_test_function_k_model_i = experiment_data_frame[mask]
        ns_samples = df_test_function_k_model_i['Number of Samples'].to_numpy()
        mapes = df_test_function_k_model_i['MAPE'].to_numpy()
        if (len(model_list)==1):
            ax[k].plot(ns_samples,mapes,'o-',linewidth=0.7, markersize=2,label = 'MAPE')
            ax[k].set_xlim(0,max(numbers_samples))
            ax[k].set_xlabel('Number of Training Samples')
            ax[k].set_ylabel('MAPE')
            ax[k].set_title(f'Test Function {k}\nModel: {modelclass.__name__}')
            ax[k].legend()

        else: 
            print('else block executed')
            ax[i,k].plot(ns_samples,mapes,'o-',linewidth=0.7, markersize=3,label = 'MAPE')
            ax[i,k].set_xlim(0,max(numbers_samples))
            ax[i,k].set_xlabel('Number of Training Samples')
            ax[i,k].set_ylabel('MAPE')
            ax[i,k].set_title(f'Test Function {k}\nModel: {modelclass.__name__}')
            ax[i,k].legend()

fig.savefig('GP_Experiment_1_MAPE_Graph')

print('Program run successful: experiment complete')



