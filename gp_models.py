import gpytorch

class BasicGPRegression(gpytorch.models.ExactGP):
    def __init__(self,train_x,train_y,likelihood):
        super().__init__(train_x,train_y,likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covariance_module = gpytorch.kernels.RBFKernel()
        #model class name for each instance of this class
        self.modelname = 'BasicGPRegression'

    def forward(self,data):
        mean_data = self.mean_module(data)
        covariance_data = self.covariance_module(data)
        return gpytorch.distributions.MultivariateNormal(mean_data,covariance_data)  
    
    