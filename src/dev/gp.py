import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from datetime import date
import scipy.stats as stats
import scipy.io
import numpy as np

np.random.seed(0)
torch.manual_seed(0)

#TODO: Consider using KeOps for fast kernel operations.

class GeneralGP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, mean_module=gpytorch.means.ConstantMean(), covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()), **kwargs):
        super(GeneralGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module 
        self.covar_module = covar_module 


    @classmethod
    def forward(self, x, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

    #TODO: Adapt this method to be flexible to the attributes of the model 
    def show_model_state(self):
        model_state = self.state_dict()

        print("Model State Dict:")
        print(f"Mean value: {model_state['mean_module.raw_constant']}")
        print(f"Kernel value: {model_state['covar_module.base_kernel.raw_lengthscale']}")
        print(f"Lengthscale value: {model_state['covar_module.base_kernel.raw_lengthscale']}")

    
    def show_model_hyperparms(self):

        print("Model's hyperparameters:")
        for param_name, param in self.named_parameters():
            print(f'Parameter name: {param_name:42} value = {param.item()}')

class SimpleSMPGP(GeneralGP):
    
    def __init__(self, train_x, train_y, likelihood, num_mixtures=4):
        super(SimpleSMPGP, self).__init__(train_x, train_y, likelihood, gpytorch.means.ConstantMean(), gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures))
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class SimpleGP(GeneralGP):

    def __init__(self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())):
        super(SimpleGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ProjectionGP(GeneralGP):
    
    def __init__(self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())):
        super(SimpleGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    #TODO: Modify this subclass init's method to mimic RPM-BO paper
    def projection_to_manifold(self, x, device):
        X = x.to(dtype=self.dtype,device=device)
        X_query = torch.relu(X @ self.W1.T + self.b1)
        X_query = X_query @ self.W2.T + self.b2
        X_return = X_query / torch.max(torch.abs(X_query), dim = 1)[0].reshape(-1,1)
        return X_return.reshape(-1,self.d_orig)
    
# Example usage
train_x = torch.linspace(0, 1, 100)
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2
# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()


model = GeneralGP(train_x, train_y, likelihood)
SMP_model = SimpleSMPGP(train_x, train_y, likelihood)



