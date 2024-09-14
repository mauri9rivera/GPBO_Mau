import math
import torch
import gpytorch
from matplotlib import pyplot as plt


#TODO: Change this to GeneralGP constructer as a wrapper/supper builder for any GP you're going to build. Add args things so that its a super constructor.
class SimplestGP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(SimplestGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean() # Where you define the prior mean
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) # Where you define which kernel to choose

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
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


# Example usage
train_x = torch.linspace(0, 1, 100)
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2
# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = SimplestGP(train_x, train_y, likelihood)

