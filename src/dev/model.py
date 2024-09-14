import gpytorch
import torch


class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GP, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def optimize(model, likelihood, training_iter, train_x, train_y, verbose=True):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01
    )  # Includes GaussianLikelihood parameters lr= 0.01
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)

        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()

        if verbose == True:

            print(
                "Iter %d/%d - Loss: %.3f   lengthscale_1: %.3f   lengthscale_2: %.3f   lengthscale_3: %.3f   lengthscale_4: %.3f    lengthscale_4: %.3f    kernelVar: %.3f   noise: %.3f"
                % (
                    i + 1,
                    training_iter,
                    loss.item(),
                    model.covar_module.base_kernel.lengthscale[0][0].item(),
                    model.covar_module.base_kernel.lengthscale[0][1].item(),
                    model.covar_module.base_kernel.lengthscale[0][2].item(),
                    model.covar_module.base_kernel.lengthscale[0][3].item(),
                    model.covar_module.base_kernel.lengthscale[0][4].item(),
                    model.covar_module.outputscale.item(),
                    model.likelihood.noise.item(),
                )
            )

        optimizer.step()

    return model, likelihood
