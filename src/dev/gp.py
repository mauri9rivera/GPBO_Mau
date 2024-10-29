import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from datetime import date
import scipy.stats as stats
import scipy.io
import numpy as np
from sklearn.utils import check_random_state
from scipy.stats import ortho_group
from torch.autograd import Variable
from kernels import OrthogonalProjectionNNGaussianKernel

np.random.seed(0)
torch.manual_seed(0)

#TODO: Consider using KeOps for fast kernel operations.

class GeneralGP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, kernel):
        super(GeneralGP, self).__init__(train_x, train_y, likelihood, kernel)
        self.mean_module = gpytorch.means.ZeroMean() 
        self.covar_module = kernel


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

#TODO: RE-evaluate if this initialization follows parent class initialization
class SimpleSMPGP(GeneralGP):
    
    def __init__(self, train_x, train_y, likelihood, num_mixtures=4):
        super(SimpleSMPGP, self).__init__(train_x, train_y, likelihood, gpytorch.means.ConstantMean(), gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures))
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(GP, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GP3(GeneralGP):
    """
    Parameters
    ----------
    train_x
    traiin_y
    likelihood
    kernel
    dtype
    initial_points_list : List of initial points
    """
    def __init__(self, train_x, train_y, likelihood, kernel, device, dtype=torch.double, init_samples=None):
        super(GP, self).__init__(train_x, train_y, likelihood, kernel)
        self.dtype = dtype
        self.device = device
        
        self.mean_module = gpytorch.means.ZeroMean()
        self.kernel = kernel.to(dtype=self.dtype, device=self.device)
        self.likelihood = likelihood
        self.init_samples = init_samples
        if init_samples is None:
            self.n_init_samples = 0
        else:
            self.initial_random_samples = int(self.init_samples.shape[0])
        self.model = None
        self.latent_model = None

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.kernel(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def select_query_point(self,  dim_searchspace, max_queries=100):

        '''
        if query > initial pts
                                    #Find the next point (max of acquisition function)
                            P_max = 0

                            if torch.isnan(MapPrediction).any():
                                print('nan in Mean map pred')
                                MapPrediction = torch.nan_to_num(MapPrediction)

                            AcquisitionMap = MapPrediction + kappa*torch.nan_to_num(torch.sqrt(VarianceMap)) # UCB acquisition
                            #AcquisitionMap = PI_ac(torch.nan_to_num(MapPrediction),y,torch.nan_to_num(VarianceMap))
                        
                    
                            # FOR STEP BY STEP, save maps
                            #YMU[s_i, c_i, k_i, rep_i, q] = MapPrediction#*MaxSeenResp
                            #YVAR[s_i, c_i, k_i, rep_i, q] = VarianceMap
                            #UCBMAP[s_i, c_i, k_i, rep_i, q] = AcquisitionMap

                            NextQuery= torch.where(AcquisitionMap.reshape(len(AcquisitionMap))==torch.max(AcquisitionMap.reshape(len(AcquisitionMap))))
                            
                            #print('Nextq',NextQuery)
                            # select next query
                            if len(NextQuery[0]) > 1:
                                # print('more than 1 next')
                                NextQuery = NextQuery[0][np.random.randint(len(NextQuery[0]))]    
                            else:   
                                NextQuery = NextQuery[0][0]
                            P_test[rep_i][q][0]= NextQuery
        '''
        
        P_test =  torch.zeros((max_queries, 2), device=self.device)
        MaxSeenResp = 0
        q = 0
        order_this = np.random.permutation(dim_searchspace)
        P_max = []
        hyp=[1.0]*(len(dim_searchspace)+2)

        self.kernel.base_kernel.lengthscale = hyp[:len(dim_searchspace)]
        self.kernel.outputscale = hyp[len(dim_searchspace)]
        self.likelihood.noise = hyp[len(dim_searchspace) + 1]

        while q < max_queries:

            if q > self.n_init_samples:
                pass
            else:
                P_test[q][0]=int(order_this[q])

            query_election = P_test[q][0]
            sample_resp = response[:, s_i, int(query_election.item())] #TODO: Change this
            test_respo = sample_resp[np.random.randint(len(sample_resp))]
            
            #print(test_respo)
            #print(torch.mean(sample_resp))
            test_respo += torch.normal(0.0, 0.02*torch.mean(sample_resp).item(), size=(1,), device=device).item()

            #Avoid zero response
            if test_respo < 0 or (test_respo == 0 and q == 0):
                test_respo= torch.tensor([0.0001], device=self.device)
            
            # The first element of P_test is the selected search
            # space point, the second the resulting value
            P_test[q][1]= test_respo

            y=(P_test[:q+1,1])
            #Update maximum response obtained in this round
            if (torch.max(torch.abs(y)) > MaxSeenResp) or (MaxSeenResp==0):
                # updated maximum response obtained in this round
                MaxSeenResp=torch.max(torch.abs(y))

            x= ch2xy[P_test[:q+1,0].long(),:].float() # search space position
            x = x.reshape((len(x),len(dim_searchspace)))

            y=y/MaxSeenResp
            y=y.float()

            if q == 0:
            
                self.model = GeneralGP(x,y, self.likelihood, self.kernel)

                if self.device == 'cuda':
                    self.model = self.model.cuda()
                    self.likelihood = self.likelihood.cuda()
            else:

                self.model.set_train_data(x, y, strict=False)

            self.model.train()
            self.likelihood.train()
            self.model, self.likelihood = vanilla_optimize(self.model, 10, x, y, verbose=False)

            self.model.eval()
            self.likelihood.eval()

    def update(self):
        pass

class GP2(GeneralGP):

    def __init__(self, likelihood, kernel, device, dtype=torch.double, init_samples=None):
        """
        Parameters
        ----------
        likelihood
        kernel
        dtype
        device
        initial_points_list : List of initial points
        """
        self.dtype = dtype
        self.device = device
        
        self.mean_module = gpytorch.means.ZeroMean()
        self.kernel = kernel.to(dtype=self.dtype, device=self.device)
        self.likelihood = likelihood
        self.init_samples = init_samples
        if init_samples is None:
            self.n_init_samples = 0
        else:
            self.n_init_samples = int(self.init_samples.shape[0])
        self.best_value = -1000000  # The process return the maximum point
        self.best_x = np.array([])
        self.X = torch.tensor([], dtype = self.dtype, device=self.device)  # running list of data
        self.y = torch.tensor([], dtype = self.dtype, device=self.device)  # running list of function evaluations

        self.model = None
        self.latent_model = None

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.kernel(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class AR_GP(GP):
    def __init__(self, train_x_low, train_y_low, train_x_high, train_y_high, likelihood_low, likelihood_high, kernel_low, kernel_high):
        # Initialize two independent GPs for each fidelity level
        super().__init__(train_x_low, train_y_low, likelihood_low, kernel_low)
        self.gp_low_fidelity = GP(train_x_low, train_y_low, likelihood_low, kernel_low)
        self.gp_high_fidelity = GP(train_x_high, train_y_high, likelihood_high, kernel_high)
        
        # Define mean and covariance modules for each fidelity level
        self.gp_low_fidelity.mean_module = gpytorch.means.ZeroMean()
        self.gp_high_fidelity.mean_module = gpytorch.means.ZeroMean()
        
        # Store the likelihoods
        self.likelihood_low = likelihood_low
        self.likelihood_high = likelihood_high

    def forward_low_fidelity(self, x_low):
        mean_x_low = self.gp_low_fidelity.mean_module(x_low)
        covar_x_low = self.gp_low_fidelity.covar_module(x_low)
        return gpytorch.distributions.MultivariateNormal(mean_x_low, covar_x_low)

    def forward_high_fidelity(self, x_high, x_low):
        # Use low-fidelity GP as input to the high-fidelity GP
        mean_low = self.forward_low_fidelity(x_low).mean
        # Concatenate low-fidelity predictions with high-fidelity inputs
        x_combined = torch.cat([x_high, mean_low.unsqueeze(-1)], dim=-1)
        
        mean_x_high = self.gp_high_fidelity.mean_module(x_combined)
        covar_x_high = self.gp_high_fidelity.covar_module(x_combined)
        return gpytorch.distributions.MultivariateNormal(mean_x_high, covar_x_high)

    def predict(self, x_low, x_high):
        # Switch to eval mode
        self.gp_low_fidelity.eval()
        self.gp_high_fidelity.eval()
        self.likelihood_low.eval()
        self.likelihood_high.eval()

        with torch.no_grad():
            # Low-fidelity predictions
            low_fidelity_pred = self.likelihood_low(self.forward_low_fidelity(x_low))
            # High-fidelity predictions conditioned on low-fidelity predictions
            high_fidelity_pred = self.likelihood_high(self.forward_high_fidelity(x_high, x_low))
        
        return low_fidelity_pred, high_fidelity_pred

class AdditiveGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, num_models, n_dims):
        super(AdditiveGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

        self.partition = [[i] for i in range(len(train_x.shape))]
        self.models = [] #Some initialization of a series of GPs


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # Task 2: Metropolis-Hastings Sampling Method
    def metropolis_hastings(self, n_samples):
        # Sample k models using Metropolis-Hastings
        models = self.models  # start with initial model
        for i in range(len(self.models)):
            current_model = models[i]
            proposed_model = self.propose_new_model()  # generate new model

            # Calculate acceptance probability A(M' | Mj)
            acceptance_ratio = self.calculate_acceptance(current_model, proposed_model)
            if np.random.rand() < acceptance_ratio:
                models.append(proposed_model)
            else:
                models.append(current_model)
        self.models = models

    def propose_new_model(self):
        # Copy the current partition
        new_partition = [component[:] for component in self.partition]
        
        if np.random.random() < 0.5:
            # Split strategy
            component_to_split = np.random.choice(new_partition)
            if len(component_to_split) > 1:  # Only split if the component has more than one element
                split_point = np.random.randint(1, len(component_to_split) - 1)
                new_partition.remove(component_to_split)
                new_partition.append(component_to_split[:split_point])
                new_partition.append(component_to_split[split_point:])
        else:
            # Merge strategy
            if len(new_partition) > 1:  # Only merge if there are at least two components
                components_to_merge = np.random.sample(new_partition, 2)
                new_partition.remove(components_to_merge[0])
                new_partition.remove(components_to_merge[1])
                new_partition.append(components_to_merge[0] + components_to_merge[1])

        # Create the new model with updated partition structure
        proposed_kernel = self.create_kernel_for_partition(new_partition)
        proposed_model = AdditiveGP(self.train_inputs[0], self.train_targets, self.likelihood, proposed_kernel, partition=new_partition)
        return proposed_model

    #TODO: Build using GPyTorch's AdditiveStructureKernel
    def create_kernel_for_partition(self, partition):
        # Define how to create a kernel from a given partition
        # This is where you would specify the kernel components (e.g., separate kernels for each partition element)
        base_kernel = gpytorch.kernels.RBFKernel()

        additive_kernel = sum([gpytorch.kernels.ScaleKernel(base_kernel) for _ in partition])
        return additive_kernel

    def calculate_acceptance(self, current_model, proposed_model):
        # Compute model evidence for each model and acceptance ratio
        likelihood_current = self.likelihood(current_model(self.train_inputs[0]))
        likelihood_proposed = self.likelihood(proposed_model(self.train_inputs[0]))
        g_proposed_given_current = stats.norm.pdf(proposed_model.covar_module.outputscale.item(), 
                                                  current_model.covar_module.outputscale.item(), 0.1)
        g_current_given_proposed = stats.norm.pdf(current_model.covar_module.outputscale.item(), 
                                                  proposed_model.covar_module.outputscale.item(), 0.1)
        
        acceptance_ratio = min(
            1,
            (likelihood_proposed.log_prob(self.train_targets).exp() * g_current_given_proposed) /
            (likelihood_current.log_prob(self.train_targets).exp() * g_proposed_given_current)
        )
        return acceptance_ratio

    # Task 3: Choose the best candidate point based on expected improvement (EI)
    def best_candidate(self, x):
        ei_vals = []
        for model in self.models:
            with torch.no_grad():
                pred = self.likelihood(model(x))
                mean, variance = pred.mean, pred.variance
                #improvement = mean - torch.max(self.train_targets)
                #ei = torch.mean(improvement * stats.norm.cdf(improvement / torch.sqrt(variance)))
                #ei_vals.append(ei.item())
        #best_model_index = int(np.argmax(ei_vals))
        return #x[best_model_index]

class ProjectionGP(GeneralGP):
    """
    Parameters
    ----------
    d_orig (int): Number of dimension for original space. 
    d_embedding (int): Number of dimensions for the lower dimensional subspace
    hidden_units (int): Number of dimensions for the hidden_units for Neural Network projection
    box_size (float): The boundary of the search space
    gamma (float): The weighting factor to balance the supervised loss and the unsupervised cosistency loss
    p (int): Number of random lambda in Equation (7)
    q (int): Number of random point in Equation (7)
    """
    def __init__(self, train_x, train_y, likelihood, kernel,
                 d_orig, d_embedding, hidden_units, device,
                 dtype=torch.double, box_size=1, gamma=1,
                 p=5, q=100):
        super(GP, self).__init__(train_x, train_y, likelihood, kernel)
        self.rng = check_random_state(0)
        self.box_size = box_size
        self.gamma = gamma
        self.dtype = dtype
        self.device = device
        self.d_embedding = d_embedding   # Dimension of the embedded space
        self.d_orig = d_orig   # Dimensions of the original space
        self.hidden_units = hidden_units  # size of hidden units for neural network
        self.p = p   # number of point on line segment
        self.q = q   # number of unlabeled data
        self.x_unlabeled = torch.rand(self.q, self.d_orig).to(dtype=self.dtype, device=self.device) * 2 - self.box_size

        #TODO: There's some manipulation the priors (initial points to modify here)

        m = ortho_group.dvs(dim=self.d_orig) # Draw a random (d_orig,d_orig) orthogonal matrix
        self.A = torch.tensor(m[:self.d_embedding,:]).to(dtype=self.dtype, device=self.device)

        # Produces (d_embedding, 2) array
        self.embedding_boundaries = torch.tensor(
            [[-np.sqrt(self.box_size * self.d_embedding),
                np.sqrt(self.box_size * self.d_embedding)]] * self.d_embedding).to(dtype=self.dtype, device=self.device)

        self.best_value = -1000000  # The process return the maximum point
        self.best_x = np.array([])

        self.X = torch.tensor([], dtype = self.dtype, device=self.device)  # running list of data
        self.X_embedded = torch.tensor([], dtype = self.dtype, device=self.device)  # running list of embedded data
        self.y = torch.tensor([], dtype = self.dtype, device=self.device)  # running list of function evaluations

        self.model = None
        self.latent_model = None

        # Create the covariance for the original GP model
        #?#
        self.k_fct = gpytorch.kernels.ScaleKernel(OrthogonalProjectionNNGaussianKernel(dim=self.d_orig, hidden_units=self.hidden_units,
                                        beta_min=0.21, beta_prior=gpytorch.priors.GammaPrior(2.0, 0.15)))
        self.k_fct.to(dtype=self.dtype, device=self.device)

        # Define the weight matrix and bias 
        self.W1 = Variable(self.k_fct.base_kernel.W1.data.clone(), requires_grad=False)
        self.b1 = Variable(self.k_fct.base_kernel.b1.data.clone(), requires_grad=False)
        self.W2 = Variable(self.k_fct.base_kernel.W2.data.clone(), requires_grad=False)
        self.b2 = Variable(self.k_fct.base_kernel.b2.data.clone(), requires_grad=False)

        # Create the covariance for the projected GP model
        #TODO: Reconsider this stepself.latent_k_fct = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=InvGammaPrior(2.0,0.15)))
        self.latent_k_fct.to(dtype=self.dtype, device=self.device, non_blocking=False) ## Cast to type of x_data

    
        # Define the likelihood function of the original GP model
        self.noise_prior = gpytorch.priors.torch_priors.GammaPrior(1.1, 0.05)
        self.noise_prior_mode = (self.noise_prior.concentration - 1) / self.noise_prior.rate
        self.lik_fct = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood(noise_prior=self.noise_prior,
                                                                          noise_constraint=
                                                                          gpytorch.constraints.GreaterThan(1e-8),
                                                                          initial_value=self.noise_prior_mode)
        self.lik_fct.to(dtype=self.dtype,device=self.device)

        # Define the likelihood function of the projected GP model
        self.latent_lik_fct = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood(noise_prior=self.noise_prior,
                                                                                 noise_constraint=
                                                                                 gpytorch.constraints.GreaterThan(1e-8),
                                                                                 initial_value=self.noise_prior_mode)
        self.latent_lik_fct.to(dtype=self.dtype, device=self.device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def projection_to_manifold(self, x):
        X = x.to(dtype=self.dtype,device=self.device)
        X_query = torch.relu(X @ self.W1.T + self.b1)
        X_query = X_query @ self.W2.T + self.b2
        X_return = X_query / torch.max(torch.abs(X_query), dim = 1)[0].reshape(-1,1)
        return X_return.reshape(-1,self.d_orig)
    
    def proposed_EI(self, z):
        Z_torch = torch.tensor(z, requires_grad=True).to(dtype=self.dtype,device=self.device)
        X = Z_torch @ self.A
        Z_proj = self.projection_to_manifold(X) @ self.A.T
        GP_pred = self.latent_model(Z_proj)
        mean = GP_pred.mean
        std = torch.sqrt(GP_pred.variance)
        posterior = torch.distributions.normal.Normal(mean, std)
        q = (mean - torch.max(self.y)) / std
        ei = (mean -  torch.max(self.y)) * posterior.cdf(q) + std * 10**posterior.log_prob(q)
        return -ei.data.cpu().numpy()
    
    #TODO: Implement all of these?
    def select_query_point(self, iteration=100):
        """
        iteration (int): 
            Number of iteration for maximizer maximize acquistion function
        """
        self.latent_model = get_fitted_model_torch(train_x=self.X_embedded, train_obj=self.y,
                                                covar_module = self.latent_k_fct, likelihood = self.latent_lik_fct, update_param = True)
        
        bounds = [(-np.sqrt(self.box_size * self.d_embedding),np.sqrt(self.box_size * self.d_embedding))]*self.d_embedding
        z = np.random.uniform(-np.sqrt(self.box_size * self.d_embedding),np.sqrt(self.box_size * self.d_embedding),(1,self.d_embedding))
        res = scipy.optimize.minimize(self.proposed_EI, z, method = "L-BFGS-B", bounds = bounds, options = {"maxiter":iteration})
        X_query_embedded = torch.tensor(res.x).reshape(-1,self.d_embedding).to(dtype=self.dtype, device=self.device)
        X_query = self.projection_to_manifold(X_query_embedded @ self.A)
        
        return torch.clamp(X_query,-self.box_size, self.box_size), X_query_embedded


    def update(self):
        """ 
        Update internal model for observed (X, y) from true function.
        Args:
            X_query ((1,d_orig) np.array):
                Point in original input space to query
            y_query (float):
                Value of black-box function evaluated at X_query
            update_param  (bool):
                Check if update the parameter of model
        """
        pass

    def get_fitted_model_semi(self):
        pass






