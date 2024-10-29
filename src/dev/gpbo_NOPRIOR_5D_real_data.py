# -*- coding: utf-8 -*-

import numpy as np
import time
import math
from datetime import date
import scipy.stats as stats
import torch
import gpytorch
import os
#from utils import *
import scipy.io

np.random.seed(0)
torch.manual_seed(0)

"""## Utils"""

class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(GP, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def optimize(model, likelihood, training_iter, train_x, train_y, verbose= True):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters lr= 0.01
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

        if verbose== True:

            print('Iter %d/%d - Loss: %.3f   lengthscale_1: %.3f   lengthscale_2: %.3f   lengthscale_3: %.3f   lengthscale_4: %.3f    lengthscale_4: %.3f    kernelVar: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale[0][0].item(),
            model.covar_module.base_kernel.lengthscale[0][1].item(),
            model.covar_module.base_kernel.lengthscale[0][2].item(),
            model.covar_module.base_kernel.lengthscale[0][3].item(),
            model.covar_module.base_kernel.lengthscale[0][4].item(),
            model.covar_module.outputscale.item(),
            model.likelihood.noise.item()))

        optimizer.step()

    return model, likelihood


# def create_ch2xy_4d(dim,sizes):
#     array=torch.zeros((torch.prod(torch.tensor(sizes)),dim),device=device)

#     count=0
#     for i in range(sizes[0]):
#         for j in range(sizes[1]):
#             for k in range(sizes[2]):
#                 for l in range(sizes[3]):
          
#                       array[count,0]=i+1
#                       array[count,1]=j+1
#                       array[count,2]=k+1
#                       array[count,3]=l+1

#                       count+=1          
#     return array


"""## **Prior creation** """

device='cpu'  # 'cuda' to use GPU\
data = scipy.io.loadmat('data/5d_rats_set/rCer1.5/ART_REJ/4x4x4x32x8_ar/4x4x4x32x8_ar.mat')['Data']

ch2xy = data[0][0][1][:,[0,1,2,5,6]]
response = data[0][0][0]

ch2xy = torch.from_numpy(ch2xy).float().to(device)
response = torch.from_numpy(response).float().to(device)

which_opt='kappa' 


this_opt=np.array([12.5])

n_subjects=4 # all 4 emgs 
n_cond=1 
n_dims=5

dim_sizes=np.array([8,4,4,4,4])
DimSearchSpace = np.prod(dim_sizes)

rho_low=0.1
rho_high=6.0
nrnd=5
noise_min=0.25
noise_max=10
MaxQueries =200
kappa=20
nRep=75
total_size= np.prod(dim_sizes)

#hyperparams = torch.zeros((n_subjects,n_cond,len(this_opt),nRep, MaxQueries,n_dims+2), device=device) # hyperparameters
# YMU = torch.zeros((n_subjects,n_cond, len(this_opt),nRep,MaxQueries,DimSearchSpace),device=device)  
# YVAR = torch.zeros((n_subjects,n_cond, len(this_opt),nRep,MaxQueries,DimSearchSpace),device=device)
PP = torch.zeros((n_subjects,n_cond,len(this_opt),nRep, MaxQueries), device=device)
PP_t = torch.zeros((n_subjects,n_cond, len(this_opt),nRep, MaxQueries), device=device)
# UCBMAP = torch.zeros((n_subjects,n_cond, len(this_opt),nRep,MaxQueries,DimSearchSpace),device=device)
Q = torch.zeros((n_subjects,n_cond,len(this_opt),nRep, MaxQueries), device=device)
# BQ = torch.zeros((n_subjects,n_cond,len(this_opt),nRep, MaxQueries), device=device)


for s_i in range(n_subjects): # for each subject

    print('subject '+ str(s_i))

    for c_i in range(n_cond):
        
        #print(c_i)
        
        # "Ground truth" map
        MPm= torch.mean(response[:, s_i], axis = 0)
        mMPm= torch.max(MPm)

        for k_i in range(len(this_opt)): # for each hyperparameter value 

            #print(which_opt + ' value :' + str(this_opt[k_i]))

            if which_opt=='nrnd':
                nrnd= this_opt[k_i]
            elif which_opt=='rho_low':
                rho_low= this_opt[k_i]
            elif which_opt=='rho_high':
                rho_high= this_opt[k_i]
            elif which_opt=='noise_min':
                noise_min= this_opt[k_i]
                noise_max= noise_min*1.1
            elif which_opt=='noise_max':
                noise_max= this_opt[k_i]
                #noise_min=0.0001
            elif which_opt=='kappa':
                kappa= this_opt[k_i]
            elif which_opt=='prior_scale':
                prior_scale=this_opt[k_i]
            elif which_opt=='prior_impl':
                prior_impl=this_opt[k_i]
            elif which_opt=='beta_coef':
                beta_coef=this_opt[k_i]        
            elif which_opt=='remove_prior_n':
                remove_prior_n=this_opt[k_i]
            elif which_opt=='noise_val':
                noise_val=this_opt[k_i]

            # Create kernel, likelihood and priors
            # Put a  prior on the two lengthscale hyperparameters, the variance and the noise
            #The lengthscale parameter is parameterized on a log scale to constrain it to be positive
            #The outputscale parameter is parameterized on a log scale to constrain it to be positive

            priorbox= gpytorch.priors.SmoothedBoxPrior(a=math.log(rho_low),b= math.log(rho_high), sigma=0.01) 
            priorbox2= gpytorch.priors.SmoothedBoxPrior(a=math.log(0.01**2),b= math.log(100.0**2), sigma=0.01) # std
            matk= gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims= n_dims, lengthscale_prior= priorbox) 
            matk_scaled = gpytorch.kernels.ScaleKernel(matk, outputscale_prior= priorbox2)
            matk_scaled.base_kernel.lengthscale= [1.0]*n_dims
            matk_scaled.outputscale= [1.0]

            prior_lik= gpytorch.priors.SmoothedBoxPrior(a=noise_min**2,b= noise_max**2, sigma=0.01) # gaussian noise variance
            likf= gpytorch.likelihoods.GaussianLikelihood(noise_prior= prior_lik)
            likf.noise= [1.0] 

            perf_explore= torch.zeros((nRep, MaxQueries), device=device)
            perf_exploit= torch.zeros((nRep, MaxQueries), device=device)
            perf_rsq= torch.zeros((nRep), device=device)
            P_test =  torch.zeros((nRep, MaxQueries, 2), device=device)
            P_max_all_temp= torch.zeros((nRep, MaxQueries), device=device)

            for rep_i in range(nRep): # for each repetition

                print('rep: ' + str(rep_i))
                
                #P_test =  torch.zeros((nRep, MaxQueries, 2), device=device) #storing all queries
                MaxSeenResp=0 
                q=0 # query number                                
                order_this= np.random.permutation(DimSearchSpace) # random permutation of each entry of the search space
                P_max=[]
                hyp=[1.0]*(n_dims+2)

                while q < MaxQueries: # MaxQueries:

                    if q>=nrnd:
                        # Find next point (max of acquisition function)
                        
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
                    else: 
                        # We will sample the search space randomly for exactly nrnd queries
                        P_test[rep_i][q][0]=int(order_this[q])
                        
                        #rint(int(order_this[q]))
                        
                    query_elec = P_test[rep_i][q][0]
                    
                    #print(int(query_elec.item()))

                    sample_resp = response[:, s_i, int(query_elec.item())]
                    test_respo = sample_resp[np.random.randint(len(sample_resp))]
                    
                    #print(test_respo)
                    #print(torch.mean(sample_resp))
                    test_respo += torch.normal(0.0, 0.02*torch.mean(sample_resp).item(), size=(1,), device=device).item() #, size=(1,)                   
                    
                    if test_respo < 0:
                        test_respo=torch.tensor([0.0001], device= device)

                    if test_respo==0 and q==0: # to avoid division by 0
                        test_respo= torch.tensor([0.0001], device=device)

                    
                    # done reading response
                    P_test[rep_i][q][1]= test_respo
                    # The first element of P_test is the selected search
                    # space point, the second the resulting value

                    y=(P_test[rep_i][:q+1,1]) 

                    if (torch.max(torch.abs(y)) > MaxSeenResp) or (MaxSeenResp==0):
                        # updated maximum response obtained in this round
                        MaxSeenResp=torch.max(torch.abs(y))

                    x= ch2xy[P_test[rep_i][:q+1,0].long(),:].float() # search space position
                    x = x.reshape((len(x),n_dims))

                    y=y/MaxSeenResp
                    y=y.float()
                    
                    if q ==0:        
                    # Update the initial value of the parameters  
                        matk_scaled.base_kernel.lengthscale= hyp[:n_dims]
                        matk_scaled.outputscale= hyp[n_dims]
                        likf.noise= hyp[n_dims+1]

                        
                    # Initialization of the model and the constraint of the Gaussian noise 
                    if q==0:

                      m= GP(x, y, likf, matk_scaled)

                      if device=='cuda':
                          m=m.cuda()
                          likf=likf.cuda()         
                    else:
                        # Update training data
                        m.set_train_data(x,y, strict=False)

                    m.train()
                    likf.train()
                    m, likf= optimize(m, likf, 10, x, y, verbose= False)

                    m.eval()
                    likf.eval()

                    with torch.no_grad():
                        X_test= ch2xy  
                        observed_pred = likf(m(X_test))

                    VarianceMap= observed_pred.variance
                    MapPrediction= observed_pred.mean

                    # We only test for gp predictions at electrodes that
                    # we had queried (presumable we only want to return an
                    # electrode that we have already queried). 
                    Tested= torch.unique(P_test[rep_i][:q+1,0]).long()
                    MapPredictionTested=MapPrediction[Tested]

                    if len(Tested)==1:
                        BestQuery=Tested
                    else:

                        BestQuery= Tested[(MapPredictionTested==torch.max(MapPredictionTested)).reshape(len(MapPredictionTested))]
                        if len(BestQuery) > 1:  
                            BestQuery = np.array([BestQuery[np.random.randint(len(BestQuery))].cpu()])

                    # Maximum response at time q 
                    P_max.append(BestQuery.item())
                    # store all info
                    #msr[s_i,c_i,k_i,rep_i,q] = MaxSeenResp
                    #YMU[s_i,c_i, k_i,rep_i,q,:]= MapPrediction

                    hyp= torch.tensor([m.covar_module.base_kernel.lengthscale[0][0].item(),
                                    m.covar_module.base_kernel.lengthscale[0][1].item(),
                                    m.covar_module.base_kernel.lengthscale[0][2].item(),
                                    m.covar_module.base_kernel.lengthscale[0][3].item(),
                                    m.covar_module.base_kernel.lengthscale[0][4].item(),
                                    m.covar_module.outputscale.item(),
                                    m.likelihood.noise[0].item()], device=device)

                    #hyperparams[s_i, c_i, k_i,rep_i,q,:] = hyp    

                    q+=1

                # BQ[s_i,c_i,k_i,rep_i]= torch.Tensor(P_max)

                # estimate current exploration performance: knowledge of best stimulation point    
                perf_explore[rep_i,:]=MPm[P_max].reshape((len(MPm[P_max])))/mMPm
                # estimate current exploitation performance: knowledge of best stimulation point 
                perf_exploit[rep_i,:]= P_test[rep_i][:,0].long()

            PP[s_i,c_i,k_i]=perf_explore
            Q[s_i,c_i,k_i] = P_test[:,:,0]
            PP_t[s_i,c_i,k_i]= MPm[perf_exploit.long().cpu()]/mMPm
          
    
np.savez('./output/vanilla_BO/rCer1.5/NOPRIOR_'+date.today().strftime("%y%m%d")+'_4channels_artRej_kappa20_lr001_5rnd.npz', PP=PP.cpu(), PP_t=PP_t.cpu(), Q=Q.cpu(),which_opt=which_opt, this_opt = this_opt, nrnd = nrnd, kappa = kappa)