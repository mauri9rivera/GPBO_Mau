import scipy.stats as stats
import torch
import gpytorch
import os
from gp import *
import scipy.io

np.random.seed(0)
torch.manual_seed(0)

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

def define_opt_param(params, chosen_opt, this_opt, k_index):

    rho_low, rho_high, nrnd, noise_min, noise_max, kappa = params

    match chosen_opt:
        case 'nrnd':
            nrnd = this_opt[k_index]
        case 'rho_low':
            rho_low = this_opt[k_index]
        case 'rho_high':
            rho_high= this_opt[k_index]
        case 'noise_min':
            noise_min= this_opt[k_index]
            noise_max= noise_min*1.1
        case 'noise_max':
            noise_max= this_opt[k_index]
            #noise_min=0.0001
        case'kappa':
            kappa= this_opt[k_index]

def run_rpm_bo_experiment():

    #Here, we define which graphs/values we want to keep track
    PP = torch.zeros((N_SUBJECTS,N_COND,len(THIS_OPT),NREP, MAXQUERIES), device=DEVICE)
    PP_t = torch.zeros((N_SUBJECTS, N_COND,len(THIS_OPT),NREP, MAXQUERIES), device=DEVICE)

    for s_i in range(N_SUBJECTS):

        print(f'Subject {s_i}')

        for c_i in range(N_COND):

            print(c_i)

            # "Ground truth" map
            MPm= torch.mean(response[:, s_i], axis = 0)
            mMPm= torch.max(MPm)

            for k_i in range(len(THIS_OPT)):
                params = rho_low, rho_high, nrnd, noise_min, noise_max, kappa
                rho_low, rho_high, nrnd, noise_min, noise_max, kappa = define_opt_param(params, WHICH_OPT, k_i)
                print(f'WHICH_OPT value : {THIS_OPT[0]}') 

                #Creating krnel. likelihood and priors
                priorbox= gpytorch.priors.SmoothedBoxPrior(a=math.log(RHO_LOW),b= math.log(RHO_HIGH), sigma=0.01) 
                priorbox2= gpytorch.priors.SmoothedBoxPrior(a=math.log(0.01**2),b= math.log(100.0**2), sigma=0.01) # std
                matk= gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims= N_DIMS, lengthscale_prior= priorbox) 
                matk_scaled = gpytorch.kernels.ScaleKernel(matk, outputscale_prior= priorbox2)
                matk_scaled.base_kernel.lengthscale= [1.0]*N_DIMS
                matk_scaled.outputscale= [1.0]

                prior_lik= gpytorch.priors.SmoothedBoxPrior(a=NOISE_MIN**2,b= NOISE_MAX**2, sigma=0.01) # gaussian noise variance
                likf= gpytorch.likelihoods.GaussianLikelihood(noise_prior= prior_lik)
                likf.noise= [1.0] 

                # Running values for graphing
                perf_explore= torch.zeros((NREP, MAXQUERIES), device=DEVICE)
                perf_exploit= torch.zeros((NREP, MAXQUERIES), device=DEVICE)
                perf_rsq= torch.zeros((NREP), device=DEVICE)
                P_test =  torch.zeros((NREP, MAXQUERIES, 2), device=DEVICE)
                P_max_all_temp= torch.zeros((NREP, MAXQUERIES), device=DEVICE)


                ### What is different
                n_dims_og = np.prod(DIM_SIZES[:-1])
                n_dims_emb = n_dims_og / 2
                proposed_model = ProjectionGP(x, y, likf, matk_scaled, n_dims_og, n_dims_emb, 35, DEVICE)
                
                # TODO: Do procedure that queries n random sample points
                """
                # TODO: Make the random initialization its own function so it can be done separately from the acquisition argmin
                # Initialize with random points
                if len(self.X) < self.initial_random_samples:
                    if self.initial_points_list is None:
                    # Select query point randomly from embedding_boundaries
                        X_query_embedded = \
                            self.rng.uniform(size=self.embedding_boundaries.shape[0]) \
                            * (self.embedding_boundaries[:, 1] - self.embedding_boundaries[:, 0]) \
                            + self.embedding_boundaries[:, 0]
                        X_query_embedded = torch.from_numpy(X_query_embedded).unsqueeze(0)
                        print("X_query_embedded.shape: {}".format(X_query_embedded.shape))
                    else:
                        #self.X = torch.cat((self.X, torch.Tensor(self.initial_points_list[len(self.X)]))).double()
                        X_query = torch.tensor(self.initial_points_list[len(self.X)], dtype=self.dtype, device=self.device)
                        X_query_embedded = self.projection_to_manifold(X_query) @ self.A.T
                        return X_query, X_query_embedded

                """           


                for rep_i in range(NREP):

                    print(f'Repetition {rep_i}')

                    X_queries, X_queries_embedded = proposed_model.select_query_point(MAXQUERIES)


                #### What is not different




                    MaxSeenResp=0 
                    q=0 # query number                                
                    order_this= np.random.permutation(DimSearchSpace) # random permutation of each entry of the search space
                    P_max=[]
                    hyp=[1.0]*(N_DIMS+2) 

                    #Some call to a function of models
                    while q < MAXQUERIES:



                        if q >=NRND:

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
                        m, likf= optimize_vanilla(m, likf, 10, x, y, verbose= False)

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
                PP_t[s_i,c_i,k_i]= MPm[perf_exploit.long().cpu()]/mMPm 

    
    return PP, PP_t

def discovering_additive_struct():
    
    #Here, we define which graphs/values we want to keep track
    PP = torch.zeros((N_SUBJECTS,N_COND,len(THIS_OPT),NREP, MAXQUERIES), device=DEVICE)
    PP_t = torch.zeros((N_SUBJECTS, N_COND,len(THIS_OPT),NREP, MAXQUERIES), device=DEVICE)
    Q = torch.zeros((N_SUBJECTS, N_COND,len(THIS_OPT),NREP, MAXQUERIES), device=DEVICE)


    for s_i in range(N_SUBJECTS):

        print(f'Subject {s_i}')

        for c_i in range(N_COND):


            # "Ground truth" map
            MPm= torch.mean(response[:, s_i], axis = 0)
            mMPm= torch.max(MPm)

            for k_i in range(len(THIS_OPT)):
                #params =RHO_LOW, RHO_HIGH, NRND, NOISE_MIN, NOISE_MAX, KAPPA
                #RHO_LOW, RHO_HIGH, NRND, NOISE_MIN, NOISE_MAX, KAPPA = define_opt_param(params, WHICH_OPT, k_i)
                #print(f'WHICH_OPT value : {THIS_OPT[0]}') 

                #Creating krnel. likelihood and priors
                priorbox= gpytorch.priors.SmoothedBoxPrior(a=math.log(RHO_LOW),b= math.log(RHO_HIGH), sigma=0.01) 
                priorbox2= gpytorch.priors.SmoothedBoxPrior(a=math.log(0.01**2),b= math.log(100.0**2), sigma=0.01) # std
                matk= gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims= N_DIMS, lengthscale_prior= priorbox) 
                matk_scaled = gpytorch.kernels.ScaleKernel(matk, outputscale_prior= priorbox2)
                matk_scaled.base_kernel.lengthscale= [1.0]*N_DIMS
                matk_scaled.outputscale= [1.0]

                prior_lik= gpytorch.priors.SmoothedBoxPrior(a=NOISE_MIN**2,b= NOISE_MAX**2, sigma=0.01) # gaussian noise variance
                likf= gpytorch.likelihoods.GaussianLikelihood(noise_prior= prior_lik)
                likf.noise= [1.0] 

                # Running values for graphing
                perf_explore= torch.zeros((NREP, MAXQUERIES), device=DEVICE)
                perf_exploit= torch.zeros((NREP, MAXQUERIES), device=DEVICE)
                perf_rsq= torch.zeros((NREP), device=DEVICE)
                P_test =  torch.zeros((NREP, MAXQUERIES, 2), device=DEVICE)
                P_max_all_temp= torch.zeros((NREP, MAXQUERIES), device=DEVICE)
          

                for rep_i in range(NREP):

                    MaxSeenResp=0 
                    q=0 # query number                                
                    order_this= np.random.permutation(DimSearchSpace) # random permutation of each entry of the search space
                    P_max=[]
                    hyp=[1.0]*(N_DIMS+2) 

                    print('rep: ' + str(rep_i))

                    #Some call to a function of models
                    while q < MAXQUERIES:

                        if q >=NRND:

                            #Find the next point (max of acquisition function)
                            

                            if torch.isnan(MapPrediction).any():
                                print('nan in Mean map pred')
                                MapPrediction = torch.nan_to_num(MapPrediction)

                            AcquisitionMap = MapPrediction + KAPPA*torch.nan_to_num(torch.sqrt(VarianceMap)) # UCB acquisition
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
                            
                            
                            
                        query_elec = P_test[rep_i][q][0]

                        sample_resp = response[:, s_i, int(query_elec.item())]
                        test_respo = sample_resp[np.random.randint(len(sample_resp))]
                        test_respo += torch.normal(0.0, 0.02*torch.mean(sample_resp).item(), size=(1,), device=DEVICE).item() #, size=(1,)                   
                        
                        if test_respo < 0:
                            test_respo=torch.tensor([0.0001], device= DEVICE)

                        if test_respo==0 and q==0: # to avoid division by 0
                            test_respo= torch.tensor([0.0001], device=DEVICE)

                        
                        # done reading response
                        P_test[rep_i][q][1]= test_respo
                        # The first element of P_test is the selected search
                        # space point, the second the resulting value

                        y=(P_test[rep_i][:q+1,1]) 

                        if (torch.max(torch.abs(y)) > MaxSeenResp) or (MaxSeenResp==0):
                            # updated maximum response obtained in this round
                            MaxSeenResp=torch.max(torch.abs(y))

                        x= ch2xy[P_test[rep_i][:q+1,0].long(),:].float() # search space position
                        x = x.reshape((len(x),N_DIMS))

                        y=y/MaxSeenResp
                        y=y.float()
                        
                        if q ==0:        
                        # Update the initial value of the parameters  
                            matk_scaled.base_kernel.lengthscale= hyp[:N_DIMS]
                            matk_scaled.outputscale= hyp[N_DIMS]
                            likf.noise= hyp[N_DIMS+1]

                            
                        # Initialization of the model and the constraint of the Gaussian noise 
                        if q==0:

                            m= GP(x, y, likf, matk_scaled)

                        if DEVICE=='cuda':
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
                                        m.likelihood.noise[0].item()], device=DEVICE)

                        #hyperparams[s_i, c_i, k_i,rep_i,q,:] = hyp    

                        q+=1
                    
                    # BQ[s_i,c_i,k_i,rep_i]= torch.Tensor(P_max)

                    # estimate current exploration performance: knowledge of best stimulation point    
                    perf_explore[rep_i,:]=MPm[P_max].reshape((len(MPm[P_max])))/mMPm
                    # estimate current exploitation performance: knowledge of best stimulation point 
                    perf_exploit[rep_i,:]= P_test[rep_i][:,0].long()

                PP[s_i,c_i,k_i]=perf_explore 
                PP_t[s_i,c_i,k_i]= MPm[perf_exploit.long().cpu()]/mMPm
                Q[s_i,c_i,k_i] = P_test[:,:,0] 

    
    return PP, PP_t, Q




DEVICE = 'cpu'
DATA = scipy.io.loadmat('data/5d_rats_set/rCer1.5/ART_REJ/4x4x4x32x8_ar/4x4x4x32x8_ar.mat')['Data']
ch2xy = DATA[0][0][1][:,[0,1,2,5,6]]
response = DATA[0][0][0]

ch2xy = torch.from_numpy(ch2xy).float().to(DEVICE)
response = torch.from_numpy(response).float().to(DEVICE)

WHICH_OPT = 'kappa'
THIS_OPT = np.array([12.5])
N_SUBJECTS = 4
N_COND = 1
N_DIMS = 5
DIM_SIZES = np.array([8,4,4,4,4])
DimSearchSpace = np.prod(DIM_SIZES)

RHO_LOW = 0.1
RHO_HIGH=6.0
NRND=5
NOISE_MIN=0.25
NOISE_MAX=10
MAXQUERIES=200
KAPPA=20
NREP=10
total_size = np.prod(DIM_SIZES)

PP, PP_t, Q = discovering_additive_struct()
# Make sure filepath exists
np.savez('./output/RPM_BO/rCer1.5/NOPRIOR_'+date.today().strftime("%y%m%d")+'_4channels_artRej_kappa20_lr001_5rnd.npz', PP=PP.cpu(), PP_t=PP_t.cpu(), Q = Q.cpu(), which_opt=WHICH_OPT, this_opt = THIS_OPT, nrnd = NRND, kappa = THIS_OPT[0])




