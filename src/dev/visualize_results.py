import matplotlib.pyplot as plt
import numpy as np

def load_results(filepath):

    data = np.load(filepath)
    return data

def plot_PPs(data):

    # Extract PP_t and PP arrays
    PP_t = data['PP_t']  # Replace 'PP_t' with the actual key in the .npz file
    PP = data['PP']      # Replace 'PP' with the actual key in the .npz file

    n_subjects = PP_t.shape[0]
    fig, axs = plt.subplots(1, n_subjects, figsize=(15, 12))

    for i in range(n_subjects):

        mean_PP_t = PP_t[i].mean(axis=2).squeeze()
        std_PP_t = PP_t[i].std(axis=2).squeeze()
        x_values = np.arange(mean_PP_t.shape[0])
        
        mean_PP = PP[i].mean(axis=2).squeeze()
        std_PP = PP[i].std(axis=2).squeeze()
        
        axs[i].plot(mean_PP_t, label=f'PP_t', color='blue')
        axs[i].plot(mean_PP, label=f'PP', color='red')
        #axs[i].fill_between(x_values, mean_PP_t - 1* std_PP_t, mean_PP_t + 1* std_PP_t,
        #    color="skyblue", alpha=0.2, label="95% Confidence Interval")
        #axs[i].fill_between(x_values, mean_PP - 1* std_PP_t, mean_PP_t + 1* std_PP_t,
        #         color="lightsalmon", alpha=0.2, label="95% Confidence Interval")
        
        axs[i].set_title(f"Exploration vs. Exploitation for subject {i}")
        
        
    plt.legend(["Exploitation", "Exploration"], loc="lower right")
    plt.tight_layout()
    plt.show()

def plot_Q(data):

# Extract PP_t and PP arrays
    Q = data['Q']

    n_subjects = Q.shape[0]
    fig, axs = plt.subplots(1, n_subjects, figsize=(15, 12))


    for i in range(n_subjects):

        mean_Q = Q[i].mean(axis=2).squeeze()
        std_Q = Q[i].std(axis=2).squeeze()
        x_values = np.arange(mean_Q.shape[0])

        axs[i].plot(mean_Q, label=f'Q', color='orange')
        axs[i].fill_between(x_values, mean_Q - std_Q, mean_Q + std_Q,
            color="moccasin", alpha=0.2, label="95% Confidence Interval")
        
        axs[i].set_title(f"Q for subject {i}")
        

    plt.tight_layout()
    plt.legend(["Q"], loc="lower right")
    plt.show()

    

if __name__ == '__main__':

    filepath = './output/vanilla_BO/rCer1.5/NOPRIOR_241028_4channels_artRej_kappa20_lr001_5rnd.npz'
    #filepath = './output/RPM_BO/rCer1.5/NOPRIOR_241028_4channels_artRej_kappa20_lr001_5rnd.npz'
    data = load_results(filepath)
    #TODO: Figure out how to fix confidence intervals
    #plot_Q(data)
    plot_PPs(data)
