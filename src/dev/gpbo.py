import torch
import numpy as np
import gpytorch
from datetime import date, datetime
from time import time
import scipy.io
from scipy.stats.mstats import gmean
import os
from model import GP, optimize
from viz import (
    plot_comparison_per_strat,
    plot_mean_scores_over_subjects,
    plot_scores_per_emg,
    plot_tradeoff,
)
from strategy import strategy
from synthetic_data import create_michalewicz
 

def train(
    ch2xy,
    response,
    inits,
    all_init_steps=None,
    kappa: int = 20,
    nb_epochs: int = 10,
    training_iter: int = 10,
    nb_queries: int = 300,
    max_query: int = 100,
    nb_init_queries: int = 5,
    device=torch.device("cpu"),
    strat: str = "GPBO",
    output_dir="../data/output/",
):
    n_emgs = response.shape[1]
    n_dims = ch2xy.shape[1]
    dim_input_space = ch2xy.shape[0]
    probability_map = None

    exploration_score = torch.zeros((n_emgs, nb_epochs, nb_queries), device=device)
    exploitation_score = torch.zeros((n_emgs, nb_epochs, nb_queries), device=device)

    # Monitor iterations
    steps = torch.zeros((n_emgs, nb_epochs, nb_queries, 2), device=device)

    for emg in range(n_emgs):  # for each subject
        print("emg " + str(emg))

        # "Ground truth" map
        ground_truth = torch.mean(response[:, emg], axis=0)
        max_ground_truth = torch.max(ground_truth)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        perf_explore = torch.zeros((nb_epochs, nb_queries), device=device)
        perf_exploit = torch.zeros((nb_epochs, nb_queries), device=device)
        optim_steps = torch.zeros((nb_epochs, nb_queries, 2), device=device)

        for epoch in range(nb_epochs):  # for each repetition
            start = time()
            print("rep: " + str(epoch))
            deleted_queries = []
            max_seen_resp = 0
            query = 0  # query number
            init_queries = inits[
                epoch
            ]  # random permutation of each entry of the search space
            best_queries_list = []

            while query < nb_queries:  # MaxQueries:

                if query >= nb_init_queries:
                    # Find next point (max of acquisition function)

                    if torch.isnan(mu).any():
                        print("nan in Mean map pred")
                        mu = torch.nan_to_num(mu)

                    acquisition = mu + kappa * torch.nan_to_num(
                        torch.sqrt(sigma)
                    )  # UCB acquisition
                    next_query = torch.where(
                        acquisition
                        == torch.max(acquisition)
                    )

                    if len(next_query[0]) > 1:
                        if query >= max_query:
                            next_query = next_query[0][
                                np.random.randint(len(next_query[0]))
                            ]
                        else:
                            next_query = next_query[0][0]
                    else:
                        next_query = next_query[0][0]
                    optim_steps[epoch, query, 0] = next_query
                else:
                    # We will sample the search space randomly for exactly nrnd queries
                    if all_init_steps is not None:
                        optim_steps[epoch, query, 0] = all_init_steps[
                            emg, epoch, query, 0
                        ]
                        optim_steps[epoch, query, 1] = all_init_steps[
                            emg, epoch, query, 1
                        ]
                    else:
                        optim_steps[epoch, query, 0] = int(init_queries[query])

                query_elec = optim_steps[epoch, query, 0]

                sample_resp = response[:, emg, int(query_elec.item())]

                if query >= max_query:
                    test_respo = sample_resp[np.random.randint(len(sample_resp))]
                else:
                    test_respo = sample_resp[0]

                #test_respo += torch.normal(
                #     0.0,
                #     0.02 * torch.abs(torch.mean(sample_resp)).item(),
                #     size=(1,),
                #     device=device,
                #).item()  # , size=(1,)

                if test_respo < 0:
                    test_respo = torch.tensor([0.0001], device=device)

                if test_respo == 0 and query == 0:  # to avoid division by 0
                    test_respo = torch.tensor([0.0001], device=device)

                # done reading response
                if optim_steps[epoch, query, 1] == 0:
                    optim_steps[epoch][query][1] = test_respo
                # The first element of P_test is the selected search
                # space point, the second the resulting value

                if query >= max_query:
                    deleted_query, probability_map = strategy(
                        model,
                        likelihood,
                        ch2xy,
                        optim_steps,
                        query,
                        deleted_queries,
                        mu,
                        sigma,
                        max_query,
                        epoch,
                        kind=strat,
                        probability_deletion_map=probability_map,
                    )
                    if deleted_query is not None:
                        deleted_queries.append(deleted_query)
        
                keeped_queries = np.delete(np.arange(0, query + 1, 1), deleted_queries)


                train_y = optim_steps[epoch, keeped_queries, 1]

                if (torch.max(torch.abs(train_y)) > max_seen_resp) or (
                    max_seen_resp == 0
                ):
                    max_seen_resp = torch.max(torch.abs(train_y))

                train_y /= max_seen_resp
                train_y = train_y.float()

                train_x = ch2xy[
                    optim_steps[
                        epoch,
                        keeped_queries,
                        0,
                    ].long(),
                    :,
                ].float()

                # Initialization of the model and the constraint of the Gaussian noise
                if query == 0:

                    model = GP(train_x, train_y, likelihood)
                    model = model.to(device)
                    likelihood = likelihood.to(device)
                else:
                    # Update training data
                    model.set_train_data(
                        train_x,
                        train_y,
                        strict=False,
                    )

                model.train()
                likelihood.train()
                model, likelihood = optimize(
                    model,
                    likelihood,
                    training_iter,
                    train_x,
                    train_y,
                    verbose=False,
                )

                model.eval()
                likelihood.eval()

                with torch.no_grad():
                    test_x = ch2xy
                    observed_pred = likelihood(model(test_x))

                sigma = observed_pred.variance
                mu = observed_pred.mean

                # We only test for gp predictions at electrodes that
                # we had queried (presumable we only want to return an
                # electrode that we have already queried).
                tested = torch.unique(optim_steps[epoch, : query + 1, 0]).long()
                mu_tested = mu[tested]

                if len(tested) == 1:
                    best_query = tested
                else:
                    best_query = tested[
                        (mu_tested == torch.max(mu_tested)).reshape(len(mu_tested))
                    ]
                    if len(best_query) > 1:
                        if query >= max_query:
                            best_query = np.array(
                                [best_query[np.random.randint(len(best_query))].cpu()]
                            )
                        else:
                            best_query = np.array([best_query[0].cpu()])

                # Maximum response at time q
                best_queries_list.append(best_query.item())

                steps[emg, epoch, query, 0] = query_elec
                steps[emg, epoch, query, 1] = test_respo
                query += 1

            # print("Different queries: ", len(tested))
            # estimate current exploration performance: knowledge of best stimulation point
            perf_explore[epoch, :] = (
                ground_truth[best_queries_list].reshape(
                    (len(ground_truth[best_queries_list]))
                )
                / max_ground_truth
            )
            # estimate current exploitation performance: knowledge of best stimulation point
            perf_exploit[epoch, :] = optim_steps[epoch, :, 0].long()
            print("Time for rep: ", time() - start)
    
        exploration_score[emg] = perf_explore
        exploitation_score[emg] = (
            ground_truth[perf_exploit.long().cpu()] / max_ground_truth
        )
    os.chdir(output_dir)
    np.savez(
        f"run_s{strat}_k{kappa}_{date.today().strftime('%y%m%d')}.npz",
        explr=exploration_score.cpu(),
        explt=exploitation_score.cpu(),
        nb_init_queries=nb_init_queries,
        kappa=kappa,
        strat=strat,
        nb_queries=nb_queries,
        nb_epochs=nb_epochs,
        max_query=max_query,
        nb_emgs=n_emgs,
        steps=steps.cpu(),
    )
    return steps


def create_folder(output_dir, data_name):
    current_date_time_day = date.today().strftime("%y%m%d")
    if not os.path.exists(f"{output_dir}/GPBO_{current_date_time_day}_{data_name}"):
        os.makedirs(f"{output_dir}/GPBO_{current_date_time_day}_{data_name}")

    os.chdir(f"{output_dir}/GPBO_{current_date_time_day}_{data_name}")
    current_date_time_hour = datetime.now().strftime("%Y-%m-%d_%Hh-%Mmin-%Ss")
    dir_of_the_run = f"{output_dir}/GPBO_{current_date_time_day}_{data_name}/run_GPBO_{current_date_time_hour}"
    if not os.path.exists(f"run_GPBO_{current_date_time_hour}"):
        os.makedirs(f"run_GPBO_{current_date_time_hour}")
    return dir_of_the_run

def optimize_kappa_strategy(
    ch2xy,
    response,
    inits,
    kappas_list,
    strats,
    nb_epochs=10,
    nb_queries=300,
    max_query=100,
    nb_init_queries=5,
    training_iter=10,
    device=torch.device("cpu"),
    output_dir="../data/outputs",
    data_name="",
):
    dir_of_the_run = create_folder(output_dir, data_name)

    for kappa in kappas_list:
        print(f"Kappa: {kappa}")

        all_init_steps = train(
            ch2xy,
            response,
            inits=inits,
            kappa=kappa,
            nb_epochs=nb_epochs,
            training_iter=training_iter,
            nb_queries=nb_queries,
            max_query=max_query,
            nb_init_queries=nb_init_queries,
            device=device,
            strat="GPBO",
            output_dir=dir_of_the_run,
        )


        if len(strats) != 0:
            all_init_steps = all_init_steps[:, :, :max_query, :]
            nb_init_queries = max_query

            for strat in strats:
                print(f"Strat: {strat}")
                train(
                    ch2xy,
                    response,
                    inits=inits,
                    all_init_steps=all_init_steps,
                    kappa=kappa,
                    nb_epochs=nb_epochs,
                    training_iter=training_iter,
                    nb_queries=nb_queries,
                    max_query=max_query,
                    nb_init_queries=nb_init_queries,
                    device=device,
                    strat=strat,
                    output_dir=dir_of_the_run,
                )
    return dir_of_the_run


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    np.random.seed(42)
    torch.manual_seed(42)

    # # Load data
    # data = scipy.io.loadmat(
    #     "data/inputs/rat/rCer1.5/ART_REJ/4x4x4x32x8_ar/4x4x4x32x8_ar.mat"
    # )["Data"]

    # data_name = "rCer1.5"

    # ch2xy = data[0][0][1][:, [0, 1, 2, 5, 6]]
    # response = data[0][0][0][:, [0], :]
    # print(ch2xy.shape, response.shape)


    data_name = "michalewicz_2d"
    ch2xy, responses = create_michalewicz(32, 2)
    ch2xy = torch.from_numpy(ch2xy).float().to(device)
    response = torch.from_numpy(responses).float().to(device)[:, 0, :].unsqueeze(1)

    nb_epochs = 30
    nb_init_queries = 1
    nb_queries = 100
    kappa = 5
    time_per_epoch = 0

    for max_query in [5, 10, 12, 15, 17, 20, 25, 30, 35, 40, 50, 60]:
        print(f"Max query: {max_query}")
        inits = np.random.randint(0, ch2xy.shape[0], size=(nb_epochs, nb_init_queries))
        dir = create_folder("C:\\Users\\Maxime\\Documents\\data\\outputs", data_name)

        # dir = optimize_kappa_strategy(
        #     ch2xy,
        #     response,
        #     inits,
        #     [kappa],
        #     ["Rand", "Rand2", "GeoMean", "Mean"],
        #     nb_epochs=nb_epochs,
        #     nb_queries=nb_queries,
        #     max_query=max_query,
        #     nb_init_queries=nb_init_queries,
        #     training_iter=10,
        #     device=device,
        #     output_dir="C:\\Users\\Maxime\\Documents\\data\\outputs",
        #     data_name=data_name,
        # )

        if max_query == 5:
            all_init_steps = train(
                ch2xy,
                response,
                inits=inits,
                kappa=kappa,
                nb_epochs=nb_epochs,
                training_iter=10,
                nb_queries=nb_queries,
                max_query=max_query,
                nb_init_queries=nb_init_queries,
                device=device,
                strat="GPBO",
                output_dir=dir,
            )
        all_init_steps_loop = all_init_steps[:, :, :max_query, :]
        nb_init_queries = max_query

        train(
            ch2xy,
            response,
            inits=inits,
            all_init_steps=all_init_steps_loop,
            kappa=kappa,
            nb_epochs=nb_epochs,
            training_iter=10,
            nb_queries=nb_queries,
            max_query=max_query,
            nb_init_queries=nb_init_queries,
            device=device,
            strat="Rand",
            output_dir=dir,
        )

    
        os.chdir(dir)
        os.mkdir("plots")
        os.mkdir("plots/mean")
        os.mkdir("plots/comparison")
        os.mkdir("plots/emg")
        #os.mkdir("plots/tradeoff")

        for file in os.listdir(dir):
            if not os.path.isdir(file):
                file = np.load(file)
                plot_mean_scores_over_subjects(file, output_dir=f"{dir}/plots/mean")
                plot_scores_per_emg(file, output_dir=f"{dir}/plots/emg")

        plot_comparison_per_strat(dir, output_dir=f"{dir}/plots/comparison")
        #plot_tradeoff(dir, output_dir=f"{dir}/plots/tradeoff")
