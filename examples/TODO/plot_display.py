"""
???
"""
from examples.plot_computations import *

import seaborn as sns
import time
import matplotlib.pyplot as plt
import os

def plot_histograms(N=30, nb_simulations=10000, rho=1e-2, *, estimator,
                    compute=True, rho_tuning=True, blanchet=True):
    '''
    Plots Kuhn's histograms that were presented at the DTU CEE Summer School 2018.
    Setting rho=0 stands for the SAA method of resolution of the stochastic problem.
    '''

    start = time.time()

    filename, rho_filename = parallel_compute_histograms(N=N, nb_simulations=nb_simulations, 
                                            estimator=estimator, compute=compute,
                                            rho_tuning=rho_tuning, blanchet=blanchet)

    with open (filename, 'rb') as f:
        eval_data_train = np.load(f)
        eval_data_test = np.load(f)
        eval_data_adv_test = np.load(f)
    f.close()

    #Create an histogram that shows the values taken for the tuning of rho
    home_dir = os.path.expanduser("~")
    if rho_tuning is True:

        with open (rho_filename, 'rb') as f:
            tuned_rho_data = np.load(f)
        f.close()

        #Some adaptation rho_possible_values and nb_bins to rho tuning method is needed (to define in rho_tuners)
        rho_possible_values = [10**(-i) for i in range(4,-4,-1)]

        nb_bins = len(rho_possible_values)

        discrete = False if blanchet is True else True

        plt.figure()
        plt.xticks(rho_possible_values)
        plt.xscale("log")
        plt.xlabel("Rho values")
        plt.ylabel("Number of occurences")
        plt.title("Distribution of rho values taken when tuning rho")
        sns.histplot(data=tuned_rho_data, bins=nb_bins, stat="count", discrete=discrete)

        #Saving the histogram
        if blanchet is True:
            image_path = os.path.join(home_dir, "rho_values_blanchet.png")
        else:
            image_path = os.path.join(home_dir, "rho_values_GS.png")
        plt.savefig(image_path)
    
    #Create an histogram showing the performance of the model on train, test and adversarial data
    plt.figure()
    plt.xlabel("Mean-risk objective")
    if rho_tuning is False: 
        if rho == 0:
            plt.title("SAA Solution with scarce data (Kuhn 2017)")
        else:
            plt.title("WDRO Solution with scarce data (Kuhn 2017) with rho = %f" %rho)
    else:
        plt.title("WDRO Solution with scarce data (Kuhn 2017) with tuning of rho")
    end = time.time()
    sns.histplot(data=eval_data_train, bins=20, stat="probability", color="green", multiple="dodge", label="In-sample")
    sns.histplot(data=eval_data_test, bins=20, stat="probability", color="red", multiple="dodge", label="Out-of-sample")
    sns.histplot(data=eval_data_adv_test, bins=20, stat="probability", color="blue", multiple="dodge", label="Adversarial Out-of-sample")
    print("Simulations with histograms took ", end-start, " seconds")
    plt.legend()

    #Saving the histogram
    if rho_tuning is True:
        if blanchet is True:
            image_path = os.path.join(home_dir, "plot_histograms_blanchet.png")  
        else:
            image_path = os.path.join(home_dir, "plot_histograms_GS.png") 
    else:
        image_path = os.path.join(home_dir, "plot_histograms_no_tuning.png")
    plt.savefig(image_path)

    #Visualization
    plt.show()  

def plot_curves(nb_simulations=200, *, estimator, compute=True):
    '''
    Plots Kuhn's curves from Section 7.2 of the 2017 WDRO paper.
    '''
    start = time.time()

    samples_size, filename = parallel_compute_curves(nb_simulations, estimator, compute)

    with open (filename, 'rb') as f:
        rho_values = np.load(f)

        for size in samples_size:

            mean_eval_data_test = np.load(f)
            reliability_test = np.load(f)

            _, ax1 = plt.subplots(num=size)

            ax1.grid(True)

            ax1.set_xlabel("Wasserstein radius")
            ax1.set_ylabel("Out-of-sample performance")
            ax1.set_title("Impact of the Wasserstein Radius (Kuhn 2017) for N = %i" %size)
            ax1.set_xticks(rho_values)
            ax1.set_xscale("log")
            ax1.plot(rho_values, mean_eval_data_test, color='blue', label="Out-of-sample loss")

            ax2 = ax1.twinx()
            ax2.set_ylabel("Reliability")
            ax2.plot(rho_values, reliability_test, linestyle='dashed', color='red', label="Reliability")
            ax2.legend()

            #Saving the curves
            home_dir = os.path.expanduser("~")
            png_name = "plot_curves_" + str(size) + ".png"
            image_path = os.path.join(home_dir, png_name)
            plt.savefig(image_path)

            if size == samples_size[-1]:
                end = time.time()
                print("Simulations with curves took ", end-start, " seconds")

        #Visualization 
        plt.show() #Show all three figures at once 
    
    f.close()

def main():
    N = 30 #Size of samples for Kuhn's histograms
    estimator_solver = "entropic_torch_post"

    n_zeta_samples = 0 if estimator_solver not in \
        {"entropic", "entropic_torch", "entropic_torch_pre", "entropic_torch_post"} else 10*N

    #Create the estimator and solve the problem
    #estimator = Portfolio(solver=estimator_solver, cost="t-NC-1-1", reparam="softmax", alpha=ALPHA, eta=ETA, n_zeta_samples=n_zeta_samples)
    #estimator = LogisticRegression(solver=estimator_solver, n_zeta_samples=n_zeta_samples)
    #estimator = NewsVendor(solver=estimator_solver, cost="t-NC-1-1", n_zeta_samples=n_zeta_samples)
    estimator = LinearRegression(solver=estimator_solver, n_zeta_samples=n_zeta_samples)
    #estimator = Weber(solver=estimator_solver, cost="t-NC-1-1", n_zeta_samples=n_zeta_samples)

    #plot_histograms(rho=0, compute=True)
    plot_histograms(nb_simulations=20, compute=True, estimator=estimator, rho_tuning=True, blanchet=True)
    #plot_curves(estimator=estimator, compute=True)


if __name__ == "__main__":
    main()
