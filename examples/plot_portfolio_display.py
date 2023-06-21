from examples.plot_portfolio_computations import *

import seaborn as sns
import time
import matplotlib.pyplot as plt

def plot_histograms(N=30, nb_simulations=10000, rho=1e-2, adv=1e-2, estimator_solver="dedicated", compute=True):
    '''
    Plots Kuhn's histograms that were presented at the DTU CEE Summer School 2018.
    Setting rho=0 stands for the SAA method of resolution of the stochastic problem.
    '''

    start = time.time()

    filename = parallel_compute_histograms(N=N, nb_simulations=nb_simulations, rho=rho, estimator_solver=estimator_solver, adv=adv, compute=compute)

    with open (filename, 'rb') as f:
        eval_data_train = np.load(f)
        eval_data_test = np.load(f)
        eval_data_adv_test = np.load(f)
    f.close()
    
    #Create the histograms
    plt.xlabel("Mean-risk objective")
    if rho == 0:
        plt.title("SAA Solution with scarce data (Kuhn 2017)")
    else:
        plt.title("DRO Solution with scarce data (Kuhn 2017) with rho = %f" %rho)
    end = time.time()
    sns.histplot(data=eval_data_train, bins=20, stat="probability", color="green", multiple="dodge", label="In-sample")
    sns.histplot(data=eval_data_test, bins=20, stat="probability", color="red", multiple="dodge", label="Out-of-sample")
    sns.histplot(data=eval_data_adv_test, bins=20, stat="probability", color="blue", multiple="dodge", label="Adversarial Out-of-sample")
    print("Simulations with histograms took ", end-start, " seconds")
    plt.legend()
    plt.show()  

def plot_curves(nb_simulations=200, estimator_solver="dedicated", compute=True):
    '''
    Plots Kuhn's curves from Section 7.2 of the 2017 WDRO paper.
    '''
    start = time.time()

    samples_size, filename = parallel_compute_curves(nb_simulations, estimator_solver, compute)

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
    
    f.close()
    end = time.time()
    print("Simulations with curves took ", end-start, " seconds")
    plt.show() #Show all three figures at once 

def main():
    N = 30 #Size of samples for Kuhn's histograms

    #plot_histograms(rho=0, adv=1/np.sqrt(N), estimator_solver="entropic_torch_pre", compute=True)
    plot_histograms(rho=1/np.sqrt(N), adv=1/np.sqrt(N), estimator_solver="entropic_torch_post", compute=True)
    #plot_curves(estimator_solver="entropic_torch_pre", compute=True)

if __name__ == "__main__":
    main()