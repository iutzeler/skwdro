from examples.plot_portfolio_computations import *

import time
import matplotlib.pyplot as plt

def plot_histograms(N=30, nb_simulations=10000, rho=1e-2, compute=True):

    compute_histograms(N, nb_simulations, rho, compute)

    with open ('./examples/stored_data/portfolio_histogram_data.npy', 'rb') as f:
        eval_data_train = np.load(f)
        eval_data_test = np.load(f)
    f.close()

    #Create the histograms
    plt.xlabel("Mean-risk objective")
    plt.ylabel("Probability")
    if rho == 0:
        plt.title("SAA Solution with scarce data (Kuhn 2017)")
    else:
        plt.title("DRO Solution with scarce data (Kuhn 2017) with rho = %f" %rho)
    plt.hist(eval_data_train, bins=20, density=True, histtype="bar", color="green")
    plt.hist(eval_data_test, bins=20, density=True, histtype="bar", color="red")
    plt.show()

def parallel_plot_histograms(N=30, nb_simulations=10000, rho=1e-2, compute=True):

    start = time.time()

    print("Before parallel computations")

    with open ('./examples/stored_data/portfolio_histogram_data.npy', 'rb') as f:
        eval_data_train = np.load(f)
        eval_data_test = np.load(f)
    f.close()

    print("After parallel computations")
    
    #Create the histograms
    plt.xlabel("Mean-risk objective")
    plt.ylabel("Probability")
    if rho == 0:
        plt.title("SAA Solution with scarce data (Kuhn 2017)")
    else:
        plt.title("DRO Solution with scarce data (Kuhn 2017) with rho = %f" %rho)
    plt.hist(eval_data_train, bins=20, normed=True, density=True, histtype="bar", color="green")
    plt.hist(eval_data_test, bins=20, normed=True, density=True, histtype="bar", color="red")
    end = time.time()
    print("Simulations with histograms took ", end-start, " seconds")
    plt.show()  

def plot_curves(nb_simulations=200, compute=True):
    start = time.time()

    samples_size = compute_curves(nb_simulations, compute)

    with open ('./examples/stored_data/portfolio_curves_data.npy', 'rb') as f:
        rho_values = np.load(f)

        for size in samples_size:

            mean_eval_data_test = np.load(f)

            #Create the curves
            plt.figure(size)
            plt.xlabel("Wasserstein radius")
            plt.ylabel("Out-of-sample performance")
            plt.title("Impact of the Wasserstein Radius (Kuhn 2017) for N = %i" %size)
            plt.xticks(rho_values)
            plt.xscale("log")
            plt.plot(rho_values, mean_eval_data_test)
    
    f.close()
    end = time.time()
    print("Simulations with curves took ", end-start, " seconds")
    plt.show() #Show all three figures at once 

def main():
    plot_histograms(rho=0, compute=False) 
    plot_histograms(rho=1/np.sqrt(30), compute=False)
    #plot_curves(nb_simulations=10, compute=True)

main()