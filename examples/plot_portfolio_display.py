from examples.plot_portfolio_computations import *

import seaborn as sns
import time
import matplotlib.pyplot as plt

def plot_histograms(N=30, nb_simulations=10000, rho=1e-2, compute=True):

    filename = compute_histograms(N, nb_simulations, rho, compute)

    with open (filename, 'rb') as f:
        eval_data_train = np.load(f)
        eval_data_test = np.load(f)
    f.close()

    #Create the histograms
    plt.xlabel("Mean-risk objective")
    if rho == 0:
        plt.title("SAA Solution with scarce data (Kuhn 2017)")
    else:
        plt.title("DRO Solution with scarce data (Kuhn 2017) with rho = %f" %rho)
    sns.histplot(data=eval_data_train, bins=20, stat="probability", color="green")
    sns.histplot(data=eval_data_test, bins=20, stat="probability", color="red")
    plt.show()

def parallel_plot_histograms(N=30, nb_simulations=10000, rho=1e-2, compute=True):

    start = time.time()

    print("Before parallel computations")

    filename = parallel_compute_histograms(N=N, nb_simulations=nb_simulations, rho=rho, compute=compute)

    with open (filename, 'rb') as f:
        eval_data_train = np.load(f)
        eval_data_test = np.load(f)
    f.close()

    print("After parallel computations")
    
    #Create the histograms
    plt.xlabel("Mean-risk objective")
    if rho == 0:
        plt.title("SAA Solution with scarce data (Kuhn 2017)")
    else:
        plt.title("DRO Solution with scarce data (Kuhn 2017) with rho = %f" %rho)
    end = time.time()
    sns.histplot(data=eval_data_train, bins=20, stat="probability", color="green")
    sns.histplot(data=eval_data_test, bins=20, stat="probability", color="red")
    print("Simulations with histograms took ", end-start, " seconds")
    plt.show()  

def plot_curves(nb_simulations=200, compute=True):
    start = time.time()

    print("Before parallel computations")

    samples_size, filename = compute_curves(nb_simulations, compute)
    #samples_size, filename = parallel_compute_curves(nb_simulations, compute)

    print("After parallel computations")

    with open (filename, 'rb') as f:
        rho_values = np.load(f)

        for size in samples_size:

            mean_eval_data_test = np.load(f)
            reliability_test = np.load(f)

            #Create the curves

            _, ax1 = plt.subplots(num=size)

            ax1.set_xlabel("Wasserstein radius")
            ax1.set_ylabel("Out-of-sample performance")
            ax1.set_title("Impact of the Wasserstein Radius (Kuhn 2017) for N = %i" %size)
            ax1.set_xticks(rho_values)
            ax1.set_xscale("log")
            ax1.plot(rho_values, mean_eval_data_test, color='blue')

            ax2 = ax1.twinx()
            ax2.set_ylabel("Reliability")
            ax2.plot(rho_values, reliability_test, linestyle='dashed', color='red')
    
    f.close()
    end = time.time()
    print("Simulations with curves took ", end-start, " seconds")
    plt.show() #Show all three figures at once 

def main():
    N = 30 #Size of samples for Kuhn's histograms

    #plot_histograms(rho=0, compute=True) 
    #plot_histograms(rho=1/np.sqrt(N), compute=True)
    #parallel_plot_histograms(rho=0, compute=False)
    #parallel_plot_histograms(rho=1/np.sqrt(N), compute=True)
    plot_curves(nb_simulations=1, compute=True)

if __name__ == "__main__":
    main()