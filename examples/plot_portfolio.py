"""
===================
Mean-Risk Portfolio
===================

An example of plots of the mean-risk portfolio problem.
"""

import matplotlib.pyplot as plt
import numpy as np
import time

from skwdro.operations_research import Portfolio
from sklearn.model_selection import train_test_split

from joblib import Parallel, delayed
import multiprocessing as mp

M = 10 #Number of assets
ALPHA = 0.2 #Confidence level
ETA = 10 #Risk-aversion quantificator of the investor

def generate_data(N,m):
    psi = np.random.normal(0,0.02,size=(1,10))
    X = np.array([])
    
    for _ in range(N):
        zeta_i = np.array([np.random.normal(0.03*j, 0.025*j + 0.02) for j in range(m)]).reshape((1,10))
        xi_i = psi + zeta_i
        if(X.shape[0] == 0): #Anticipate on vstack's behaviour regarding the dimensions of arrays
            X = np.array(xi_i)
        else:
            X = np.vstack((X,xi_i))
    X_train, X_test = train_test_split(X, train_size=0.5, test_size=0.5)
    return X_train, X_test


def plot_curves():
    start = time.time()

    samples_size = np.array([30,300,3000])
    nb_simulations = 200
    rho_values = np.array([10**(-i) for i in range(4,-1,-1)])

    eval_data_test = np.array([])

    for i in range(nb_simulations):
        for size in samples_size:
            for rho_value in rho_values:

                #Define the training and tesing data
                N = size #Number of samples
                X_train, X_test = generate_data(N=N, m=M)
                
                #Create the estimator and solve the problem
                estimator = Portfolio(solver="dedicated", rho=rho_value, alpha=ALPHA, eta=ETA)
                estimator.fit(X_train)

                #Evaluate the loss value for the training and testing datasets
                eval_test = estimator.eval(X_test)

                #Stock the evaluated losses
                eval_data_test = np.append(eval_data_test, eval_test)

        print("Simulations done: ", i*100/nb_simulations, "%")

    #Create the curves
    plt.xlabel("Wasserstein radius")
    plt.ylabel("Out-of-sample performance")
    plt.title("Impact of the Wasserstein Radius (Kuhn 2017)")
    plt.plot(rho_values, eval_data_test)
    end = time.time()
    print("Simulations with curves took ", end-start, " seconds")
    plt.show()

def for_loop_histograms(N, nb_simulations):
    eval_data_train = np.array([])
    eval_data_test = np.array([])

    for i in range(nb_simulations):

        #Define the training and tesing data
        X_train, X_test = generate_data(N=N, m=M)

        #Create the estimator and solve the problem
        estimator = Portfolio(solver="dedicated", alpha=ALPHA, eta=ETA, rho=1/np.sqrt(N))
        estimator.fit(X_train)

        #Evaluate the loss value for the training and testing datasets
        eval_train = estimator.eval(X_train)
        eval_test = estimator.eval(X_test)

        #Stock the evaluated losses
        eval_data_train = np.append(eval_data_train, eval_train)
        eval_data_test = np.append(eval_data_test, eval_test)

        print("Simulations done: ", i*100/nb_simulations, "%")

    return eval_data_train, eval_data_test

def parallel_for_loop_histograms(N, eval_data_train, eval_data_test):

    #Define the training and tesing data
    X_train, X_test = generate_data(N=N, m=M)

    #Create the estimator and solve the problem
    estimator = Portfolio(solver="dedicated", alpha=ALPHA, eta=ETA, rho=1/np.sqrt(N))
    estimator.fit(X_train)

    #Evaluate the loss value for the training and testing datasets
    eval_train = estimator.eval(X_train)
    eval_test = estimator.eval(X_test)

    #Stock the evaluated losses
    eval_data_train = np.append(eval_data_train, eval_train)
    eval_data_test = np.append(eval_data_test, eval_test)


def plot_histograms():
    start = time.time()
    
    N = 30 #Number of samples
    nb_simulations = 10000

    eval_data_train, eval_data_test = for_loop_histograms(N, nb_simulations)

    #Create the histograms
    plt.xlabel("Mean-risk objective")
    plt.ylabel("Probability")
    plt.title("DRO Solution with scarce data (Kuhn 2017)")
    plt.hist(eval_data_train, bins=20, density=True, histtype="bar", color="green")
    plt.hist(eval_data_test, bins=20, density=True, histtype="bar", color="red")
    end = time.time()
    print("Simulations with histograms took ", end-start, " seconds")
    plt.show()

'''
def parallel_plot_histograms():

    start = time.time()
    
    N = 30 #Number of samples
    nb_simulations = 10000

    print("Before parallel computations")

    eval_data_train, eval_data_test = \
        Parallel(n_jobs=int(mp.cpu_count()), timeout=99999)\
            (delayed(for_loop_histograms)(N=N, nb_simulations=simu) for simu in range(nb_simulations))

    pool = mp.Pool(mp.cpu_count())
    eval_data_train, eval_data_test = pool.map(for_loop_histograms, [simu for simu in range(nb_simulations)])
    pool.close()
    pool.join()
    
    print("After parallel computations")

    #Create the histograms
    plt.xlabel("Mean-risk objective")
    plt.ylabel("Probability")
    plt.title("DRO Solution with scarce data (Kuhn 2017)")
    plt.hist(eval_data_train, bins=20, density=True, histtype="bar", color="green")
    plt.hist(eval_data_test, bins=20, density=True, histtype="bar", color="red")
    end = time.time()
    print("Simulations with histograms took ", end-start, " seconds")
    plt.show()
'''

def parallel_plot_histograms():

    start = time.time()
    
    N = 30 #Number of samples
    nb_simulations = 10000

    print("Before parallel computations")

    with mp.Pool(processes=4) as pool:
        eval_data_train = np.array([])
        eval_data_test = np.array([])
        pool.map(parallel_for_loop_histograms, ((N,eval_data_train, eval_data_test) for _ in range(nb_simulations)))

    print("After parallel computations")
    
    #Create the histograms
    plt.xlabel("Mean-risk objective")
    plt.ylabel("Probability")
    plt.title("DRO Solution with scarce data (Kuhn 2017)")
    plt.hist(eval_data_train, bins=20, density=True, histtype="bar", color="green")
    plt.hist(eval_data_test, bins=20, density=True, histtype="bar", color="red")
    end = time.time()
    print("Simulations with histograms took ", end-start, " seconds")
    plt.show()   


def main():
    #plot_histograms()
    #parallel_plot_histograms()
    plot_curves()

main()