"""
===================
Mean-Risk Portfolio
===================

An example of plots of the mean-risk portfolio problem.
"""

import matplotlib.pyplot as plt
import numpy as np

from skwdro.operations_research import Portfolio
from sklearn.model_selection import train_test_split

M = 10 #Number of assets
ALPHA = 0.2 #Confidence level
ETA = 10 #Risk-aversion of the investor

def generate_data(N,m):
    psi = np.random.normal(0,0.02,size=(1,10))
    X = np.array([])
    
    for _ in range(N):
        zeta_i = np.array([np.random.normal(0.03*j, 0.025*j + 0.02) for j in range(m)]).reshape((1,10))
        xi_i = psi + zeta_i
        print("xi_i:", xi_i)
        if(X.shape[0] == 0): #Anticipate on vstack's behaviour regarding the dimensions of arrays
            X = np.array(xi_i)
            print(X)
        else:
            X = np.vstack((X,xi_i))
            print(X)
    return X


def plot_curves():
    samples_size = [30,300,3000]
    nb_simulations = 200
    rho_values = [10**(-i) for i in range(4,-1,-1)]

    for _ in range(nb_simulations):
        for size in samples_size:
            for rho_value in rho_values:

                N = size #Number of samples
                X = generate_data(N=N, m=M)

                #Create the estimator and solve the problem
                estimator = Portfolio(solver="dedicated", rho=rho_value, alpha=ALPHA, eta=ETA)
                estimator.fit(X)

                theta_train = estimator.coef_
    
    return NotImplementedError()

def plot_histograms():
    
    N = 2 #Number of samples
    nb_simulations = 10000

    eval_data_train = np.array([])
    eval_data_test = np.array([])

    for _ in range(nb_simulations):
        
        #Define the training and tesing data
        X = generate_data(N=N, m=M)
        X_train, X_test = train_test_split(X, train_size=0.5, test_size=0.5)

        #Create the estimator and solve the problem (with default value for rho)
        estimator = Portfolio(solver="dedicated", alpha=ALPHA, eta=ETA)
        estimator.fit(X_train)

        #Evaluate the loss value for the training and testing datasets
        eval_train = estimator.eval(X_train)
        eval_test = estimator.eval(X_test)

        #Stock the evaluated losses
        eval_data_train = np.append(eval_data_train, eval_train)
        eval_data_test = np.append(eval_data_test, eval_test)

    #Create the histograms
    plt.hist(eval_data_train, bins=20)
    plt.hist(eval_data_test, bins=20)
    plt.show()

def main():
    plot_histograms()


main()