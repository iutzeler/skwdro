
import numpy as np

from skwdro.operations_research import Portfolio
from sklearn.model_selection import train_test_split
import torch as pt

import multiprocessing as mp
from os import makedirs

M = 10 #Number of assets
ALPHA = 0.2 #Confidence level
ETA = 10 #Risk-aversion quantificator of the investor

def generate_data(N,m):
    '''
    Generates data based on an model for the assets that takes systematic and unsystematic
    risk factors into account. This model is described in Section 7.2 of Daniel Kuhn's paper on WDRO.
    '''
    psi = np.random.normal(0,0.02,size=(1,10))
    X = np.array([])
    
    for _ in range(N):
        zeta_i = np.array([np.random.normal(0.03*j, 0.025*j + 0.02) for j in range(m)]).reshape((1,10))
        xi_i = psi + zeta_i
        if(X.shape[0] == 0): #Anticipate on vstack's behaviour regarding the dimensions of arrays
            X = np.array(xi_i)
        else:
            X = np.vstack((X,xi_i))

    return X

def generate_train_test_data(N,m):
    '''
    Generates data as described above and splits data into a training and a testing sample.
    '''
    X = generate_data(N,m)
    X_train, X_test = train_test_split(X, train_size=0.5, test_size=0.5, random_state=42)
    return X_train, X_test

def stochastic_problem_approx(estimator,size=10000):
    '''
    Evaluates the real objective value with lots of samples as an approximation of the objective value
    of the original Mean-Value Prtfolio Stochastic problem.
    '''
    X = generate_data(N=size, m=estimator.problem_.d)
    approx_obj_value = estimator.eval(X)
    return approx_obj_value

def parallel_for_loop_histograms(N, rho):
    '''
    Parallelization of the for loop on the number of simulations.
    '''
    #Define the training and tesing data
    X_train, X_test = generate_train_test_data(N=N, m=M)

    #Create the estimator and solve the problem
    estimator = Portfolio(solver="dedicated", alpha=ALPHA, eta=ETA, rho=rho)
    estimator.fit(X_train)

    #Evaluate the loss value for the training and testing datasets
    eval_train = estimator.eval(X_train)
    eval_test = estimator.eval(X_test)

    return eval_train, eval_test

def parallel_compute_histograms(N, nb_simulations, rho, compute):
    '''
    Computes Kuhn's histograms that were presented at the DTU CEE Summer School 2018.
    '''
    makedirs("./examples/stored_data", exist_ok=True)
    if rho == 0:
        filename = './examples/stored_data/parallel_portfolio_histogram_SAA_data.npy'
    else:
        filename = './examples/stored_data/parallel_portfolio_histogram_WDRO_data.npy'

    if compute is True:

        with mp.Pool(processes=mp.cpu_count()) as pool:

            eval_data = pool.starmap(parallel_for_loop_histograms, zip((N for _ in range(nb_simulations)), \
                                        (rho for _ in range(nb_simulations))))
            eval_data_train = [x for x, _ in eval_data]
            eval_data_test = [y for _, y in eval_data]

        #We store the computed data
        with open (filename, 'wb') as f:
            np.save(f, eval_data_train)
            np.save(f, eval_data_test)
        
        f.close()

    return filename

def parallel_for_loop_curves(N, rho):
    '''
    Parallelization of the for loop on the number of simulations.
    '''
    reliability_cpt = 0

    #Define the training and tesing data
    X_train, X_test = generate_train_test_data(N=N, m=M)
    
    #Create the estimator and solve the problem
    estimator = Portfolio(solver="dedicated", rho=rho, alpha=ALPHA, eta=ETA)
    estimator.fit(X_train)

    #Evaluate the loss value for the testing dataset
    eval_test = estimator.eval(X_test)

    #Approximate the real loss value and compate it to the WDRO loss value
    eval_approx_loss = stochastic_problem_approx(estimator)
    if eval_approx_loss <= estimator.result_:
        reliability_cpt += 1

    return eval_test, reliability_cpt

def parallel_compute_curves(nb_simulations, compute):
    '''
    Computes Kuhn's curves from Section 7.2 of the 2017 WDRO paper.
    '''
    samples_size = np.array([30, 300])
    #samples_size = np.array([30,300,3000])
    rho_values = np.array([10**(-i) for i in range(4,-1,-1)])

    filename = './examples/stored_data/parallel_portfolio_curve_data.npy'

    if compute is True:

        with open (filename, 'wb') as f:

            np.save(f, rho_values)

            for size in samples_size:
                mean_eval_data_test = np.array([]) #Mean value of the out-of-sample performance for each rho
                reliability_test = np.array([]) #Probability array that the WDRO objective value is a supremum of the real value
                for rho_value in rho_values:
                    with mp.Pool(processes=mp.cpu_count()) as pool:
                        eval_reliability_data_test = pool.starmap(parallel_for_loop_curves, zip((size for _ in range(nb_simulations)), \
                                                           (rho_value for _ in range(nb_simulations))))
                        eval_data_test = [x for x, _ in eval_reliability_data_test]
                        reliability = sum([y for _, y in eval_reliability_data_test])/nb_simulations
                    #At the end of each set of 200 simulations, we compute the mean value for the out-of-sample performance
                    mean_eval_data_test = np.append(mean_eval_data_test,np.mean(eval_data_test))
                    reliability_test = np.append(reliability_test, reliability)
                np.save(f, mean_eval_data_test)
                np.save(f, reliability_test)

        f.close()

    return samples_size, filename
