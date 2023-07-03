
import numpy as np

from skwdro.operations_research import Portfolio, NewsVendor
from sklearn.model_selection import train_test_split
import torch as pt

from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, KFold

from joblib import Parallel, delayed
from os import makedirs

from skwdro.base.rho_tuner import RhoTunedEstimator

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
    of the original Mean-Value Portfolio stochastic problem.
    '''
    X = generate_data(N=size, m=estimator.problem_.d)
    return estimator.eval(X)

def parallel_for_loop_histograms(N, estimator, rho_tuning):
    '''
    Parallelization of the for loop on the number of simulations.
    '''
    #Define the training and testing data

    X_train, X_test = generate_train_test_data(N=N, m=M)

    rho_tuner = RhoTunedEstimator(estimator)
    rho_tuner.fit(X=X_train, y=None)

    best_estimator = rho_tuner.best_estimator_

    tuned_rho = best_estimator.rho if rho_tuning is True else 0

    #Define adversarial data
    adv = 1/np.sqrt(N)
    X_adv_test = X_test - adv*best_estimator.coef_

    #Evaluate the loss value for the training and testing datasets
    eval_train = best_estimator.eval(X_train)
    eval_test = best_estimator.eval(X_test)
    eval_adv_test = best_estimator.eval(X_adv_test)
    print("Eval train: ", eval_train)
    print("Eval test: ", eval_test)
    print("Eval adv test: ", eval_adv_test)

    return eval_train, eval_test, eval_adv_test, tuned_rho

def parallel_compute_histograms(N, nb_simulations, estimator_solver, compute, rho_tuning):
    '''
    Computes Kuhn's histograms that were presented at the DTU CEE Summer School 2018.
    '''
    makedirs("./examples/stored_data", exist_ok=True)

    '''
    if rho == 0:
        filename = './examples/stored_data/parallel_portfolio_histogram_SAA_data.npy'
    else:
        filename = './examples/stored_data/parallel_portfolio_histogram_WDRO_data.npy'
    '''

    filename = './examples/stored_data/parallel_portfolio_histogram_WDRO_data.npy'

    if compute is True: 

        #Define sigma for adversarial distribution pi_{0} and number of its samples
        '''
        sigma = 0 if estimator_solver not in \
            {"entropic", "entropic_torch", "entropic_torch_pre", "entropic_torch_post"} else (rho if rho != 0 else 0.1)
        '''
        n_zeta_samples = 0 if estimator_solver not in \
            {"entropic", "entropic_torch", "entropic_torch_pre", "entropic_torch_post"} else 10*N

        #Create the estimator and solve the problem
        estimator = Portfolio(solver=estimator_solver, reparam="softmax", alpha=ALPHA, eta=ETA, n_zeta_samples=n_zeta_samples)

        print("Before joblib parallel computations")
        eval_data = Parallel(n_jobs=-1)(
            delayed(parallel_for_loop_histograms)(N=N, estimator=estimator, rho_tuning=rho_tuning)
            for _ in range(nb_simulations)
        )
        eval_data_train = [x for x, _, _, _ in eval_data]
        eval_data_test = [y for _, y, _, _ in eval_data]
        eval_data_adv_test = [z for _, _, z, _ in eval_data]
        tuned_rho_data = [t for _, _, _, t in eval_data]
        print("After joblib parallel computations")

        #We store the computed data
        with open (filename, 'wb') as f:
            '''
            The datatypes in the three lists are the same so we only test on one of them
            '''
            if isinstance(eval_data_train[0], pt.torch.Tensor):
                eval_data_train = [x.detach().numpy() for x in eval_data_train]
                eval_data_test = [x.detach().numpy() for x in eval_data_test]
                eval_data_adv_test = [x.detach().numpy() for x in eval_data_adv_test]

            np.save(f, eval_data_train)
            np.save(f, eval_data_test)
            np.save(f, eval_data_adv_test)
        
        f.close()

        #We store in a different file the chosen rho values for each simulation 
        rho_filename = './examples/stored_data/rho_tuning_data.npy'
        with open(rho_filename, 'wb') as f:
            np.save(f, tuned_rho_data)
        f.close()

    return filename, rho_filename

def simulations_parallel_for_loop_curves(N, estimator):
    '''
    Parallelization of the loop on the number of simulations.
    '''
    reliability_cpt = 0

    #Define the training and testing data
    X_train, X_test = generate_train_test_data(N=N, m=M)

    estimator.fit(X_train)

    #Evaluate the loss value for the testing dataset
    eval_test = estimator.eval(X_test)
    print("eval_test: ", eval_test)

    #Approximate the real loss value and compate it to the WDRO loss value
    eval_approx_loss = stochastic_problem_approx(estimator)
    print("eval_approx_loss: ", eval_approx_loss)
    print("estimator.result_:", estimator.result_)
    if eval_approx_loss <= estimator.result_:
        reliability_cpt += 1

    return eval_test, reliability_cpt

def rho_parallel_for_loop_curves(N, estimator_solver, rho_value, nb_simulations):
    '''
    Parallelization of the loop on the number of simulations.
    '''

    #Define sigma for adversarial distribution pi_{0}
    sigma = 0 if estimator_solver \
        not in {"entropic", "entropic_torch", "entropic_torch_pre", "entropic_torch_post"} else (rho_value if rho_value != 0 else 0.1)
    n_zeta_samples = 0 if estimator_solver \
        not in {"entropic", "entropic_torch", "entropic_torch_pre", "entropic_torch_post"} else 10*N
    
    #Create the estimator and solve the problem
    estimator = Portfolio(solver=estimator_solver, rho=rho_value, reparam="softmax", solver_reg=sigma, alpha=ALPHA, eta=ETA, n_zeta_samples=n_zeta_samples)

    eval_reliability_data_test = Parallel(n_jobs=-1)(
        delayed(simulations_parallel_for_loop_curves)(N=N, estimator=estimator)
        for _ in range(nb_simulations)
    )

    #The datatypes in the two lists are the same so we only test on one of them
    if isinstance(eval_reliability_data_test[0][0], pt.torch.Tensor):
        eval_data_test = [x.detach().numpy() for x, _ in eval_reliability_data_test]
    else:
        eval_data_test = [x for x, _ in eval_reliability_data_test]
    
    reliability = sum([y for _, y in eval_reliability_data_test])/nb_simulations

    return np.mean(eval_data_test), reliability

def parallel_for_loop_curves(N, estimator_solver, rho_values, nb_simulations):
    '''
    Parallelization of the loop on the number of simulations.
    '''

    mean_eval_reliability = Parallel(n_jobs=-1)(
    delayed(rho_parallel_for_loop_curves)(N=N, estimator_solver=estimator_solver, rho_value=rho_value, nb_simulations=nb_simulations)
    for rho_value in rho_values)

    mean_eval_data_test = [x for x, _ in mean_eval_reliability]
    reliability_test = [y for _, y in mean_eval_reliability]

    return mean_eval_data_test, reliability_test


def parallel_compute_curves(nb_simulations, estimator_solver, compute):
    samples_size = np.array([30])
    rho_values = np.array([10**(-i) for i in range(4,-4,-1)])

    filename = './examples/stored_data/parallel_portfolio_curve_data.npy'

    if compute is True:

        with open (filename, 'wb') as f:

            np.save(f, rho_values)

            mean_eval_rel_sizes = Parallel(n_jobs=-1)(
                delayed(parallel_for_loop_curves)(N=size, estimator_solver=estimator_solver, rho_values=rho_values, nb_simulations=nb_simulations)
                for size in samples_size)
            
            mean_eval_data_test_sizes = [x for x, _ in mean_eval_rel_sizes]
            reliability_test_sizes = [y for _, y in mean_eval_rel_sizes]

            for i in range(len(mean_eval_data_test_sizes)):
                np.save(f, mean_eval_data_test_sizes[i])
                np.save(f, reliability_test_sizes[i])

        f.close()

    return samples_size, filename

