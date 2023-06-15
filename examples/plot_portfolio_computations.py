
import numpy as np

from skwdro.operations_research import Portfolio
from sklearn.model_selection import train_test_split
import torch as pt

from joblib import Parallel, delayed
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

def generate_train_test_data(N,m,estimator_solver):
    '''
    Generates data as described above and splits data into a training and a testing sample.
    '''
    X = generate_data(N,m)
    X_train, X_test = train_test_split(X, train_size=0.5, test_size=0.5, random_state=42)

    return X_train, X_test

def stochastic_problem_approx(estimator,size=10000):
    '''
    Evaluates the real objective value with lots of samples as an approximation of the objective value
    of the original Mean-Value Portfolio Stochastic problem.
    '''
    X = generate_data(N=size, m=estimator.problem_.d)

    if isinstance(X, (np.ndarray,np.generic)):
        X = pt.from_numpy(X)
    
    #We stock temporarily the real_value of n_samples if we solve the entropic problem 

    if estimator.solver in {"entropic", "entropic_torch"}:
        #initial_n_samples = estimator.n_samples
        #estimator.problem_.loss.n_samples = size
        #print(estimator.problem_.loss.n_samples)
        approx_obj_value = estimator.problem_.loss.loss(X, None).mean(dim=0)
        print(approx_obj_value)
        #estimator.problem_.loss.n_samples = initial_n_samples
    else:
        approx_obj_value = estimator.eval(X)
    return approx_obj_value

def parallel_for_loop_histograms(N, rho, estimator_solver, adv):
    '''
    Parallelization of the for loop on the number of simulations.
    '''
    #Define the training and testing data
    X_train, X_test = generate_train_test_data(N=N, m=M, estimator_solver=estimator_solver)

    #Define sigma for adversarial distribution pi_{0} and number of its samples
    sigma = 0 if estimator_solver not in {"entropic", "entropic_torch"} else (rho if rho != 0 else 0.1)
    n_zeta_samples = 0 if estimator_solver not in {"entropic", "entropic_torch"} else 10*N

    #Create the estimator and solve the problem
    estimator = Portfolio(solver=estimator_solver, solver_reg=sigma, alpha=ALPHA, eta=ETA, rho=rho, n_zeta_samples=n_zeta_samples)
    estimator.fit(X_train)

    #Define adversarial data
    X_adv_test = X_test - adv*estimator.coef_

    #Evaluate the loss value for the training and testing datasets
    eval_train = estimator.eval(X_train)
    eval_test = estimator.eval(X_test)
    eval_adv_test = estimator.eval(X_adv_test)
    print(eval_train)
    print(eval_test)
    print(eval_adv_test)

    return eval_train, eval_test, eval_adv_test

def parallel_compute_histograms(N, nb_simulations, rho, estimator_solver, adv, compute):
    '''
    Computes Kuhn's histograms that were presented at the DTU CEE Summer School 2018.
    '''
    makedirs("./examples/stored_data", exist_ok=True)
    if rho == 0:
        filename = './examples/stored_data/parallel_portfolio_histogram_SAA_data.npy'
    else:
        filename = './examples/stored_data/parallel_portfolio_histogram_WDRO_data.npy'

    if compute is True:

        print("Before joblib parallel computations")
        eval_data = Parallel(n_jobs=-1)(
            delayed(parallel_for_loop_histograms)(N, rho, estimator_solver, adv)
            for _ in range(nb_simulations)
        )
        eval_data_train = [x for x, _, _ in eval_data]
        eval_data_test = [y for _, y, _ in eval_data]
        eval_data_adv_test = [z for _, _, z in eval_data]
        print("After joblib parallel computations")

        #We store the computed data
        with open (filename, 'wb') as f:
            #The datatypes in the three lists are the same so we only test on one of them
            if isinstance(eval_data_train[0], pt.torch.Tensor):
                eval_data_train = [x.detach().numpy() for x in eval_data_train]
                eval_data_test = [x.detach().numpy() for x in eval_data_test]
                eval_data_adv_test = [x.detach().numpy() for x in eval_data_adv_test]

            np.save(f, eval_data_train)
            np.save(f, eval_data_test)
            np.save(f, eval_data_adv_test)
        
        f.close()

    return filename

def parallel_for_loop_curves(N, estimator_solver, rho):
    '''
    Parallelization of the loop on the number of simulations.
    '''
    reliability_cpt = 0

    #Define the training and testing data
    X_train, X_test = generate_train_test_data(N=N, m=M, estimator_solver=estimator_solver)

    #Define sigma for adversarial distribution pi_{0}
    sigma = 0 if estimator_solver not in {"entropic", "entropic_torch"} else (rho if rho != 0 else 0.1)
    n_zeta_samples = 0 if estimator_solver not in {"entropic", "entropic_torch"} else 10*N
    
    #Create the estimator and solve the problem
    estimator = Portfolio(solver=estimator_solver, rho=rho, solver_reg=sigma, alpha=ALPHA, eta=ETA, n_zeta_samples=n_zeta_samples)
    estimator.fit(X_train)

    #Evaluate the loss value for the testing dataset
    eval_test = estimator.eval(X_test)

    #Approximate the real loss value and compate it to the WDRO loss value
    eval_approx_loss = stochastic_problem_approx(estimator)
    if eval_approx_loss <= estimator.result_:
        reliability_cpt += 1

    return eval_test, reliability_cpt

def parallel_compute_curves(nb_simulations, estimator_solver, compute):
    '''
    Computes Kuhn's curves from Section 7.2 of the 2017 WDRO paper.
    '''
    samples_size = np.array([30])
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

                    eval_reliability_data_test = Parallel(n_jobs=-1)(
                        delayed(parallel_for_loop_curves)(N=size, estimator_solver=estimator_solver, rho=rho_value)
                        for _ in range(nb_simulations)
                    )
                    eval_data_test = [x for x, _ in eval_reliability_data_test]
                    reliability = sum([y for _, y in eval_reliability_data_test])/nb_simulations

                    #At the end of each set of 200 simulations, we compute the mean value for the out-of-sample performance
                    mean_eval_data_test = np.append(mean_eval_data_test,np.mean(eval_data_test))
                    reliability_test = np.append(reliability_test, reliability)

                #The datatypes in the two lists are the same so we only test on one of them
                if isinstance(mean_eval_data_test[0], pt.torch.Tensor):
                    mean_eval_data_test = [x.detach().numpy() for x in mean_eval_data_test]
                    reliability_test = [x.detach().numpy() for x in mean_eval_data_test]
                
                np.save(f, mean_eval_data_test)
                np.save(f, reliability_test)

        f.close()

    return samples_size, filename