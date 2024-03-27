"""
???
"""
import numpy as np

from skwdro.operations_research import *
from skwdro.linear_models import * 
from sklearn.model_selection import train_test_split
import torch as pt
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from os import makedirs

from skwdro.base.rho_tuners import *

##### PORTFOLIO PARAMETERS #####
M = 10 #Number of assets
ALPHA = 0.2 #Confidence level
ETA = 10 #Risk-aversion quantificator of the investor

##### VERIFICATION CODE FOR LOG_REG #####

def plot_line(est, x):
    c0, c1 = est.coef_
    return -(x*c0 + est.intercept_) / c1

def plot_sim_log_reg(estimator, X, y):
    plt.scatter(X[y==0, 0], X[y==0, 1], color="r")
    plt.scatter(X[y==1, 0], X[y==1, 1], color="b")
    line_plot = [X[:, 0].min(), X[:, 0].max()]
    plt.plot(line_plot, [plot_line(estimator, line_plot[0]), plot_line(estimator, line_plot[1])], \
            'k--', label=f"{estimator.solver}: {estimator.score(X, y)}")

    plt.legend()
    plt.show()


def generate_data(N,m,estimator):
    '''
    Generates data based on an model for the assets that takes systematic and unsystematic
    risk factors into account. This model is described in Section 7.2 of Daniel Kuhn's paper on WDRO.
    '''

    class_name = estimator.__class__.__name__
    X = np.array([])
    y = np.array([])

    match class_name:
        case "Portfolio":
            psi = np.random.normal(0,0.02,size=(1,M))
            for _ in range(N):
                zeta_i = np.array([np.random.normal(0.03*j, 0.025*j + 0.02) for j in range(m)]).reshape((1,10))
                xi_i = psi + zeta_i
                if(X.shape[0] == 0): #Anticipate on vstack's behaviour regarding the dimensions of arrays
                    X = np.array(xi_i)
                else:
                    X = np.vstack((X,xi_i))

            return X, None, class_name
        case "LogisticRegression":

            X_left_x = np.random.uniform(low=-0.5, high=0.2, size=100)
            X_left_y = np.random.uniform(low=-0.5,high=0.2, size=100)
            X_left = np.array([[x,y] for (x,y) in zip(X_left_x, X_left_y)])

            X_right_x = np.random.uniform(low=-0.2, high=0.5, size=50)
            X_right_y = np.random.uniform(low=-0.2,high=0.5, size=50)
            X_right = np.array([[x,y] for (x,y) in zip(X_right_x, X_right_y)])

            X = np.concatenate((X_left, X_right))
            y = np.random.choice([1,-1], size=len(X), p=[0.5,0.5])

            return X, y, class_name

        case "NewsVendor":
            X = np.random.exponential(scale=2.0,size=(20,1))
            return X, None, class_name
        case "Weber":
            raise NotImplementedError()
        case "LinearRegression":
            d = 10
            m = 100

            x0 = np.random.randn(d)

            X = np.random.randn(m,d)

            y = X.dot(x0) +  np.random.randn(m)

            return X, y, class_name          
        case _:
            raise TypeError("Class name for the problem not recognized")

def generate_train_test_data(N,m,estimator):
    '''
    Generates data as described above and splits data into a training and a testing sample.
    '''
    X, y, class_name = generate_data(N,m,estimator)
    if class_name in {"Portfolio", "NewsVendor", "Weber"}:
        X_train, X_test = train_test_split(X, train_size=0.5, test_size=0.5, random_state=42)
        return X_train, X_test, None, None, class_name
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, test_size=0.5, random_state=42)
    return X_train, X_test, y_train, y_test, class_name

def stochastic_problem_approx(estimator,size=10000):
    '''
    Evaluates the real objective value with lots of samples as an approximation of the objective value
    of the original Mean-Value Portfolio stochastic problem.
    '''
    X, y, class_name = generate_data(N=size, m=estimator.problem_.d, estimator=estimator)

    return eval(estimator=estimator, class_name=class_name, X=X, y=y)

#TODO: Adapt structure (uniformization of numpy value signatures)
def eval(estimator, class_name, X, y):

    if class_name == "Portfolio":
        eval = estimator.eval(X)
    else:
        if estimator.solver == "dedicated":
            eval = estimator.problem_.loss.value_split(theta=estimator.coef_, X=X, y=y)
        else:
            if y is None:
                eval = estimator.problem_.loss.primal_loss.value(xi=pt.from_numpy(X).float(),
                                                                xi_labels=None).mean()
            else:
                eval = estimator.problem_.loss.primal_loss.value(xi=pt.from_numpy(X).float(),
                                                                xi_labels=pt.from_numpy(y).float().unsqueeze(-1)).mean()
    
    return eval

def parallel_for_loop_histograms(N, estimator, rho_tuning, blanchet):
    '''
    Parallelization of the for loop on the number of simulations.
    '''
    #Define the training and testing data

    X_train, X_test, y_train, y_test, class_name = generate_train_test_data(N=N, m=M, estimator=estimator)

    if rho_tuning is True:
        rho_tuner = BlanchetRhoTunedEstimator(estimator) if blanchet is True else RhoTunedEstimator(estimator)
        rho_tuner.fit(X=X_train, y=y_train)

        best_estimator = rho_tuner.best_estimator_

        tuned_rho = best_estimator.rho if rho_tuning is True else 0
    else:
        best_estimator = estimator
        best_estimator.fit(X=X_train, y=y_train)
        tuned_rho = 0

    #Define adversarial data
    adv = 1/np.sqrt(N)
    best_decision = best_estimator.coef_
    X_adv_test = X_test - adv*best_decision

    #Evaluate the loss value for the training and testing datasets

    eval_train = eval(estimator=best_estimator, class_name=class_name, X=X_train, y=y_train)
    eval_test = eval(estimator=best_estimator, class_name=class_name, X=X_test, y=y_test)
    eval_adv_test = eval(estimator=best_estimator, class_name=class_name, X=X_adv_test, y=y_test)

    print("Eval train: ", eval_train)
    print("Eval test: ", eval_test)
    print("Eval adv test: ", eval_adv_test)
    print("Best decision: ", best_decision)

    '''
    Visualization part for the Logistic problem: to move elsewhere
    
    if class_name == "LogisticRegression":
        plot_sim_log_reg(estimator, X_train, y_train)
    '''

    return eval_train, eval_test, eval_adv_test, tuned_rho

def parallel_compute_histograms(N, nb_simulations, estimator, compute, rho_tuning, blanchet):
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
    rho_filename = './examples/stored_data/rho_tuning_data.npy'

    if compute is True: 

        eval_data = Parallel(n_jobs=-1, verbose=10)(
            delayed(parallel_for_loop_histograms)(N=N, estimator=estimator, rho_tuning=rho_tuning, blanchet=blanchet)
            for _ in range(nb_simulations)
        )

        #debug mode (to use in case of issues inside parallelized code)
        #eval_train, eval_test, eval_adv_test, tuned_rho = parallel_for_loop_histograms(N=N, estimator=estimator, rho_tuning=rho_tuning, blanchet=blanchet)

        eval_data_train = [x for x, _, _, _ in eval_data]
        eval_data_test = [y for _, y, _, _ in eval_data]
        eval_data_adv_test = [z for _, _, z, _ in eval_data]
        tuned_rho_data = [t for _, _, _, t in eval_data]

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

        if rho_tuning is True:
            #We store in a different file the chosen rho values for each simulation 
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
    X_train, X_test, y_train, y_test, class_name = generate_train_test_data(N=N, m=M, estimator=estimator)

    estimator.fit(X=X_train, y=y_train)

    #Evaluate the loss value for the testing dataset
    eval_test = eval(estimator=estimator, class_name=class_name, X=X_test, y=y_test)
    print("eval_test: ", eval_test)

    #Approximate the real loss value and compare it to the WDRO loss value
    eval_approx_loss = stochastic_problem_approx(estimator)
    eval_result = estimator.result_

    print("eval_approx_loss: ", eval_approx_loss)
    print("estimator.result_:", eval_result)
    if eval_approx_loss <= eval_result:
        reliability_cpt += 1

    return eval_test, reliability_cpt

def rho_parallel_for_loop_curves(N, estimator, rho_value, nb_simulations):
    '''
    Parallelization of the loop on the number of simulations.
    '''

    #Create a new instance to solve concurrential access issues and solve the problem 
    cost = estimator.cost
    if estimator.__class__.__name__ == "Portfolio":
        curves_estimator = Portfolio(solver=estimator.solver, cost=cost, rho=rho_value, reparam=estimator.reparam, 
                            alpha=ALPHA, eta=ETA, n_zeta_samples=estimator.n_zeta_samples)
    else:
        curves_estimator = estimator.__class__(solver=estimator.solver, rho=rho_value, cost=cost,
                                        n_zeta_samples=estimator.n_zeta_samples)

    eval_reliability_data_test = Parallel(n_jobs=-1, verbose=10)(
        delayed(simulations_parallel_for_loop_curves)(N=N, estimator=curves_estimator)
        for _ in range(nb_simulations)
    )

    #debug mode (to use in case of issues inside parallelized code)
    #_, _ = simulations_parallel_for_loop_curves(N=N, estimator=curves_estimator)

    #The datatypes in the two lists are the same so we only test on one of them
    if isinstance(eval_reliability_data_test[0][0], pt.torch.Tensor):
        eval_data_test = [x.detach().numpy() for x, _ in eval_reliability_data_test]
    else:
        eval_data_test = [x for x, _ in eval_reliability_data_test]
    
    reliability = sum([y for _, y in eval_reliability_data_test])/nb_simulations

    return np.mean(eval_data_test), reliability

def parallel_for_loop_curves(N, estimator, rho_values, nb_simulations):
    '''
    Parallelization of the loop on the number of simulations.
    '''

    mean_eval_reliability = Parallel(n_jobs=-1, verbose=10)(
    delayed(rho_parallel_for_loop_curves)(N=N, estimator=estimator, rho_value=rho_value, nb_simulations=nb_simulations)
    for rho_value in rho_values)

    #debug mode (to use in case of issues inside parallelized code)
    #_, _ = rho_parallel_for_loop_curves(N=N, estimator=estimator, rho_value=1e-3, nb_simulations=nb_simulations)

    mean_eval_data_test = [x for x, _ in mean_eval_reliability]
    reliability_test = [y for _, y in mean_eval_reliability]

    return mean_eval_data_test, reliability_test


def parallel_compute_curves(nb_simulations, estimator, compute):
    samples_size = np.array([3000])
    rho_values = np.array([10**(-i) for i in range(4,-1,-1)])

    filename = './examples/stored_data/parallel_portfolio_curve_data.npy'

    if compute is True:

        with open (filename, 'wb') as f:

            np.save(f, rho_values)

            mean_eval_rel_sizes = Parallel(n_jobs=-1, verbose=10)(
                delayed(parallel_for_loop_curves)(N=size, estimator=estimator, rho_values=rho_values, nb_simulations=nb_simulations)
                for size in samples_size)

            #debug mode (to use in case of issues inside parallelized code)
            _, _ = parallel_for_loop_curves(N=30, estimator=estimator, rho_values=rho_values, nb_simulations=nb_simulations)
            
            mean_eval_data_test_sizes = [x for x, _ in mean_eval_rel_sizes]
            reliability_test_sizes = [y for _, y in mean_eval_rel_sizes]

            for i in range(len(mean_eval_data_test_sizes)):
                np.save(f, mean_eval_data_test_sizes[i])
                np.save(f, reliability_test_sizes[i])

        f.close()

    return samples_size, filename
