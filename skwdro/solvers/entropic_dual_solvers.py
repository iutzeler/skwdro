import numpy as np
from numpy.random import sample

from skwdro.solvers.utils import *
from skwdro.solvers.optim_cond import OptCond

# import progressbar
# widgets = [' [',
#          progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
#          '] ',
#            progressbar.Bar('*'),' (',
#            progressbar.ETA(), ') ',
#           ]



def WDROEntropicSolver(WDROProblem=None, epsilon=1e-2, Nsamples = 20,fit_intercept=False, opt_cond=OptCond(2)):
    #return WangGaoXie_v1(WDROProblem=WDROProblem, epsilon=epsilon, Nsamples = Nsamples,fit_intercept=fit_intercept)
    return WangGaoXie_v2(WDROProblem=WDROProblem, epsilon=epsilon, n_samples = Nsamples,fit_intercept=fit_intercept, opt_cond=opt_cond)

def lr_decay_schedule(iter_idx, offset: int=10, lr0=1e-1):
    return lr0 * (iter_idx + offset)**-0.8

# ### Initializations and sampling #############################
def init_theta(d):
    # Glorot init
    # Fan_in = n or n+1 depending on intercept
    theta = np.random.randn(d) / np.sqrt(.5 * (d+1))
    return theta

def prepare_data(samples, m, d, n_samples, epsilon, fit_intercept):
    # pi_0_samples = sample_pi_0(epsilon, n_samples, samples)
    if fit_intercept:
        xi = np.concatenate((np.ones((m, 1)), samples), axis=1)
        # zeta = np.concatenate((np.ones((n_samples, m, 1)), pi_0_samples), axis=2)
        theta = init_theta(d + 1)
    else:
        xi = samples
        theta = init_theta(d)
        # zeta = pi_0_samples
    zeta = sample_pi_0(epsilon, n_samples, xi)
    return xi, theta, zeta

def sample_pi_0(epsilon, n_samples, xi):
    m, d = xi.shape

    sigma = epsilon
    pi_0_noise = np.random.randn(n_samples, m, d)
    pi_0_raw_samples = pi_0_noise * sigma + xi[None, :, :]

    # TODO: constrain to Xi_bounds?
    zeta = pi_0_raw_samples
    return zeta
# ##############################################################

# TODO: factorize exp mean code
# ### Steps ####################################################
def step_lam_wol(xi, zeta, theta, lam, cost, loss, t, rho, epsilon):
    c = cost(xi[None, :, :], zeta) # (n_samples, m, 1)

    loss_outputs = loss(theta, zeta)
    exps_coefs = loss_outputs - lam * c
    exps_coefs /= epsilon

    minus_full_grads = non_overflow_exp_mean(exps_coefs, c)
    grad_estimate = rho - minus_full_grads.mean()
    lr = lr_decay_schedule(t)
    return -lr * grad_estimate

def step_lam_wl(xi, xi_labels, zeta, zeta_labels, theta, lam, cost, loss, t, rho, epsilon):
    c = cost(xi[None, :, :], zeta, xi_labels[None, :, :], zeta_labels) # (n_samples, m, 1)

    loss_outputs = loss(theta, zeta, zeta_labels)
    exps_coefs = loss_outputs - lam * c
    exps_coefs /= epsilon

    minus_full_grads = non_overflow_exp_mean(exps_coefs, c)
    grad_estimate = rho - minus_full_grads.mean()
    lr = lr_decay_schedule(t)
    return -lr * grad_estimate

def step_theta_wol(xi, zeta, theta, lam, cost, loss_fns, step_id, epsilon):
    loss, loss_grad = loss_fns
    grads_theta_loss = loss_grad(theta, zeta) # array of shape (n_samples, m, d)

    c = cost(xi[None, :, :], zeta) # (n_samples, m, 1)
    loss_outputs = loss(theta, zeta)
    exps_coefs = loss_outputs - lam * c
    exps_coefs /= epsilon

    full_grads = non_overflow_exp_mean(exps_coefs, grads_theta_loss)
    grad_estimate = full_grads.mean(axis=0)

    lr = lr_decay_schedule(step_id)
    return -lr * grad_estimate

def step_theta_wl(xi, xi_labels, zeta, zeta_labels, theta, lam, cost, loss_fns, step_id, epsilon):
    loss, loss_grad = loss_fns
    grads_theta_loss = loss_grad(theta, zeta, zeta_labels) # array of shape (n_samples, m, d)

    c = cost(xi[None, :, :], zeta, xi_labels[None, :, :], zeta_labels) # (n_samples, m, 1)
    loss_outputs = loss(theta, zeta, zeta_labels)
    exps_coefs = loss_outputs - lam * c
    exps_coefs /= epsilon

    full_grads = non_overflow_exp_mean(exps_coefs, grads_theta_loss)
    grad_estimate = full_grads.mean(axis=0)

    lr = lr_decay_schedule(step_id)
    return -lr * grad_estimate


def step_wgx_wol(xi, zeta, theta, lam, cost, loss_fns, t, rho_eps):
    rho, epsilon = rho_eps

    K = 5
    step_theta = step_theta_wol(xi, zeta, theta, lam, cost, loss_fns, t, epsilon)
    for k in range(1, K):
        step_theta = step_theta_wol(xi, zeta, theta, lam, cost, loss_fns, t+k, epsilon)
        theta += step_theta
    step_lambda = step_lam_wol(xi, zeta, theta, lam, cost, loss_fns[0], t, rho, epsilon)
    lam += step_lambda
    return theta, lam, (step_theta, step_lambda)

def step_wgx_wl(xi, xi_labels, zeta, zeta_labels, theta, lam, cost, loss_fns, t, rho_eps):
    rho, epsilon = rho_eps

    K = 5
    step_theta = step_theta_wl(xi, xi_labels, zeta, zeta_labels, theta, lam, cost, loss_fns, t, epsilon)
    for k in range(1, K):
        step_theta = step_theta_wl(xi, xi_labels, zeta, zeta_labels, theta, lam, cost, loss_fns, t+k, epsilon)
        theta += step_theta
    step_lambda = step_lam_wl(xi, xi_labels, zeta, zeta_labels, theta, lam, cost, loss_fns[0], t, rho, epsilon)
    lam += step_lambda
    return theta, lam, (step_theta, step_lambda)
# ##############################################################


def wgx_v2_wo_labels(d, data_structure, m, rho_eps, lam_0, n_samples, cost, loss_fns, fit_intercept, opt_cond):

    xi, theta, zeta = prepare_data(data_structure.samples, m, d, n_samples, rho_eps[1], fit_intercept)

    c = lambda samples_x, z_x, *_: cost(samples_x, z_x)

    t = 1
    lam = lam_0
    while t > 0:
        theta, lam, grads = step_wgx_wol(xi, zeta, theta, lam, c, loss_fns, t, rho_eps)
        if opt_cond(grads, t): break
        else: t += 1
    return theta, lam


def wgx_v2_w_labels(d, data_structure, m, rho_eps, lam_0, n_samples, cost, loss, fit_intercept, opt_cond):

    xi, theta, zeta = prepare_data(data_structure.samplesX, m, d, n_samples, rho_eps[1], fit_intercept)
    xi_labels = data_structure.samplesY[:, None]
    zeta_labels = sample_pi_0(rho_eps[1], n_samples, xi_labels)

    c = lambda samples_x, z_x, samples_y, z_y: cost(samples_x, z_x, samples_y, z_y)

    t = 1
    lam = lam_0
    while t > 0:
        theta, lam, grads = step_wgx_wl(xi, xi_labels, zeta, zeta_labels, theta, lam, c, loss, t, rho_eps)
        if opt_cond(grads, t): break
        else: t += 1
    return theta, lam

def WangGaoXie_v2(WDROProblem, epsilon=0.1, n_samples=50, fit_intercept=False, opt_cond=OptCond(2)):
    """ Algorithm of Wang et al. but with epsilon >0 and delta = 0 (regularization in the objective)"""

    n = WDROProblem.n
    d = WDROProblem.d

    no_labels = WDROProblem.dLabel == 0

    data_structure = WDROProblem.P

    m = WDROProblem.P.m
    rho =  WDROProblem.rho

    lamL = 1e-10
    lamU = 1e10
    lam = (lamL+lamU)/2.0

    lam_eps = 1e2

    # TODO: fix
    lam = 1e1

    loss = WDROProblem.loss
    if no_labels:
        theta, lam = wgx_v2_wo_labels(
                d,
                data_structure,
                m,
                (rho, epsilon),
                lam,
                n_samples,
                WDROProblem.c,
                (loss.value, loss.grad_theta),
                fit_intercept,
                opt_cond
            )
    else:
        theta, lam = wgx_v2_w_labels(
                d,
                data_structure,
                m,
                (rho, epsilon),
                lam,
                n_samples,
                WDROProblem.c,
                (loss.valueSplit, loss.grad_thetaSplit),
                fit_intercept,
                opt_cond
            )
    if fit_intercept:
        return theta[1:], theta[0], lam
    else:
        return theta, None, lam

def WangGaoXie_v1(WDROProblem, epsilon=0.1, Nsamples = 50,fit_intercept=False):
    """ Algorithm of Wang et al. but with epsilon >0 and delta = 0 (regularization in the objective)"""

    n = WDROProblem.n
    d = WDROProblem.d

    NoLabels = WDROProblem.dLabel == 0

    if NoLabels:
        samples = WDROProblem.P.samples
        labels = np.zeros(samples.shape[0]) # placeholder for custom cost calls
    else:
        samples = WDROProblem.P.samplesX
        labels  = WDROProblem.P.samplesY


    theta = np.random.rand(n)*0.5
    intercept = 0.0

    m = WDROProblem.P.m
    rho =  WDROProblem.rho

    def custom_cost(pbm):
        if NoLabels:
            return lambda samples_x, z_x, *_: pbm.c(samples_x, z_x)
        else:
            return lambda samples_x, z_x, samples_y, z_y: pbm.c(samples_x, z_x, samples_y, z_y)
    c = custom_cost(WDROProblem)

    lamL = 1e-10
    lamU = 1e10
    lam = (lamL+lamU)/2.0

    lam_eps = 1e-2

    z = np.zeros((m,d,Nsamples))
    sigma = epsilon
    for i in range(m):
        for j in range(Nsamples):
            tz = sigma*np.random.randn(d)+samples[i]
            while(np.max(tz)>WDROProblem.Xi_bounds[1] and np.min(tz)< WDROProblem.Xi_bounds[0]):
                tz = sigma*np.random.randn(d)+samples[i]
            z[i,:,j] = tz

    if not NoLabels:
        zl = np.zeros((m,Nsamples))
        sigma = epsilon
        for i in range(m):
            for j in range(Nsamples):
                tz = sigma*np.random.randn()+labels[i]
                while(np.max(tz)>WDROProblem.Xi_bounds[1] and np.min(tz)< WDROProblem.Xi_bounds[0]):
                    tz = sigma*np.random.randn()+labels[i]
                zl[i,j] = tz
    else:
        zl = np.zeros((m, Nsamples))

        # Plunge \Xi^d in \Xi^{d+1} w/ the labels
        # z = np.concatenate((z, zl[:, None, :]), axis=1) # Concatenate label as "dim d+1"
        # samples = np.concatenate((samples, labels[:, None]), axis=1) # Concatenate labels at dim d+1
        # d += 1





    T = 50
    # bar = progressbar.ProgressBar(max_value=T,widgets=widgets).start()
    for t in range(T):
        lam = (lamL+lamU)/2.0

        # some Gradient step on theta / intercept
        K = 5
        for k in range(K):
            grad_est = np.zeros(n)

            if fit_intercept:
                grad_estI = 0.0

            for i in range(m):

                if NoLabels: # No labels (and no intercept)
                    b = np.zeros(Nsamples)
                    for j in range(Nsamples):
                        b[j] = (WDROProblem.loss.value(theta , z[i,:,j]) -lam*(c(samples[i],z[i,:,j], None, None)))/epsilon

                    a = np.zeros(Nsamples)
                    for j in range(Nsamples):
                        a[j] = WDROProblem.loss.grad_theta(theta, z[i,:,j])

                    comp_grad = weightedExpAverage(a,b)

                else:
                    if fit_intercept: # labels and intercept
                        b = np.zeros(Nsamples)
                        for j in range(Nsamples):
                            b[j] = (WDROProblem.loss.valueSplit(theta , z[i,:,j],zl[i,j],intercept=intercept) -lam*c(samples[i],z[i,:,j], labels[i], zl[i, j]))/epsilon

                        a = np.zeros((Nsamples,n))
                        for j in range(Nsamples):
                            a[j,:] = WDROProblem.loss.grad_thetaSplit(theta, z[i,:,j],zl[i,j],intercept=intercept)


                        comp_grad = weightedExpAverage(a,b)


                        aI = np.zeros(Nsamples)
                        for j in range(Nsamples):
                            aI[j] = WDROProblem.loss.grad_interceptSplit(theta, z[i,:,j],zl[i,j],intercept=intercept)

                        comp_gradI = weightedExpAverage(aI,b)

                    else: # labels but no intercept
                        b = np.zeros(Nsamples)
                        for j in range(Nsamples):
                            b[j] = (WDROProblem.loss.valueSplit(theta , z[i,:,j],zl[i,j]) -lam*c(samples[i],z[i,:,j], labels[i], zl[i, j]))/epsilon

                        a = np.zeros((Nsamples,n))
                        for j in range(Nsamples):
                            a[j,:] = WDROProblem.loss.grad_thetaSplit(theta, z[i,:,j],zl[i,j])


                        comp_grad = weightedExpAverage(a,b)


                grad_est += comp_grad/m

                if fit_intercept:
                    grad_estI += comp_gradI/m


            step = 1/(t+k+10)**0.8
            theta = np.minimum(np.maximum(theta - step*grad_est,WDROProblem.Theta_bounds[0]),WDROProblem.Theta_bounds[1])

            if fit_intercept:
                intercept = intercept - step*grad_estI

        # Gradient computation for lambda
        d = 0
        for i in range(m):

            if NoLabels:
                b = np.zeros(Nsamples)
                for j in range(Nsamples):
                    b[j] = (WDROProblem.loss.value(theta , z[i,:,j]) -lam*c(samples[i],z[i,:,j], None, None))/epsilon

                a = np.zeros(Nsamples)
                for j in range(Nsamples):
                    a[j] = c(samples[i],z[i,:,j], None, None)

                comp_grad = weightedExpAverage(a,b)
            else:
                if fit_intercept: # labels and intercept
                    b = np.zeros(Nsamples)
                    for j in range(Nsamples):
                        b[j] = (WDROProblem.loss.valueSplit(theta , z[i,:,j],zl[i,j],intercept=intercept) -lam*c(samples[i],z[i,:,j], labels[i], zl[i, j]))/epsilon

                    a = np.zeros(Nsamples)
                    for j in range(Nsamples):
                        a[j] = c(samples[i],z[i,:,j], labels[i], zl[i, j])

                else:
                    b = np.zeros(Nsamples)
                    for j in range(Nsamples):
                        b[j] = (WDROProblem.loss.valueSplit(theta , z[i,:,j],zl[i,j]) -lam*c(samples[i],z[i,:,j], labels[i], zl[i, j]))/epsilon

                    a = np.zeros(Nsamples)
                    for j in range(Nsamples):
                        a[j] = c(samples[i],z[i,:,j], labels[i], zl[i, j])

                comp_grad = weightedExpAverage(a,b)

            d += comp_grad/m

        a = rho - d

        #lam = lam - K*step*a
        #print(a)
        if a>0:
            lamU = lam
        else:
            lamL = lam

        #print(theta,lam)

        #if fit_intercept:
        #    print(intercept)

        #if lamU-lamL<lam_eps:
            #print(grad_est)
            # break

        # bar.update(t)

    # bar.finish("Done")

    return theta, intercept, lam


def PiatEtAl(WDROProblem=None, epsilon=1.0, Nsamples = 5):

    theta = np.random.rand(WDROProblem.n)*10

    m = WDROProblem.P.m

    T = 100
    for t in range(T):
        u = (np.random.rand(Nsamples)-0.5)*2*np.sqrt(2*WDROProblem.rho)

        d = 0.0
        for i in range(m):
            zi = WDROProblem.P.samples[i]

            b = np.zeros(Nsamples)
            for l in range(Nsamples):
                b[l] = WDROProblem.loss.value(theta , zi + u[l] )/epsilon

            a = np.zeros(Nsamples)
            for j in range(Nsamples):
                a[j] = WDROProblem.loss.grad_theta(theta, zi + u[j] )

            delg = weightedExpAverage(a,b)


            d += delg/m

        step = 10/(t+10)**0.8
        theta = np.minimum(np.maximum(theta - step*d,WDROProblem.Theta_bounds[0]),WDROProblem.Theta_bounds[1])



    return theta

