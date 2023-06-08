import numpy as np
from numpy.random import sample

from skwdro.solvers.utils import *
from skwdro.solvers.optim_cond import OptCond
from skwdro.solvers.gradient_estimates import step_wgx_wol, step_wgx_wl

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



# ### Gradient iterations ######################################
def wgx_v2_wo_labels(d, data_structure, m, rho_eps, lam_0, n_samples, cost, loss_fns, fit_intercept, opt_cond):
    """
    Launch the optimization, stopping when opt_cond is fulfilled, when the model does not penalize label switching.
    """
    xi, theta, zeta = prepare_data(data_structure.samples, m, d, n_samples, rho_eps[1], fit_intercept)

    c = lambda samples_x, z_x, *_: cost(samples_x, z_x)

    t = 1
    lam = lam_0
    while t > 0:
        # Perform step
        theta, lam, grads = step_wgx_wol(xi, zeta, theta, lam, c, loss_fns, t, rho_eps)
        # Check stopping on the 5th theta gradient and last lambda projected gradient
        if opt_cond(grads, t): break
        else: t += 1
    return theta, lam


def wgx_v2_w_labels(d, data_structure, m, rho_eps, lam_0, n_samples, cost, loss, fit_intercept, opt_cond):
    """
    Launch the optimization, stopping when opt_cond is fulfilled, when the model does not penalize label switching.
    """
    xi, theta, zeta = prepare_data(data_structure.samplesX, m, d, n_samples, rho_eps[1], fit_intercept)
    xi_labels = data_structure.samplesY
    zeta_labels = sample_pi_0(rho_eps[1], n_samples, xi_labels)

    c = lambda samples_x, z_x, samples_y, z_y: cost(samples_x, z_x, samples_y, z_y)

    t = 1
    lam = lam_0
    while t > 0:
        # Perform step
        theta, lam, grads = step_wgx_wl(xi, xi_labels, zeta, zeta_labels, theta, lam, c, loss, t, rho_eps)
        # Check stopping on the 5th theta gradient and last lambda projected gradient
        if opt_cond(grads, t): break
        else: t += 1
    return theta, lam
# ##############################################################

def WangGaoXie_v2(WDROProblem, epsilon=0.1, n_samples=50, fit_intercept=False, opt_cond=OptCond(2)):
    r"""
    Full method to solve the dual problem of entropy-regularized WDRO.

    .. math::
        \inf_{\lambda\ge 0, \theta}
            \lambda\rho + \epsilon
                \mathbb{E}_{\xi\sim\mathbb{P}^N}
                \ln\mathbb{E}_{\zeta\sim\mathcal{N}(\xi, \sigma)}
                e^{\frac{1}{\epsilon}(L_\theta(\zeta)-\lambda c(\zeta, \xi))}

    Parameters
    ==========
    WDROProblem: WDROProblem
        Instance containing all important Parameters
    epsilon: float
        Relative importance/penalty attributed to the entropy in the primal
    n_samples: int
        Number of samples drown to approximate :math:`\mathbb{E}_{\zeta\sim\mathcal{N}(\xi, \sigma)}`
    fit_intercept: bool
        Set to true to plunge data in a ``d+1`` dimensional space in order to capture an affine model
    opt_cond: OptCond
        Instance containing information for stopping criteria wrt gradient descent
    """

    d = WDROProblem.d

    no_labels = WDROProblem.dLabel == 0

    data_structure = WDROProblem.P

    m = WDROProblem.P.m
    rho =  WDROProblem.rho

    # Starting lambda is 1/rho  
    lam = 1.0/rho

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

    # Format of output:
    # * linear coefficients (vector format)
    # * intercept (None if fit_intercept==False)
    # * optimal lambda
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

