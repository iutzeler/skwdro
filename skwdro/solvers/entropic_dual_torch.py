import torch
import torch as pt
import torch.optim as optim

import math
from skwdro.solvers.oracle_torch import entropic_loss_oracle

from skwdro.solvers.utils import *
from skwdro.base.problems import WDROProblem

# import progressbar
# widgets = [' [',
#          progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
#          '] ',
#            progressbar.Bar('*'),' (',
#            progressbar.ETA(), ') ',
#           ]


def WDROEntropicSolver(WDROProblem=None, epsilon=1e-2, Nsamples = 10,fit_intercept=False):
    return approx_BFGS(WDROProblem=WDROProblem, epsilon=epsilon, n_samples=Nsamples, fit_intercept=fit_intercept)
    # return Approx_BFGS(WDROProblem=WDROProblem, epsilon=epsilon, Nsamples = Nsamples,fit_intercept=fit_intercept)

def approx_BFGS(WDROProblem:WDROProblem, epsilon: pt.Tensor=pt.tensor(.1), n_samples: int=10, fit_intercept: bool=False):
    """ Approximation and then BFGS"""

    n = WDROProblem.n
    d = WDROProblem.d

    m = WDROProblem.P.m
    rho = WDROProblem.rho
    if isinstance(rho, float): rho = pt.tensor(rho)
    if isinstance(epsilon, float): epsilon = pt.tensor(epsilon)

    c = WDROProblem.c

    NoLabels = WDROProblem.dLabel == 0

    if NoLabels:
        xi = torch.Tensor(WDROProblem.P.samples)
        xi_labels = None
    else:
        xi = torch.Tensor(WDROProblem.P.samplesX)
        xi_labels  = torch.Tensor(WDROProblem.P.samplesY)
    kappa = pt.tensor(1e3)# controls the relative weight between points and labels

    # Init
    theta = torch.normal(0,1,size=(n,))
    theta.requires_grad = True

    intercept = torch.Tensor([0.0])
    lam = torch.Tensor([1.0/rho])
    lam.requires_grad = True


    # [WIP] init of zeta left for loss
    # ================================
    # z = torch.zeros((m,d,n_samples))
    # sigma = epsilon
    # for i in range(m):
    #     for j in range(n_samples):
    #         tz = sigma*torch.normal(0,1,size=(d,))+xi[i]
    #         while(torch.max(tz)>WDROProblem.Xi_bounds[1] and torch.min(tz)< WDROProblem.Xi_bounds[0]):
    #             tz = sigma*torch.normal(0,1,size=(d,))+xi[i]
    #         z[i,:,j] = tz

    # if not NoLabels:
    #     zl = torch.zeros((m,n_samples))
    #     sigma = 0.0
    #     for i in range(m):
    #         for j in range(n_samples):
    #             tz = sigma*torch.normal(0,1,size=(1,))+labels[i]
    #             while(torch.max(tz)>WDROProblem.Xi_bounds[1] and torch.min(tz)< WDROProblem.Xi_bounds[0]):
    #                 tz = sigma*torch.normal(0,1,size=(1,))+labels[i]
    #             zl[i,j] = tz
    # =================================
    loss = WDROProblem.loss
    assert loss is not None
    if loss._sampler is None:
        loss.sampler = loss.default_sampler(xi, xi_labels, epsilon)
    zeta, zeta_labels = WDROProblem.loss.sample_pi0(n_samples)
    zeta = zeta.swapdims(0, 2).swapdims(0, 1)
    zeta.clip(*WDROProblem.Xi_bounds)

    def EntropicProblem( theta, lam, intercept=0.0, rho=rho, epsilon=epsilon ):
        if not NoLabels: zeta_labels = zeta_labels[..., :]
        #if lam < 0:
        #    return torch.inf

        loss = lam*rho

        for i in range(m):
            integrand = torch.zeros(n_samples)
            if NoLabels: # No labels (and no intercept)
                for j in range(n_samples):
                    integrand[j] = (WDROProblem.loss.value(theta , zeta[i,:,j]) -lam*(c(xi[i],zeta[i,:,j])))/epsilon
                loss += (torch.logsumexp(integrand,0) - math.log(n_samples))*epsilon/m
            else: # w/labels (and no intercept)
                for j in range(n_samples):
                    integrand[j] = (WDROProblem.loss.value(theta , zeta[i,:,j],zeta_labels[i,j]) -lam*(c(xi[i],zeta[i,:,j])+kappa*c(xi_labels[i],zeta_labels[i,j])))/epsilon
                loss += (torch.logsumexp(integrand,0) - math.log(n_samples))*epsilon/m

        return loss

    # L-BFGS w/ PyTorch

    lbfgs = optim.LBFGS([theta,lam],
                        history_size=10,
                        max_iter=100,
                        tolerance_grad = 1e-4,
                        line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        objective = EntropicProblem( theta, lam)
        objective.backward()
        return objective

    T = 1
    for t in range(T):
        lbfgs.step(closure)
        # print(theta,lam)

    return theta.detach().numpy(), intercept.detach().numpy(), lam.detach().numpy()


def Approx_BFGS(WDROProblem=None, epsilon=0.1, Nsamples = 10,fit_intercept=False):
    """ Approximation and then BFGS"""

    n = WDROProblem.n
    d = WDROProblem.d

    m = WDROProblem.P.m
    rho = WDROProblem.rho

    c = WDROProblem.c

    NoLabels = WDROProblem.dLabel == 0

    if NoLabels:
        samples = torch.Tensor(WDROProblem.P.samples)
    else:
        samples = torch.Tensor(WDROProblem.P.samplesX)
        labels  = torch.Tensor(WDROProblem.P.samplesY)
        kappa   = 1e3 # controls the relative weight between points and labels

    # Init
    theta = torch.normal(0,1,size=(n,))
    theta.requires_grad = True

    intercept = torch.Tensor([0.0])
    lam = torch.Tensor([1.0/rho])
    lam.requires_grad = True


    z = torch.zeros((m,d,Nsamples))
    sigma = epsilon
    for i in range(m):
        for j in range(Nsamples):
            tz = sigma*torch.normal(0,1,size=(d,))+samples[i]
            while(torch.max(tz)>WDROProblem.Xi_bounds[1] and torch.min(tz)< WDROProblem.Xi_bounds[0]):
                tz = sigma*torch.normal(0,1,size=(d,))+samples[i]
            z[i,:,j] = tz
    
    if not NoLabels:
        zl = torch.zeros((m,Nsamples))
        sigma = 0.0
        for i in range(m):
            for j in range(Nsamples):
                tz = sigma*torch.normal(0,1,size=(1,))+labels[i]
                while(torch.max(tz)>WDROProblem.Xi_bounds[1] and torch.min(tz)< WDROProblem.Xi_bounds[0]):
                    tz = sigma*torch.normal(0,1,size=(1,))+labels[i]
                zl[i,j] = tz      
 
    def EntropicProblem( theta, lam, intercept=0.0, rho=rho, epsilon=epsilon ):

        if lam < 0:
            return torch.inf

        loss = lam*rho 
        
        # for i in range(m):
        #         integrand = 0
        #         if NoLabels: # No labels (and no intercept)
        #             for j in range(Nsamples):
        #                 integrand += 1/Nsamples*torch.exp((WDROProblem.loss.value(theta , z[i,:,j]) -lam*(c(samples[i],z[i,:,j])))/epsilon)
        #             loss += torch.log(integrand)*epsilon

        for i in range(m):
            integrand = torch.zeros(Nsamples)
            if NoLabels: # No labels (and no intercept)
                for j in range(Nsamples):
                    integrand[j] = (WDROProblem.loss.value(theta , z[i,:,j]) -lam*(c(samples[i],z[i,:,j])))/epsilon
                loss += (torch.logsumexp(integrand,0) - math.log(Nsamples))*epsilon/m
            else: # w/labels (and no intercept)
                for j in range(Nsamples):
                    integrand[j] = (WDROProblem.loss.value(theta , z[i,:,j],zl[i,j]) -lam*(c(samples[i],z[i,:,j])+kappa*c(labels[i],zl[i,j])))/epsilon
                loss += (torch.logsumexp(integrand,0) - math.log(Nsamples))*epsilon/m

        return loss
    


    # L-BFGS w/ PyTorch


    lbfgs = optim.LBFGS([theta,lam],
                        history_size=10, 
                        max_iter=50,
                        tolerance_grad = 1e-4,
                        line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        objective = EntropicProblem( theta, lam)
        objective.backward()
        return objective

    # theta = torch.normal(0,1,size=(n,),dtype=torch.float64)
    # theta.requires_grad = True

    # intercept = torch.Tensor([0.0])
    # intercept.requires_grad = True


    T = 2
    # bar = progressbar.ProgressBar(max_value=T,widgets=widgets).start()
    for t in range(T):
        lbfgs.step(closure)
        
        # print(theta,lam)
    
        # bar.update(t)

    #bar.finish("Done")

    
    return theta.detach().numpy(), intercept.detach().numpy(), lam.detach().numpy()

