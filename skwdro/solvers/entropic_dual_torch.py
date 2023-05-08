import torch
import torch.optim as optim

import math

from skwdro.solvers.utils import *

# import progressbar
# widgets = [' [',
#          progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
#          '] ',
#            progressbar.Bar('*'),' (',
#            progressbar.ETA(), ') ',
#           ]


def WDROEntropicSolver(WDROProblem=None, epsilon=1e-2, Nsamples = 10,fit_intercept=False):
    return Approx_BFGS(WDROProblem=WDROProblem, epsilon=epsilon, Nsamples = Nsamples,fit_intercept=fit_intercept)


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

