import numpy as np
import progressbar
 


from skwdro.solvers.utils import *


widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]

 

def WDROEntropicSolver(WDROProblem=None, epsilon=1e-2, Nsamples = 20,fit_intercept=False):
    return WangGaoXie_v1(WDROProblem=WDROProblem, epsilon=epsilon, Nsamples = Nsamples,fit_intercept=fit_intercept)


def WangGaoXie_v1(WDROProblem=None, epsilon=0.1, Nsamples = 50,fit_intercept=False):
    """ Algorithm of Wang et al. but with epsilon >0 and delta = 0 (regularization in the objective)"""

    n = WDROProblem.n
    d = WDROProblem.d

    NoLabels = WDROProblem.dLabel == 0

    if NoLabels:
        samples = WDROProblem.P.samples
    else:
        samples = WDROProblem.P.samplesX
        labels  = WDROProblem.P.samplesY
    

    theta = np.random.rand(n)*0.5
    intercept = 0.0

    m = WDROProblem.P.m
    rho =  WDROProblem.rho

    c = WDROProblem.c
    kappa = 1000

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
 

    


    T = 50
    bar = progressbar.ProgressBar(max_value=T,widgets=widgets).start()
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
                        b[j] = (WDROProblem.loss.value(theta , z[i,:,j]) -lam*(c(samples[i],z[i,:,j])))/epsilon
                    
                    a = np.zeros(Nsamples)
                    for j in range(Nsamples):
                        a[j] = WDROProblem.loss.grad_theta(theta, z[i,:,j])
                    
                    comp_grad = weightedExpAverage(a,b)   

                else:      
                    if fit_intercept: # labels and intercept
                        b = np.zeros(Nsamples)                  
                        for j in range(Nsamples):
                            b[j] = (WDROProblem.loss.valueSplit(theta , z[i,:,j],zl[i,j],intercept=intercept) -lam*(c(samples[i],z[i,:,j]) + kappa*np.abs(zl[i,j]-labels[i])))/epsilon
                        
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
                            b[j] = (WDROProblem.loss.valueSplit(theta , z[i,:,j],zl[i,j]) -lam*(c(samples[i],z[i,:,j]) + kappa*np.abs(zl[i,j]-labels[i])))/epsilon
                        
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
                    b[j] = (WDROProblem.loss.value(theta , z[i,:,j]) -lam*(c(samples[i],z[i,:,j]) ))/epsilon
                
                a = np.zeros(Nsamples)
                for j in range(Nsamples):
                    a[j] = c(samples[i],z[i,:,j]) 
                
                comp_grad = weightedExpAverage(a,b)
            else:
                if fit_intercept: # labels and intercept
                    b = np.zeros(Nsamples)
                    for j in range(Nsamples):
                        b[j] = (WDROProblem.loss.valueSplit(theta , z[i,:,j],zl[i,j],intercept=intercept) -lam*(c(samples[i],z[i,:,j]) + kappa*np.abs(zl[i,j]-labels[i])))/epsilon
                    
                    a = np.zeros(Nsamples)
                    for j in range(Nsamples):
                        a[j] = c(samples[i],z[i,:,j]) +  kappa*np.abs(zl[i,j]-labels[i])
                               
                else:
                    b = np.zeros(Nsamples)
                    for j in range(Nsamples):
                        b[j] = (WDROProblem.loss.valueSplit(theta , z[i,:,j],zl[i,j]) -lam*(c(samples[i],z[i,:,j]+ kappa*np.abs(zl[i,j]-labels[i]))))/epsilon
                    
                    a = np.zeros(Nsamples)
                    for j in range(Nsamples):
                        a[j] = c(samples[i],z[i,:,j]) +  kappa*np.abs(zl[i,j]-labels[i])
                
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

        bar.update(t)

    bar.finish("Done")
    
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

        print(theta)

    
    return theta

