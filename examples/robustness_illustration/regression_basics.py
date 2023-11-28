import numpy as np

from illustration_common import getEmpiricalAndTrueRisk, plotRiskHistograms

from skwdro.linear_models import LinearRegression as wdroLR
from sklearn.linear_model import LinearRegression as skLR

from sklearn.metrics import mean_squared_error

def generateRegressionData(n=100,d=20,noise=0.1,random_seed=42):

    params = {"n":n,"d":d,"noise":noise,"rnd_seed":random_seed}


    # x0 and co. are drawn consistently from the input seed ...
    np.random.seed(random_seed)

    x0 = np.random.randn(d)
    
    mean = np.random.randn(d)

    M    = np.random.randn(d,d)
    cov  = M.dot(M.T)

    params["x0"] = x0
    
    # ... the rest is purely random
    np.random.seed()

    X = np.random.multivariate_normal(mean=mean,cov=cov,size=n)

    y = X.dot(x0) + np.sqrt(noise)*np.random.randn(n)

    return X,y


# X,y = generateRegressionData(n=5,d=2)

# print(X,y)


NewData = True

if NewData:
    estimatorsList = [skLR(),wdroLR(solver="dedicated",rho=1.0)]
    empRisk,trueRisk = getEmpiricalAndTrueRisk(generateRegressionData,estimatorsList,score=mean_squared_error,nTrain=30,nTrials=100)
    np.savez("risks.npz",empRisk=empRisk,trueRisk=trueRisk)
else:
    data = np.load("risks.npz")
    empRisk = data["empRisk"]
    trueRisk = data["trueRisk"]



plotRiskHistograms(empRisk=empRisk,trueRisk=trueRisk)
