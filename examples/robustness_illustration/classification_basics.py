import numpy as np

from illustration_common import getEmpiricalAndTrueRisk, plotRiskHistograms

from skwdro.linear_models import LogisticRegression as wdroLR
from sklearn.linear_model import LogisticRegression as skLR

from sklearn.metrics import log_loss

def generateClassificationData(n=100,d=5,noise=0.1,random_seed=42, shift=None):

    params = {"n":n,"d":d,"noise":noise,"rnd_seed":random_seed}


    # x0 and co. are drawn consistently from the input seed ...
    np.random.seed(random_seed)

    x0 = np.random.randn(d)
    
    mean = np.random.randn(d)

    M    = np.random.randn(d,d)
    cov  = M.dot(M.T)

    params["x0"] = x0


    if shift is not None:
        mean = mean + shift
    
    # ... the rest is purely random
    np.random.seed()

    X = np.random.multivariate_normal(mean=mean,cov=cov,size=n)

    y = np.sign(X.dot(x0) + np.sqrt(noise)*np.random.randn(n))

    return X,y


# X,y = generateRegressionData(n=5,d=2)

# print(X,y)


NewData = True

if NewData:
    estimatorsList = [skLR(penalty=None),wdroLR(rho=1.0)]
    empRisk,trueRisk = getEmpiricalAndTrueRisk(generateClassificationData,estimatorsList,score=log_loss,nTrain=30,nTrials=50)
    np.savez("./risks.npz",empRisk=empRisk,trueRisk=trueRisk)
else:
    data = np.load("./risks.npz")
    empRisk = data["empRisk"]
    trueRisk = data["trueRisk"]



plotRiskHistograms(empRisk=empRisk,trueRisk=trueRisk)
