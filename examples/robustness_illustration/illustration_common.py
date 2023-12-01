"""
===========================
Illustration of Robustness
===========================
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def getEmpiricalAndTrueRisk(generateData,estimatorsList,score,nTrain,nTrials):

    nTest = nTrain*1000
    Xtest,ytest = generateData(n=nTest,shift=.05)

    nEst = len(estimatorsList)

    empRisk = np.zeros((nTrials,nEst))
    trueRisk = np.zeros((nTrials,nEst))

    for nt in range(nTrials):

        print("Trial {:6d}/{:6d}".format(nt+1,nTrials), end='\r')

        Xtrain,ytrain = generateData(n=nTrain)

        for i,est in enumerate(estimatorsList):
            est.fit(Xtrain,ytrain)
            pred_train = est.predict(Xtrain)
            pred_test  = est.predict(Xtest)

            empRisk[nt,i] = score(pred_train,ytrain)
            trueRisk[nt,i] = score(pred_test,ytest)

    

    return empRisk,trueRisk

            

def plotRiskHistograms(empRisk,trueRisk):

    nTrial = empRisk.shape[0]
    nEst   = empRisk.shape[1]

    left  = min(np.min(np.min(empRisk)),np.min(np.min(trueRisk)))
    right = max(np.max(np.max(empRisk)),np.max(np.max(trueRisk)))

    nBins  = int(nTrial/5)

    bins = np.linspace(left, right, num=nBins+1)

    print(left,right,bins)


    for i in range(nEst): 

        plt.figure()

        plt.hist(empRisk[:,i],color='r',density=True,label="Empirical risk")
        plt.hist(trueRisk[:,i],color='b',density=True,label="True risk")

        plt.xlim([left,right])

        plt.legend()

        plt.savefig("./Estimator_{}.png".format(i+1))
        plt.show()










