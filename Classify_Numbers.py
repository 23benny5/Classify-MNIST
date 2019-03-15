#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 14:56:06 2017

@author: bennyng_211
"""
import numpy as np
from plotBoundary import *
import pylab as plt
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn import linear_model

from cvxopt import matrix, solvers
import pylab as pl
from math import pi, e
from sklearn import linear_model, svm
import matplotlib.pyplot as plt
import pdb


def getMnistNo(datanumber):
    name = str(datanumber)
    # load data from csv files
    train = pl.loadtxt('/mnist_digit_'+name+'.csv')
    #Xtrain = train[:,0:2]
    #Ytrain = train[:,2:3]
    return train#Xtrain, Ytrain

Data0= getMnistNo(0)
Data1= getMnistNo(1)
Data2= getMnistNo(2)
Data3= getMnistNo(3)
Data4= getMnistNo(4)
Data5= getMnistNo(5)
Data6= getMnistNo(6)
Data7 = getMnistNo(7)
Data8 = getMnistNo(8)
Data9 = getMnistNo(9)

#Data0[0:200,:].shape
#np.append(Data0[0:200,:],Data7[0:200,:]).reshape(-1,784).shape

def GetData(input1,input2):
    y1 = np.ones(len(input1))
    y2 = -1* np.ones(len(input2))
    
    Train1 = input1[0:200,:]
    Train2 = input2[0:200,:]
    Valid1 = input1[200:350,:]
    Valid2 = input2[200:350,:]
    Test1 = input1[350:500,:]
    Test2 = input2[350:500,:]
    
    Train1y = y1[0:200]
    Train2y = y2[0:200]
    Valid1y = y1[200:350]
    Valid2y = y2[200:350]
    Test1y = y1[350:500]
    Test2y = y2[350:500]
    
    TraindataX = np.append(Train1,Train2).reshape(-1,784)
    TraindataY = np.append(Train1y,Train2y).reshape(-1,1)
    ValiddataX = np.append(Valid1,Valid2).reshape(-1,784)
    ValiddataY = np.append(Valid1y,Valid2y).reshape(-1,1)
    TestdataX = np.append(Test1,Test2).reshape(-1,784)
    TestdataY = np.append(Test1y,Test2y).reshape(-1,1)
    
    return TraindataX, TraindataY, ValiddataX, ValiddataY, TestdataX,TestdataY


Xtrain,Ytrain,Xvalid,Yvalid,Xtest,Ytest = GetData(Data1,Data7)
Xtrain,Ytrain,Xvalid,Yvalid,Xtest,Ytest = GetData(Data3,Data5)
Xtrain,Ytrain,Xvalid,Yvalid,Xtest,Ytest = GetData(Data4,Data9)

XtrainE1,YtrainE1,XvalidE1,YvalidE1,XtestE1,YtestE1 = GetData(Data0,Data1)
XtrainE2,YtrainE2,XvalidE2,YvalidE2,XtestE2,YtestE2 = GetData(Data2,Data3)
XtrainE3,YtrainE3,XvalidE3,YvalidE3,XtestE3,YtestE3 = GetData(Data4,Data5)
XtrainE4,YtrainE4,XvalidE4,YvalidE4,XtestE4,YtestE4 = GetData(Data6,Data7)
XtrainE5,YtrainE5,XvalidE5,YvalidE5,XtestE5,YtestE5 = GetData(Data8,Data9)

Xtrain = np.append(Xtrain,XtrainE5).reshape(-1,784)
Ytrain = np.append(Ytrain,YtrainE5).reshape(-1,1)
Xtest = np.append(Xtest,XtestE5).reshape(-1,784)
Ytest = np.append(Ytest,YtestE5).reshape(-1,1)

Xvalid = np.append(Xvalid,XvalidE5).reshape(-1,784)
Yvalid = np.append(Yvalid,YvalidE5).reshape(-1,1)

#float(2)*Data0/255-1
#Data1

def GetDataNorm(input1,input2):
    y1 = np.ones(len(input1))
    y2 = -1* np.ones(len(input2))
    input1Norm = float(2)*input1/255-1
    input2Norm = float(2)*input2/255-1
    
    Train1 = input1Norm[0:200,:]
    Train2 = input2Norm[0:200,:]
    Valid1 = input1Norm[200:350,:]
    Valid2 = input2Norm[200:350,:]
    Test1 = input1Norm[350:500,:]
    Test2 = input2Norm[350:500,:]
    
    Train1y = y1[0:200]
    Train2y = y2[0:200]
    Valid1y = y1[200:350]
    Valid2y = y2[200:350]
    Test1y = y1[350:500]
    Test2y = y2[350:500]
    
    TraindataX = np.append(Train1,Train2).reshape(-1,784)
    TraindataY = np.append(Train1y,Train2y).reshape(-1,1)
    ValiddataX = np.append(Valid1,Valid2).reshape(-1,784)
    ValiddataY = np.append(Valid1y,Valid2y).reshape(-1,1)
    TestdataX = np.append(Test1,Test2).reshape(-1,784)
    TestdataY = np.append(Test1y,Test2y).reshape(-1,1)
    
    return TraindataX, TraindataY, ValiddataX, ValiddataY, TestdataX,TestdataY    

XtrainNorm,YtrainNorm,XvalidNorm,YvalidNorm,XtestNorm,YtestNorm = GetDataNorm(Data1,Data7)
XtrainNorm,YtrainNorm,XvalidNorm,YvalidNorm,XtestNorm,YtestNorm = GetDataNorm(Data3,Data5)
XtrainNorm,YtrainNorm,XvalidNorm,YvalidNorm,XtestNorm,YtestNorm = GetDataNorm(Data4,Data9)

XtrainNorm = float(2)*Xtrain/255-1
YtrainNorm = Ytrain
XtestNorm = float(2)*Xtest/255-1
YtestNorm = Ytest

XvalidNorm = float(2)*Xvalid/255-1
YvalidNorm = Yvalid

def LR(x,y,penalty,lamda):
    
    C = 1/float(lamda)
    clf_LR = linear_model.LogisticRegression(C=C, penalty = penalty, tol=0.0001)
    clf_LR.fit(x,y)
    
    return clf_LR    

#c1 = LR()   
#Predict = c1.predict_proba(X)
    
    
def TestErrorRate (xtrain,ytrain,xtest,ytest,penalty,lamda):
    
    ErrorRates = []
    for i in range(len(lamda)):
        c1 = LR(xtrain,ytrain,penalty,lamda[i])
        
        Prediction = c1.predict(xtest)
        Correct = 0
        Wrong = 0
        for j in range(len(Prediction)):
        
            if Prediction[j] == ytest[j]:
                
                Correct = Correct+1
            else:
                Wrong = Wrong + 1
        Total = Correct+Wrong
        Error = Wrong/float(Total)
            
        ErrorRates.append(Error)
    

    return ErrorRates   

#Lambda = [0.01,0.1,1,100]
Lambda = [0.01]
TestErrorRate(Xtrain,Ytrain,Xtest,Ytest,'l1',Lambda)
TestErrorRate(XtrainNorm,YtrainNorm,XtestNorm,YtestNorm,'l1',Lambda)
TestErrorRate(Xtrain,Ytrain,Xtest,Ytest,'l2',Lambda)
TestErrorRate(XtrainNorm,YtrainNorm,XtestNorm,YtestNorm,'l2',Lambda)
      

def SolverKernel(y,x,C,Kernelfunction,gamma):
    
    K = Kernelfunction(x,x,gamma)
    #K = Kernelfunction(x,x)
    def CreateP (y,x): 
        
#        Consolidate = []
        Newy = []
        for i in range(len(y)):
            Newy.append(y[i])
        diagy = np.diagflat(Newy)
        
#        P = np.dot(np.dot(diagy,Kernelfunction(x,x)),diagy)
        P = np.dot(np.dot(diagy,K),diagy)
                        
        return P
    
    P = CreateP(y,x)
    P = matrix(P)
    
    def CreateQ(x):
        Row = []
        for i in range(len(x)):
            Row.append(float(-1))
        
            
        Q = np.matrix(Row)
        
        return Q
#    G = Q.reshape(np.shape(Q)[1],1)
#    H = np.matrix(H)
#    H = H.reshape(np.shape(H)[1],1)
     
    
    Q = CreateQ(x)
    Q = Q.T
    Q = matrix(Q)
    
    def CreateG(x):
        I1 = np.identity(len(x))
        I2 = -np.identity(len(x))
#        I = []
        for i in range(len(x)):
            I1 = np.append(I1,I2[i])
            #print I2[i]
        
        I1 = I1.reshape(2*len(x),len(x))
    #    for j in range(len(x)):
    #        I.append(I1[:,j])
    #    I = np.array(I)    
        return I1
    
    G = CreateG(x)
    G = matrix(G)
    
    def CreateH(x,C):
        H = []
        for i in range(len(x)):
            H.append(float(C))
        for i in range(len(x)):
            H.append(float(0))
        H = np.array(H)
        return H
    
    H = CreateH(x,C)
    H = matrix(H) 
    
    def CreateA(y):
        A = []
        for i in range(len(y)):
            A.append(float(y[i]))
        A = np.array(A)
        A = A.reshape(1,np.shape(A)[0])
        return A
    
    A = CreateA(y)
    A = matrix(A)
    
    b = matrix(0.0)
        
    solution = solvers.qp(P, Q, G, H, A, b)
    
    return solution


def GetW(Alpha,y,x):
    W = 0
    Yneg = []
    Ypos = []
    #count = 0
    for i in range(len(Alpha)):
        W = W + Alpha[i]*y[i]*x[i]
        #count= count+1
        
    for j in range(len(x)):
        if y[j] > 0:    
            Ypos.append(np.dot(W,x[j]))
        else:
            Yneg.append(np.dot(W,x[j]))
    W0 = (max(Yneg) + min(Ypos))*(-0.5)
#    index1 = ER1.index(min(ER1))
#    l2minER = ER1[index1]
#    l2minLambda = LambdaER1[index1]        
    return W, W0

#Data1Sol = SolverKernel(Ytrain,Xtrain,0.01,linear_kernel)
#GetW(Data1Sol['x'],Ytrain,Xtrain)

def SVMErrorRate (xtrain,ytrain,xtest,ytest):

    Data1Sol = SolverKernel(ytrain,xtrain,0.01,linear_kernel,0)
    W,W0 = GetW(Data1Sol['x'],ytrain,xtrain)
    
    #ErrorRates = [] 
    
    def predictvalue(W,W0,xtest):
        ypredict = []
        for i in range(len(xtest)):
            if np.dot(W,xtest[i]) + W0 >=0:
                ypredict.append(1.0)
            else:
                ypredict.append(-1.0)
        
        return ypredict
    
    Prediction = predictvalue(W,W0,xtest)
    
    Correct = 0
    Wrong = 0
    for j in range(len(Prediction)):
        if Prediction[j] == ytest[j]:
            Correct = Correct+1
        else:
            Wrong = Wrong + 1
    Total = Correct+Wrong
    Error = Wrong/float(Total)
        
    #ErrorRates.append(Error)

    return  Error

1- SVMErrorRate (Xtrain,Ytrain,Xtest,Ytest)
1-SVMErrorRate (XtrainNorm,YtrainNorm,XtestNorm,YtestNorm)


def svmRBFerrorRate (xtrain,ytrain,xtest,ytest,C,Gamma):

    Data1Sol = SolverKernel(ytrain,xtrain,C,rbf_kernel,Gamma)
    W,W0 = GetW(Data1Sol['x'],ytrain,xtrain)
    
    #ErrorRates = [] 
    
    def predictvalue(W,W0,xtest):
        ypredict = []
        for i in range(len(xtest)):
            if np.dot(W,xtest[i]) + W0 >=0:
                ypredict.append(1.0)
            else:
                ypredict.append(-1.0)
        
        return ypredict
    
    Prediction = predictvalue(W,W0,xtest)
    
    Correct = 0
    Wrong = 0
    for j in range(len(Prediction)):
        if Prediction[j] == ytest[j]:
            Correct = Correct+1
        else:
            Wrong = Wrong + 1
    Total = Correct+Wrong
    Error = Wrong/float(Total)

    return  Error

1-svmRBFerrorRate(Xtrain,Ytrain,Xtest,Ytest,0.01,0.25)
1-svmRBFerrorRate(XtrainNorm,YtrainNorm,XtestNorm,YtestNorm,1,0.5)


C = np.logspace(-2,2,5)
gamma =  list(np.logspace(2,-2,5, base = 2))

def OptimizeRBF(xtrain,ytrain,xvalid,yvalid,C,gamma):
    Values = []
    Error = []
    for i in range(len(C)):
        for j in range(len(gamma)):
            E = svmRBFerrorRate(xtrain,ytrain,xvalid,yvalid,C[i],gamma[j])
            Error.append(E)
            Values.append((C[i],gamma[j]))
    index = Error.index(min(Error))
    
    return Error[index], Values[index]


Error, Value = OptimizeRBF(Xtrain,Ytrain,Xvalid,Yvalid,C,gamma)
ErrorN, ValueN = OptimizeRBF(XtrainNorm,YtrainNorm,XvalidNorm,YvalidNorm,C,gamma)
ErrorN2, ValueN2 = OptimizeRBF(XtrainNorm,YtrainNorm,XvalidNorm,YvalidNorm,C,gamma)


"""Compare Pegasos QP"""
"RBF"
    
def PegasosKernalRBF(x,y,lamda,Kernal,max_epochs,Gamma):
    t = 0
    Alpha = np.zeros(x.shape[0])
    epoch = 0
    K = Kernal(x,x,gamma = Gamma)
    while epoch < max_epochs:
        for i in range(len(x)):
            t = t+1
            step = (t*float(lamda))**(-1)
            
            if y[i]*(np.dot(K[i],Alpha)) < 1:
                Alpha[i] = (1-step*lamda)*Alpha[i] + step*y[i]
                
            else:
                Alpha[i] = (1-step*lamda)*Alpha[i]
        epoch = epoch + 1
    return Alpha

Alpha = PegasosKernalRBF(Xtrain,Ytrain,0.02,rbf_kernel, 1000,4)

def predict_gaussianSVM(xnew,Alpha,Gamma,xtrain):
    return (np.dot(Alpha, rbf_kernel(xtrain,xnew.reshape(1,-1), Gamma)))

def PegasosErrorRateSVM_RBF(xtrain,ytrain,xtest,ytest,alpha,gamma):
    correct = 0
    wrong = 0
    for i in range(len(Xtest)):
        if ytest[i]*np.sign(predict_gaussianSVM(xtest[i],alpha,gamma,xtrain)) < 1:
            wrong +=1
            #print ytest
            #print np.sign(predict_gaussianSVM(xtest[i],alpha,gamma,xtrain))
        
        else:
            correct +=1
    
    Total = correct + wrong
    Error = wrong/float(Total)
    
    return Error
            
#np.sign(-10)
GaussianSVMerror = PegasosErrorRateSVM_RBF(Xtrain,Ytrain,Xtest,Ytest,Alpha,4)
GaussianSVMerror_nomalized = PegasosErrorRateSVM_RBF(XtrainNorm,YtrainNorm,XtestNorm,YtestNorm,Alpha,4)

def PegasosKernal_linear(x,y,lamda,Kernal,max_epochs):
    t = 0
    Alpha = np.zeros(x.shape[0])
    epoch = 0
    K = Kernal(x,x)
    while epoch < max_epochs:
        for i in range(len(x)):
            t = t+1
            step = (t*float(lamda))**(-1)
            
            if y[i]*(np.dot(K[i],Alpha)) < 1:
                Alpha[i] = (1-step*lamda)*Alpha[i] + step*y[i]
                
            else:
                Alpha[i] = (1-step*lamda)*Alpha[i]
        epoch = epoch + 1
    return Alpha

AlphaLin = PegasosKernal_linear(Xtrain,Ytrain,0.02,linear_kernel,1000)

def predictSVM_linear(xnew,Alpha,xtrain):
    return (np.dot(Alpha, linear_kernel(xtrain,xnew.reshape(1,-1))))

def PegasosErrorRateSVM_linear(xtrain,ytrain,xtest,ytest,alpha):
    correct = 0
    wrong = 0
    for i in range(len(Xtest)):
        if ytest[i]*np.sign(predictSVM_linear(xtest[i],alpha,xtrain)) < 1:
            wrong +=1
            #print ytest
            #print np.sign(predict_gaussianSVM(xtest[i],alpha,gamma,xtrain))
        
        else:
            correct +=1
    
    Total = correct + wrong
    Error = wrong/float(Total)
    
    return Error

LinearSVMerror = PegasosErrorRateSVM_linear(Xtrain,Ytrain,Xtest,Ytest,AlphaLin)
LinearSVMerror_Normalized = PegasosErrorRateSVM_linear(XtrainNorm,YtrainNorm,XtestNorm,YtestNorm,AlphaLin)

def GetDataMore(input1,input2,NumberOfTraining):
    y1 = np.ones(len(input1))
    y2 = -1* np.ones(len(input2))
    input1Norm = float(2)*input1/255-1
    input2Norm = float(2)*input2/255-1
    
    Train1 = input1Norm[0:NumberOfTraining,:]
    Train2 = input2Norm[0:NumberOfTraining,:]
    #Valid1 = input1[200:350,:]
    #Valid2 = input2[200:350,:]
    TestNumber = NumberOfTraining+150
    Test1 = input1Norm[NumberOfTraining:TestNumber,:]
    Test2 = input2Norm[NumberOfTraining:TestNumber,:]
    
    Train1y = y1[0:NumberOfTraining]
    Train2y = y2[0:NumberOfTraining]
    #Valid1y = y1[200:350]
    #Valid2y = y2[200:350]
    Test1y = y1[NumberOfTraining:TestNumber]
    Test2y = y2[NumberOfTraining:TestNumber]
    
    TraindataX = np.append(Train1,Train2).reshape(-1,784)
    TraindataY = np.append(Train1y,Train2y).reshape(-1,1)
    #ValiddataX = np.append(Valid1,Valid2).reshape(-1,784)
    #ValiddataY = np.append(Valid1y,Valid2y).reshape(-1,1)
    TestdataX = np.append(Test1,Test2).reshape(-1,784)
    TestdataY = np.append(Test1y,Test2y).reshape(-1,1)
    
    return TraindataX, TraindataY, TestdataX,TestdataY#, ValiddataX, ValiddataY

Xtrain_300, Ytrain_300,Xtest_300, Ytest_300 = GetDataMore(Data1,Data7,300) 
Xtrain_400, Ytrain_400,Xtest_400, Ytest_400 = GetDataMore(Data1,Data7,400) 
Xtrain_500, Ytrain_500,Xtest_500, Ytest_500 = GetDataMore(Data1,Data7,500) 

Alpha = PegasosKernalRBF(Xtrain,Ytrain,0.02,rbf_kernel, 1000,4)
Data1Sol = SolverKernel(Ytrain,Xtrain,0.01,rbf_kernel,4)

import time
Training = [[XtrainNorm,YtrainNorm],[Xtrain_300,Ytrain_300],[Xtrain_400,Ytrain_400],[Xtrain_500,Ytrain_500]]

Test = [[XtestNorm,YtestNorm],[Xtest_300,Ytest_300],[Xtest_400,Ytest_400],[Xtest_500,Ytest_500]]

Epochs = [10,100,1000]
L = [0.01,0.1,1,10] 

V = []
Time = []
Accuracy = []
for i in range(len(L)):
    
    for j in range(len(Epochs)):
        TimePeg = []
        AccuracyPeg = []
        Variable = [Epochs[j],L[i]]
        for z in range(len(Training)):
            start_time = time.time()
            
            A_Peg = PegasosKernalRBF(Training[z][0],Training[z][1],L[i],rbf_kernel, Epochs[j],0.02)
        
            TimePeg.append(time.time()-start_time)
            
            A_nomalized = PegasosErrorRateSVM_RBF(Training[z][0],Training[z][1],Test[z][0],Test[z][1],A_Peg,0.02)
            AccuracyPeg.append(1- A_nomalized)
        V.append(Variable)
        Time.append(TimePeg)
        Accuracy.append(AccuracyPeg)
            #print TimePeg
    #print time.time()-start_time
    
TimeSVM = []
AccuracySVM = []
for z in range(len(Training)):
    start_time = time.time()
    
    A_Svm = SolverKernel(Training[z][1],Training[z][0],0.01,rbf_kernel,0.02)

    TimeSVM.append(time.time()-start_time)
    
    SVM_A_nomalized = svmRBFerrorRate(Training[z][0],Training[z][1],Test[z][0],Test[z][1],0.01,0.02)
    AccuracySVM.append(1-SVM_A_nomalized)
    print TimeSVM    


f, axarr = plt.subplots(2, 2)
Datanumber = [200,300,400,500]
#Datanumber, y01 = range(10), tB[1]
#Datanumber, y10 = range(10), tB[2]
#Datanumber, y11 = range(10), tB[3]

axarr[0,0].plot(Datanumber, Time[0],label = "Epochs = 10")
axarr[0,0].plot(Datanumber, Time[1],label = "Epochs = 100")
axarr[0,0].plot(Datanumber, Time[2])
axarr[0,0].plot(Datanumber, TimeSVM)
#axarr[0,0].plot(Datanumber, Time[2],label = "Epochs = 1000")
#axarr[0,0].plot(Datanumber, TimeSVM,label = "SVM")
axarr[0,0].legend()
axarr[0,0].set_title('Lambda = 0.01')
axarr[0,0].tick_params(bottom = 'off')
axarr[0,0].get_xaxis().set_visible(False)
#axarr[0,0].set_ylabel('Classification Error')
axarr[0,0].set_xlabel('Lambda')

#axarr[0,1].plot(Datanumber, Time[3],label = "Epochs = 10")
#axarr[0,1].plot(Datanumber, Time[4],label = "Epochs = 100")
axarr[0,1].plot(Datanumber, Time[3])
axarr[0,1].plot(Datanumber, Time[4])
axarr[0,1].plot(Datanumber, Time[5],label = "Epochs = 1000")
axarr[0,1].plot(Datanumber, TimeSVM,label = "QP Benchmark")
axarr[0,1].set_title('Lambda = 0.1')
axarr[0,1].legend()
axarr[0,1].tick_params(bottom = 'off')
axarr[0,1].get_xaxis().set_visible(False)

axarr[1,0].plot(Datanumber, Time[6],label = "Epochs = 10")
axarr[1,0].plot(Datanumber, Time[7],label = "Epochs = 100")
axarr[1,0].plot(Datanumber, Time[8],label = "Epochs = 1000")
axarr[1,0].plot(Datanumber, TimeSVM,label = "QP Benchmark")
axarr[1,0].set_title('Lambda = 1')
#axarr[1,0].legend()
axarr[1,0].set_ylabel('Running Time')
axarr[1,0].set_xlabel('Dataset number')

axarr[1,1].plot(Datanumber, Time[9],label = "Epochs = 10")
axarr[1,1].plot(Datanumber, Time[10],label = "Epochs = 100")
axarr[1,1].plot(Datanumber, Time[11],label = "Epochs = 1000")
axarr[1,1].plot(Datanumber, TimeSVM,label = "QP Benchmark")
axarr[1,1].set_title('Lambda = 10')
axarr[1,1].set_xlabel('Dataset number')
#axarr[1,1].legend()

plt.savefig('4_3a(0.02).eps',format='eps',dpi=1000)

f, axarr = plt.subplots(2, 2)
Datanumber = [200,300,400,500]
#Datanumber, y01 = range(10), tB[1]
#Datanumber, y10 = range(10), tB[2]
#Datanumber, y11 = range(10), tB[3]

axarr[0,0].plot(Datanumber, Accuracy[0],label = "Epochs = 10")
axarr[0,0].plot(Datanumber, Accuracy[1],label = "Epochs = 100")
axarr[0,0].plot(Datanumber, Accuracy[2])
axarr[0,0].plot(Datanumber, AccuracySVM)
#axarr[0,0].plot(Datanumber, Time[2],label = "Epochs = 1000")
#axarr[0,0].plot(Datanumber, TimeSVM,label = "SVM")
axarr[0,0].legend()
axarr[0,0].set_title('Lambda = 0.01')
axarr[0,0].tick_params(bottom = 'off')
axarr[0,0].get_xaxis().set_visible(False)
#axarr[0,0].set_ylabel('Classification Error')
axarr[0,0].set_xlabel('Lambda')

#axarr[0,1].plot(Datanumber, Time[3],label = "Epochs = 10")
#axarr[0,1].plot(Datanumber, Time[4],label = "Epochs = 100")
axarr[0,1].plot(Datanumber, Accuracy[3])
axarr[0,1].plot(Datanumber, Accuracy[4])
axarr[0,1].plot(Datanumber, Accuracy[5],label = "Epochs = 1000")
axarr[0,1].plot(Datanumber, AccuracySVM,label = "QP Benchmark")
axarr[0,1].set_title('Lambda = 0.1')
axarr[0,1].legend()
axarr[0,1].tick_params(bottom = 'off')
axarr[0,1].get_xaxis().set_visible(False)

axarr[1,0].plot(Datanumber, Accuracy[6],label = "Epochs = 10")
axarr[1,0].plot(Datanumber, Accuracy[7],label = "Epochs = 100")
axarr[1,0].plot(Datanumber, Accuracy[8],label = "Epochs = 1000")
axarr[1,0].plot(Datanumber, AccuracySVM,label = "QP Benchmark")
axarr[1,0].set_title('Lambda = 1')
#axarr[1,0].legend()
axarr[1,0].set_ylabel('Accuracy')
axarr[1,0].set_xlabel('Dataset number')

axarr[1,1].plot(Datanumber, Accuracy[9],label = "Epochs = 10")
axarr[1,1].plot(Datanumber, Accuracy[10],label = "Epochs = 100")
axarr[1,1].plot(Datanumber, Accuracy[11],label = "Epochs = 1000")
axarr[1,1].plot(Datanumber, AccuracySVM,label = "QP Benchmark")
axarr[1,1].set_title('Lambda = 10')
axarr[1,1].set_xlabel('Dataset number')
#axarr[1,1].legend()

plt.savefig('4_3b.eps',format='eps',dpi=1000)
