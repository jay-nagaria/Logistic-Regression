# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:52:04 2018

@author: Ashok
"""
from sklearn import metrics
import random as rnd
#import data_handler_almost as DH
import numpy as np
import math
from math import exp
import csv
import matplotlib
trainingSet=[]
testSet=[]   
#dataset[]

'''----------------------------------------Data Handler to get and split the data-----------------------------'''
def splitData2TestTrain(filename,no_per_class,test_instances,trainingSet,testSet):
    
    for x in range(len(filename)):
        
        if x%no_per_class >= test_instances[0] and x%no_per_class <= test_instances[1]:
            testSet.append(filename[x])
        else:
            trainingSet.append(filename[x])
            

def pickDataClass(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        #print(type(lines))
        dataset = list(lines)
        #print(type(dataset1))
        #dataset=[list(i) for i in zip(*dataset1)]
        for x in range(len(dataset)):
            for y in range(0,len(dataset[0])-1):
                dataset[x][y] = float(dataset[x][y])
        #new_dataset=[]
        #for x in range(len(dataset)):
         #   if dataset[x][0] in class_ids:
         #      new_dataset.append(dataset[x])
    #print(new_dataset)    
    return(dataset)

#list_1=DH.letter_2_digit_convert('AC')
#dataset=DH.pickDataClass('G:\Programs\ATNTFaceImages400.txt',list_1)
dataset=pickDataClass('G:\Programs\myDataSet.txt')



splitData2TestTrain(dataset,4000,[1000, 2000],trainingSet,testSet)
#print(testSet)
#print(len(dataset))
'''--------------------------------------------initialize variable parameters-----------------------------------------'''

w=[]
for x in range(len(trainingSet[0])):
    w.append(0.2)
fnet=0    
bias=0.05
trainY=[float(trainingSet[i][2]) for i in range(len(trainingSet))]
#print(trainY)
#print(y)
#for x in range(len(trainingSet)):
out_all=[]
final_error=[]
error=0.0
epoch= 1000
#print(len(trainingSet))
#print(w[1])
out_list = []

'''--------------------------------------Logistic Regression........................................'''

for x in range(epoch):
    sum_error = 0
    for x in range(len(trainingSet)):
        for y in range(len(trainingSet[0])-1):
            fnet=fnet+float(w[y])*float(trainingSet[x][y])   
        fnet=fnet+bias
        #print(fnet)
        out= 1.0/(1.0+math.e**(-fnet))
        out_list.append(round(out))
        #out = 1/(1+math.e**(-fnet))
        #error = math.sqrt((trainY[x]-out)**2)
        error = trainY[x]-1-out # for simple error calc
        out_all.append(out)
        final_error.append(error)
        sum_error = sum_error + -(trainY[x]-1) *np.log(out) - (1 - (trainY[x]-1))* np.log(1-out) # for cross entropy
        #sum_error = sum_error +error**2
        lrn_rate = 0.05
        for y in range(len(w)):
            w[y] = w[y] + lrn_rate * float(trainingSet[x][y])* error * out * (1.0 - out)
        
        bias = bias + lrn_rate * error * out * (1.0 - out)
        fnet=0
        out=0
        error=0
        sum_error = 0
        
    sum_error = 0
    out=0
    error=0
    #print("---")

'''-------------------------------------------------TEST---------------------------------------------------'''
       
predicted_true=0
false_positive = 0
true_positive = 0
false_negative = 0
true_negative = 0

testY=[float(testSet[i][-1]) for i in range(len(testSet))]

out_all = []
for x in range(len(testSet)):
    for y in range(len(testSet[0])-1):
        fnet=fnet+float(w[y])*float(testSet[x][y])   
    fnet=fnet+bias
    #print(fnet)
    out= 1.0/(1.0+math.e**(-fnet))
    #out = 1/(1+math.e**(-fnet))
    #error = math.sqrt((trainY[x]-out)**2)
    error = float(testSet[x][-1])-out
    #print(fnet)
    #print(trainY[x])
    #print(out)
    #print(round(out))  
    out_all.append(round(out))
    if float(testSet[x][-1])-1 == 1:
        if round(out) == 1:
            true_positive = true_positive + 1
        else:
            false_negative = false_negative + 1
            
    if float(testSet[x][-1])-1 == 0:
        #print("ya...")
        if round(out) == 0:
            true_negative = true_negative + 1
        else:
            false_positive = false_positive + 1
            
    if round(out) == float(testSet[x][-1])-1:
        predicted_true = predicted_true + 1
        
        
    
        #print(error)
    fnet=0
    out=0
    error=0

'''-------------------------------------------------Ploting accuracy and ROC curve---------------------------------------------------'''
       
    
#print(true_positive)
#print(true_negative)
#print(false_positive)
#print(false_negative)
#print(predicted_true/len(testSet))
print((true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative))
#print(testSet[:][-1])
fpr, tpr, thresholds = metrics.roc_curve(testY, out_all, pos_label=2)

matplotlib.pyplot.plot(fpr, tpr, 'go-', label='line 1', linewidth=2) 

