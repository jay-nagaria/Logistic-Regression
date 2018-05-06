
import random as rnd
#import data_handler_almost as DH
import numpy as np
import math
from math import exp
import csv


'''----------------------------------------Data Handler to get and split the data-----------------------------'''


trainingSet=[]
testSet=[]   

def splitData2TestTrain(filename,no_per_class,test_instances,mySet):
    
    for x in range(len(filename)):
        
        if x%no_per_class >= test_instances[0] and x%no_per_class <= test_instances[1]:
            mySet.append(filename[x])
            
            
def pickDataClass(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        
        dataset = list(lines)
        
        for x in range(len(dataset)):
            for y in range(0,len(dataset[0])-1):
                dataset[x][y] = float(dataset[x][y])
       
    return(dataset)


trainingSet1=pickDataClass('G:\Programs\mnist_train_data.csv')
testSet1=pickDataClass('G:\Programs\mnist_test_data.csv')
#print(len(trainingSet1))

splitData2TestTrain(trainingSet1,6000,[0,999],trainingSet)
splitData2TestTrain(testSet1,980,[0,499],testSet)

#print(trainingSet)

'''-----------------------------------------Weights and bias for each class--------------------------------------'''
w = []
bias = [0.2,0.2,0.2,0.2,0.2]
list_1=[0.2 for y in range(len(trainingSet[0])-1)]
w.append(list_1)
w.append(list_1)
w.append(list_1)
w.append(list_1)
w.append(list_1)
#bias.append([0.2,0.2,0.2,0.2,0.2])
w = np.array(w)
w = w.astype(float)
bias = np.array(bias)
bias = bias.astype(float)
#print(w)  

'''-----------------------------------Split training and testing set into label(Y) and features(X)-------------------'''
trainingSetY = []
trainingSetX = []


for x in range(len(trainingSet)):
    #print(trainingSet[x][-1])
    trainingSetY.append(trainingSet[x][-1])
    trainingSetX.append(trainingSet[x][:-1])
    
testSetY = []
testSetX = []


for x in range(len(testSet)):
  
    testSetY.append(testSet[x][-1])
    testSetX.append(testSet[x][:-1])
    
#print(len(trainingSetX[0]))

#print(trainingSetX)
trainingSetX = np.array(trainingSetX)
trainingSetY = np.array(trainingSetY)
trainingSetY = trainingSetY.astype(float)
trainingSetX = trainingSetX.astype(float)

testSetX = np.array(testSetX)
testSetY = np.array(testSetY)
testSetY = testSetY.astype(float)
testSetX = testSetX.astype(float)

'''---------------------------------------------------SoftMax for each class---------------------------------''' 


trainingSetY_new = []
for x in range(len(trainingSetY)):
    if trainingSetY[x] == 0:
       
        trainingSetY_new.append([1,0,0,0,0])
        
    elif trainingSetY[x] == 1:
       
        trainingSetY_new.append([0,1,0,0,0])
    
    elif trainingSetY[x] == 2:
        
        trainingSetY_new.append([0,0,1,0,0])
        
    elif trainingSetY[x] == 3:
       
        trainingSetY_new.append([0,0,0,1,0])
    
    elif trainingSetY[x] == 4:
        
        trainingSetY_new.append([0,0,0,0,1])

trainingSetY_new = np.array(trainingSetY_new)
trainingSetY_new = trainingSetY_new.astype(float)  


for x in range(len(testSetX)):
    max_val = max(testSetX[x])
    for y in range(len(testSetX[0])):
        testSetX[x][y]= testSetX[x][y]/max_val


print(bias)
print(len(bias))
print(len(trainingSetY_new[0]))

'''--------------------------------------------initialize variables-----------------------------------------'''

fnet=0
out=0
error=0
sum_error = 0
lrn_rate = 0.01


for x in range(len(trainingSetX)):
    max_val = max(trainingSetX[x])
    for y in range(len(trainingSetX[0])):
        trainingSetX[x][y]= trainingSetX[x][y]/max_val
        
'''--------------------------------------Logistic Regression........................................'''
epoch = 50

for ep in range(epoch):
    for cls_id in range(5):
        for x in range(len(trainingSetX)):
            for y in range(len(trainingSetX[0])):
              
                fnet=fnet+float(w[cls_id][y])*float(trainingSetX[x][y])
            
            fnet = fnet + bias[cls_id]
            out = 1/(1+exp(-fnet))
            #print(out)
            '''
            if fnet > 0:
                out = fnet
            else:
                out = 0
            '''
            error = trainingSetY_new[x][cls_id]- round(out,3)
            #error = -trainingSetY_new[x][cls_id] * np.log(round(out,3)) - (1 - trainingSetY_new[x][cls_id]) * np.log(1 - round(out,3))
            #print(error)
            sum_error = sum_error +error**2
            
            for y in range(len(trainingSetX[0])):
                w[cls_id][y] = w[cls_id][y] + lrn_rate * float(trainingSetX[x][y])* error 
            bias[cls_id] = w[cls_id][y] + lrn_rate * error 
            
            fnet=0
            out=0
            error=0
        sum_error = math.sqrt(sum_error)
        #print(sum_error)
                
        sum_error = 0
    print('---')
    
    
    '''-------------------------------------------------TEST and accuracy for each epoch---------------------------------------------------'''
 

    true=0
    max_val = 0
    max_cls = 0
    for x in range(len(testSet)):
        for cls_id in range(5):
            for y in range(len(testSetX[0])):
                  
                fnet=fnet+float(w[cls_id][y])*float(testSetX[x][y])
            
            fnet = fnet + bias[cls_id]
            out = 1/(1+exp(-fnet))
            
            if out > max_val:
                max_val = out
                max_cls = cls_id
            
            fnet = 0
            out = 0
        
        #print(max_cls)
        
        if max_cls == testSetY[x]:
            true = true + 1
        
        max_cls = 0
        max_val = 0
        
    
    print(true/len(testSetY))
        
 
        
 #'''-------------------------------------------------TEST---------------------------------------------------'''
       


