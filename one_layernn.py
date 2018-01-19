#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:36:33 2017

@author: snigdha
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import timeit


def softmax(x):
    e_x = np.exp(x) + 0.01
    e_x /= np.sum(e_x)
    return e_x

def sigmoid(x):
    e_x = np.exp(-x)
    return (1 / (1 + e_x))

def tanh(x):
    return np.tanh(x)

#Reference of relu function : Code review website
def relu(x, derivative=False):
    if (derivative == True):
        for i in range(0, len(x)):
            for k in range(len(x[i])):
                if x[i][k] > 0:
                    x[i][k] = 1
                else:
                    x[i][k] = 0
        return x
    
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            if x[i][k] > 0:
                pass 
            else:
                x[i][k] = 0
    return x

# Loading the given train, validation and test data
string1=[]
string2=[]
string3 = []

f1 = open("digitstrain.txt", 'r')
for line1 in f1:
     string1.append(line1.strip().split(','))
     
f2 = open("digitsvalid.txt", 'r')
for line2 in f2:
     string2.append(line2.strip().split(','))
     
f3 = open("digitstest.txt", 'r')
for line3 in f3:
     string3.append(line3.strip().split(','))

x_train1, y_train1 = [], []
for subl1 in string1:
    x_train1.append(subl1[:-1])
    y_train1.append([subl1[-1]])

x_valid, y_valid = [], []
for subl2 in string2:
    x_valid.append(subl2[:-1])
    y_valid.append([subl2[-1]])
    
x_test, y_test = [], []
for subl in string3:
    x_test.append(subl[:-1])
    y_test.append([subl[-1]])
   
x_train1 = np.float64(x_train1)
y_train1 = np.array(y_train1)
x_valid = np.float64(x_valid)
y_valid = np.array(y_valid)
x_test = np.float64(x_test)
y_test = np.array(y_test)


#shuffling the data randomly in order to 
#train the model better
x_train,y_train = shuffle(x_train1,y_train1, random_state=5)

#Converting the label data of training, validation and testing
#to one haute encoding

y_encod = np.zeros((3000,10))
for p in range(0,len(y_train)):
    k = np.int(y_train[p,0])
    y_encod[p,k] = 1

p=0
k=0
y_encodv = np.zeros((1000,10))
for p in range(0,len(y_valid)):
    k = np.int(y_valid[p,0])
    y_encodv[p,k] = 1
    
p=0
k=0
y_encodt = np.zeros((3000,10))
for p in range(0,len(y_test)):
    k = np.int(y_test[p,0])
    y_encodt[p,k] = 1

#Initialization of hidden units and number of random 
#initialization seeds
ini = 5
numhid = 100

soft_op_train = np.zeros((3000,10))
soft_op_valid = np.zeros((1000,10))
soft_op_test = np.zeros((3000,10))

cross_entr = np.zeros(3000)
cross_entr_valid = np.zeros(1000)
cross_entr_test = np.zeros(3000)

avg_cr_train = np.zeros((200,ini))
avg_cr_valid = np.zeros((200,ini))
avg_cr_test = np.zeros((200,ini))

avg_cl_train = np.zeros((200,ini))
avg_cl_valid = np.zeros((200,ini))
avg_cl_test = np.zeros((200,ini))


avg_class_train = np.zeros(200)
avg_cren_train = np.zeros(200)


avg_class_valid = np.zeros(200)
avg_cren_valid = np.zeros(200)

avg_cren_test = np.zeros(200)
avg_class_test = np.zeros(200)

start = timeit.default_timer()
for iter1 in range(0,ini):
    
#Initializing the weights and biases    
    w1 = np.random.normal(0, 0.1,(784,numhid))
    b1= 0
    w2 = np.random.normal(0, 0.1,(numhid,10))
    b2= 0
    learn_rate = 0.1
    dw1 = 0
    dw2 = 0 
    db1 = 0
    db2 = 0
    momentum = 0.5
    for iter2 in range(0,200):
        for i in range(0,3000):
            
            #Forward pass
            x_input = x_train[i,:]
            xinr = x_input.reshape((784,1))
    
            a1 = np.matmul(np.transpose(xinr),w1) + b1
            

            h1= sigmoid(a1) 
            h1r = h1.reshape((numhid,1))
    
            a2 = np.matmul(h1,w2) + b2
        
        
            h2 =softmax(a2)
            h2r = h2.reshape((10,1))
            soft_op_train[i,:] = np.transpose(h2r)      
    
            y_encodr = y_encod[i,:].reshape((10,1))
            
            #Back pass
            
            deltaw2 = np.transpose(np.matmul((h2r - y_encodr),np.transpose(h1r)))
            deltab2 = np.transpose(h2r - y_encodr)
            
                        
            temp1 = np.multiply(h1r,(1-h1r))
            temp2 = np.matmul(w2,(h2r-y_encodr))
            temp3 = np.multiply(temp1,temp2)
            deltaw1 = np.transpose(np.matmul(temp3,np.transpose(xinr)))
            deltab1 = np.transpose(temp3)
            
            w1 = w1 - learn_rate * (deltaw1 + momentum * dw1) - learn_rate * 0.001 * w1
            b1 = b1 - learn_rate * (deltab1 + momentum * db1) 
            
            w2 = w2 - learn_rate * (deltaw2 + momentum * dw2) - learn_rate * 0.001 * w2
            b2 = b2 - learn_rate * (deltab2 + momentum * db2)
            
            dw2 = deltaw2
            db2 = deltab2
            dw1 = deltaw1
            db1 = deltab1
    
#To determine cross entropy error of training set        
        for l in range(0,3000):
            cross_entr[l] = (-1)*np.sum(np.multiply(y_encod[l,:],np.transpose(np.log(soft_op_train[l,:]))))
    
        avg_cr_train[iter2,iter1] = np.sum(cross_entr)/3000
#To determine classification error of training set              
        number1 = np.zeros(3000)
        index1 = np.zeros(3000)
        index1 = np.argmax(soft_op_train,axis=1)
        for m1 in range(0,3000):
            if index1[m1] == np.int(y_train[m1,0]):
                number1[m1] =  0
            else:
                number1[m1] =  1
                      
        avg_cl_train[iter2,iter1] = (np.float(np.sum(number1))/3000)
        

    #Forward pass for validation

        for j in range(0,1000):
            x_input_v = x_valid[j,:]
            xinr_v = x_input_v.reshape((784,1))
    
            a1_v = np.matmul(np.transpose(xinr_v),w1) + b1

            h1_v = sigmoid(a1_v)   
            h1rv = h1_v.reshape((numhid,1))
    
            a2_v = np.matmul(h1_v,w2) + b2
    
        
            h2_v =softmax(a2_v)
            h2rv = h2_v.reshape((10,1))
            soft_op_valid[j,:] = np.transpose(h2rv)      
    
#To determine classification error of validation set      
        number2 = np.zeros(1000)
        index2 = np.zeros(1000)
        index2 = np.argmax(soft_op_valid,axis=1)
        for m2 in range(0,1000):
            if index2[m2] == np.int(y_valid[m2,0]):
                number2[m2] = 0
            else:
                number2[m2] = 1
                      
        avg_cl_valid[iter2,iter1] = (np.float(np.sum(number2))/1000 )
        
#To determine cross entropy error of validation set          
        for l in range(0,1000):
            cross_entr_valid[l] = (-1)*np.sum(np.multiply(y_encodv[l,:],np.transpose(np.log(soft_op_valid[l,:]))))
            
        avg_cr_valid[iter2,iter1] = np.sum(cross_entr_valid)/1000 
                      
         #Forward pass for testing

        for m in range(0,3000):
            x_input_t = x_test[m,:]
            xinr_t = x_input_t.reshape((784,1))
    
            a1_t = np.matmul(np.transpose(xinr_t),w1) + b1

            h1_t =sigmoid(a1_t)
            h1rt = h1_t.reshape((numhid,1))
    
            a2_t = np.matmul(h1_t,w2) + b2
    
        
            h2_t =softmax(a2_t)
            h2rt = h2_t.reshape((10,1))
            soft_op_test[m,:] = np.transpose(h2rt)      
    
#To determine classification error of test set    
        number3 = np.zeros(3000)
        index3 = np.zeros(3000)
        index3 = np.argmax(soft_op_test,axis=1)
        for m3 in range(0,3000):
            if index3[m3] == np.int(y_test[m3,0]):
                number3[m3] = 0
            else:
                number3[m3] = 1
                      
        avg_cl_test[iter2,iter1] = (np.float(np.sum(number3))/3000 )
        
#To determine cross entropy error of test set        
        for l in range(0,3000):
            cross_entr_test[l] = (-1)*np.sum(np.multiply(y_encodt[l,:],np.transpose(np.log(soft_op_test[l,:]))))
            
        avg_cr_test[iter2,iter1] = np.sum(cross_entr_test)/1000 

#averaging errors obtained over all initializations
avg_cren_train = np.mean(avg_cr_train, axis=1)  
avg_cren_valid = np.mean(avg_cr_valid,axis=1) 
avg_cren_test = np.mean(avg_cr_test,axis=1)    

avg_class_train = np.mean(avg_cl_train, axis=1)  
avg_class_valid = np.mean(avg_cl_valid,axis=1)  
avg_class_test = np.mean(avg_cl_test,axis=1) 
  
stop = timeit.default_timer()
print("Time taken",stop-start)
numar = np.arange(0,200,1)

plt.figure(1)                
plt.plot(numar, avg_cren_train, 'r', label='Avg Cross Entropy Training Error')
plt.plot(numar, avg_cren_valid, 'b', label='Avg Cross Entropy Validation Error')
plt.plot(numar, avg_cren_test, 'g', label='Avg Cross Entropy Test Error')
plt.title('Observation of average cross-entropy error of training,validation and test')
plt.xlabel('Number of epochs')
plt.ylabel('Prediction error')
plt.legend()
plt.show()

plt.figure(2)                
plt.plot(numar, avg_class_train, 'r', label='Avg Classification Training Error')
plt.plot(numar, avg_class_valid, 'b', label='Avg Classification Validation Error')
plt.plot(numar, (avg_class_test), 'g', label='Avg Classification Test Error')
plt.title('Observation of average classification error of training,validation and test')
plt.xlabel('Number of epochs')
plt.ylabel('Prediction error')
plt.legend()
plt.show()

#W visualized with MATLAB as montage



    
    
