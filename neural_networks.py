#!/usr/bin/env python
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import datasets
np.set_printoptions(suppress=True)

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def randomizeTheta(hidden_layer):
    theta = {}
    for i in range(1, len(hidden_layer)):
        theta['W' + str(i)] = np.random.normal(0, 1, size=(hidden_layer[i],hidden_layer[i-1]+1))
    return theta

def forwardPropagate(hidden_layer, theta, X):
    params = {}
    m = X.shape[0]
    a = X.transpose()
    a = np.insert(a, 0, [1]*m, axis=0)
    params['a' + str(1)] = a
    
    for i in range(1, len(hidden_layer)-1):
        z = np.matmul(theta['W' + str(i)],a)
        a = sigmoid(z)
        a = np.insert(a, 0, [1]*m, axis=0)
        params['a' + str(i+1)] = a
        
    z = np.matmul(theta['W' + str(len(hidden_layer)-1)],a)
    a = sigmoid(z)
    params['a' + str(len(hidden_layer))] = a
    
    return params

def costFunction(hidden_layer, theta, params, X, y, reg):
    m = X.shape[0]
    pred = params['a' + str(len(hidden_layer))]
    S = 0
    
    J = np.multiply(-y,np.log(pred)) - np.multiply(1-y,np.log(1-pred))
    J = J.sum()/m
    
    for i in range(1, len(hidden_layer)):
        weight = theta['W' + str(i)][:,1:]
        S += np.square(weight).sum()
   
    J = J + (reg/(2*m))*S
    return J

def backPropagate(hidden_layer, params, theta, X, y, reg):
    grads = {}
    m = X.shape[0]
    pred = params['a' + str(len(hidden_layer))]
    
    for i in range(len(hidden_layer)-1, 0, -1):
        grads['D' + str(i)] = np.zeros((hidden_layer[i],hidden_layer[i-1]+1))
    
    for i in range(m):
        delta = pred[:,i] - y[:,i] 
        delta = delta[:, None]
        
        grads['delta' + str(len(hidden_layer))] = delta
        
        for j in range(len(hidden_layer), 1, -1):
            weight = theta['W' + str(j-1)].transpose()
            a = params['a' + str(j-1)]
            a_T = np.array([a[:,i]])

            grads['D' + str(j-1)] += np.matmul(delta, a_T)
            
            delta = np.multiply(np.multiply(np.matmul(weight, delta), a_T.transpose()), (1-a_T.transpose()))            
            delta = np.delete(delta, 0, 0)
            
            grads['delta' + str(j-1)] = delta
    
    for i in range(1, len(hidden_layer)):
        P = reg * theta['W' + str(i)]
        P[:,0] = 0
        grads['D' + str(i)] = (grads['D' + str(i)] + P)/m
    
    return grads    

def updateTheta(hidden_layer, theta, grads, learning_rate):
    theta_updated = {}
    for i in range(1, len(hidden_layer)):
        theta_updated['W' + str(i)] = theta['W' + str(i)] - learning_rate*grads['D' + str(i)]
    return theta_updated

def evaluate(pred, y):
    tp = 0.1
    tn = 0.1
    fp = 0.1
    fn = 0.1
    
    if y.shape[0] != 1:
        real = np.argmax(y, axis=0)
    else:
        real = y[0]
    
    predicted = np.argmax(pred, axis=0)
    m = pred.shape[1]
    
    for i in range(m):
        if (predicted[i] == real[i]):
            if (real[i] == 1):
                tp += 1
            else: 
                tn += 1
        else:
            if (real[i] == 0):
                fp += 1
            else: 
                fn += 1

    M = np.array([[tp, fn], [fp, tn]])
    accuracy = (M[0,0] + M[1,1])/ M.sum()
    precision = (M[0,0])/ (M[0,0] + M[1,0])
    recall = (M[0,0])/ (M[0,0] + M[0,1])
    f1_score = 2*(precision*recall)/(precision+recall)
    
    return (M, accuracy, precision, recall, f1_score)

def predict(hidden_layer, X_train, y_train, X_test, y_test, learning_rate, num_iters, reg):
    theta = randomizeTheta(hidden_layer)
    cost = []
    for i in range(num_iters):
        params_train = forwardPropagate(hidden_layer, theta, X_train)
        params_test = forwardPropagate(hidden_layer, theta, X_test)
        J_train = costFunction(hidden_layer, theta, params_train, X_train, y_train, reg)
        J_test = costFunction(hidden_layer, theta, params_test, X_test, y_test, reg)
        cost.append(J_test)
        
        grads = backPropagate(hidden_layer, params_train, theta, X_train, y_train, reg)
        theta = updateTheta(hidden_layer, theta, grads, learning_rate)
        print('Cost at iteration ' + str(i+1) + ' = ' + str(J_train) + '\n')
    return theta, cost

def normalize(X_train, X_test):
    min_val = np.amin(X_train, axis=0)
    max_val = np.amax(X_train, axis=0)
    X_test = (X_test - min_val)/(max_val - min_val)
    return X_test

def standardize(X_train, X_test):
    epsilon = 0.00001
    avg = np.mean(X_train, axis = 0) 
    var = np.var(X_train, axis=0)
    X_test = (X_test - avg)/np.sqrt(var + epsilon)
    return X_test

def cross_validation(df, col_name):
    size = []
    dataset = []
    
    label = [y for x, y in df.groupby(col_name, as_index=False)]
    level = len(label)
    
    for i in range(level):
        size.append(math.floor(len(label[i])/10))
    
    for i in range(10):  
        fold = []
        for j in range(level):
            label[j] = label[j].reset_index(drop=True)
            index = random.sample(range(0, len(label[j])), size[j])
            fold.append(label[j].iloc[index])
            label[j] = label[j].drop(index)
        dataset.append(pd.concat(fold))

    for i in range(len(dataset)):
        dataset[i] = dataset[i].reset_index(drop=True)
    
    return dataset


def execute(df, col_name, layer, alpha, num_iters, reg):
    dataset = cross_validation(df, col_name)
    ohe = OneHotEncoder()
    
    metrics = np.zeros((10,2))
    table = np.zeros((len(layer),2))
    
    for i in range(len(layer)):
        for k in range(10):
            #train set
            train = pd.concat(dataset[:k] + dataset[k+1:])
            X_train = train.iloc[:,0:-1].to_numpy(dtype='float64')
            X_train_norm = standardize(X_train, X_train)
    
            transformed = ohe.fit_transform(train[[col_name]])
            y_train = transformed.toarray().transpose()
    
            #test set
            test = dataset[k]
            X_test = test.iloc[:,0:-1].to_numpy(dtype='float64')
            X_test_norm = standardize(X_train, X_test)
    
            transformed = ohe.fit_transform(test[[col_name]])
            y_test = transformed.toarray().transpose()
    
            #implement
            theta, cost = predict(layer[i], X_train_norm, y_train, \
                                  X_test_norm, y_test, alpha, num_iters, reg)
            params = forwardPropagate(layer[i], theta, X_test_norm)
            pred = params['a' + str(len(layer[i]))]
            (M, accuracy, precision, recall, f1_score) = evaluate(pred, y_test)
            metrics[k] = [accuracy, f1_score]
    
        table[i] = metrics.mean(axis = 0)
    return table

def learning_curve(df, col_name, layer, alpha, num_iters, reg):
    # split and shuffle dataset
    train, test = train_test_split(df, test_size = 0.2, shuffle=True) 
    ohe = OneHotEncoder()
    
    #train set
    X_train = train.iloc[:,0:-1].to_numpy(dtype='float64')
    X_train_norm = standardize(X_train, X_train)
    
    transformed = ohe.fit_transform(train[[col_name]])
    y_train = transformed.toarray().transpose()
    
    #test set
    X_test = test.iloc[:,0:-1].to_numpy(dtype='float64')
    X_test_norm = standardize(X_train, X_test)
    
    transformed = ohe.fit_transform(test[[col_name]])
    y_test = transformed.toarray().transpose()
    
    #implement
    theta, cost = predict(layer[0], X_train_norm, y_train, \
                          X_test_norm, y_test, alpha, num_iters, reg)   
      
    #plot graph    
    x1 = np.arange(1, num_iters+1, 1)
    y1 = np.array(cost)
    fix, ax = plt.subplots()
    ax.errorbar(x1, y1)
    ax.set_xlabel('Number of instances')
    ax.set_ylabel('Cost J')
    plt.show()


def main():
    # #1. Digits Recognition Dataset
    # digits = datasets.load_digits(return_X_y = True)
    # digits_dataset_X = digits[0]
    # digits_dataset_y = np.array([digits[1]])
    # arr = np.concatenate((digits_dataset_X, digits_dataset_y.T), axis=1)
    # df = pd.DataFrame(arr)
    # df.columns = [*df.columns[:-1], 'Class']
    # col_name = 'Class'
    # # hidden_layer = [[64,4,10],[64,10,16,10],[64,4,2,4,10],[64,2,4,4,2,10],\
    # #                 [64,6,4,10],[64,16,10]]
    # hidden_layer = [[64,4,10],[64,4,2,10],[64,4,2,4,10],[64,2,4,4,2,10],[64,16,10]]
    # final_layer = [[64,16,10]]
    # alpha = 1.5
    # num_iters = 500
    # reg = 0.5

    
    ##Result: 
        # [[0.8446712  0.49788254]
        #  [0.50283447 0.16768299]
        #  [0.44217687 0.15656288]
        #  [0.2324263  0.04898598]
        #  [0.95011338 0.7995651 ]]

    
    #2. Titanic Dataset
    # df = pd.read_csv('titanic.csv')

    # ##move predict column to last
    # cols = df.columns.tolist()
    # cols = cols[1:] + cols[:1]
    # df = df[cols]

    # ##remove name column
    # df = df.drop(columns = ['Name'])

    # ##convert sex column to numerical
    # df['Sex'] = pd.factorize(df['Sex'])[0]
    
    # col_name = 'Survived'
    # hidden_layer = [[6,4,2],[6,4,2,2],[6,4,2,4,2],[6,2,4,4,2,2],[6,16,2]]
    # final_layer = [[6,2,4,4,2,2]]
    # alpha = 1.5
    # num_iters = 300
    # reg = 0.1
    
    # ##Result 
    
    # [[0.80995475 0.73203292]
    #  [0.80769231 0.72374728]
    #  [0.81561086 0.73018861]
    #  [0.66063348 0.20890895]
    #  [0.82126697 0.74764605]]
    
    # [[0.07613636 0.06037736]]
    
    # #3. Loan Dataset
    # df = pd.read_csv('loan.csv')

    # ##remove loan_id column
    # df = df.drop(columns = ['Loan_ID'])

    # ##convert categorical column to numerical
    # df['Gender'] = pd.factorize(df['Gender'])[0]
    # df['Married'] = pd.factorize(df['Married'])[0]
    # df['Dependents'] = pd.factorize(df['Dependents'])[0]
    # df['Education'] = pd.factorize(df['Education'])[0]
    # df['Self_Employed'] = pd.factorize(df['Self_Employed'])[0]
    # df['Property_Area'] = pd.factorize(df['Property_Area'])[0]
    # df['Loan_Status'] = pd.factorize(df['Loan_Status'])[0]
    
    # col_name = 'Loan_Status'
    # hidden_layer = [[11,4,2],[11,4,2,2],[11,4,2,4,2],[11,2,4,4,2,2],\
    #                 [11,6,4,2],[11,16,2]]
    # hidden_layer = [[11,4,2],[11,4,2,2],[11,4,2,4,2],[11,2,4,4,2,2],[11,16,2]]
    # final_layer = [[11,4,2,4,2]]
    # alpha = 1.5
    # num_iters = 500
    # reg = 0.25
    
    ##Result 
    
    # [0.78481013 0.85590188]
    #  [0.77637131 0.85071894]
    #  [0.78059072 0.85146632]
    #  [0.74472574 0.84259406]
    #  [0.79535865 0.86380508]
    
    # #4. Parkinsons Dataset
    # df = pd.read_csv('parkinsons.csv')
    # col_name = 'Diagnosis'
    # hidden_layer = [[22,4,2],[22,4,2,2],[22,4,2,4,2],[22,2,4,4,2,2],[22,16,2]]
    # final_layer = [[22,4,2]]
    # alpha = 0.1
    # num_iters = 1000
    # reg = 0.8
    
    # ##Result 
    
    # [[0.85      , 0.91263441],
    # [0.78333333, 0.87782258],
    # [0.77777778, 0.875     ],
    # [0.77777778, 0.875     ],
    # [0.86111111, 0.91559352]]
    
    # [0.08888889 0.09333333]
    
    # #5. Contraceptive Dataset
    # df = pd.read_csv('contraceptive.csv', names = ['wife_age', 'wife_edu', 'husband_edu', 
    #                                         'child','wife_religion', 'wife_work', 
    #                                         'husband_work', 'living', 'media', 'Method'])
    # col_name = 'Method'
    # # hidden_layer = [[9,4,3],[9,4,2,3],[9,4,2,4,3],[9,2,4,4,2,3],[9,6,4,3],[9,16,3]]
    # hidden_layer = [[9,4,3],[9,4,2,3],[9,4,2,4,3],[9,2,4,4,2,3],[9,16,3]]
    # final_layer = [[9,4,2,3]]
    # alpha = 3
    # num_iters = 300
    # reg = 0.8
    
    ##Result:
    
    # [[0.46721311, 0.16634948],
    #    [0.43374317, 0.11907814],
    #    [0.42418033, 0.09845515],
    #    [0.42486339, 0.00236967],
    #    [0.46653005, 0.1362345 ]]
    
    # table = execute(df, col_name, hidden_layer, alpha, num_iters, reg)
    # print(table)
    learning_curve(df, col_name, final_layer, alpha, num_iters, reg)
main()
