#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import operator
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import datasets
np.set_printoptions(suppress=True)


def normalize(train, test):
    '''Normalize data to range (0,1)'''
    max_val = train.iloc[:,:-1].max()
    min_val = train.iloc[:,:-1].min()
    test.iloc[:,:-1] = (test.iloc[:,:-1] - max_val)/(max_val - min_val)
  
    return test 

def distance(train, test):
    '''Find distances between train and test instances'''
    num_train = train.shape[0] # number of train data
    num_test = test.shape[0] # number of test data
    dist = np.zeros((num_test, num_train)) # create a zero matrix size(num_test, num_train)
    
    for i in range(num_test):
        # calculate distance using Manhattan metric
        dist[i, :] = np.sqrt(np.sum((train.iloc[:,:-1] - test.iloc[i,:-1])**2, axis = 1))
            
    return dist

def find_label(train, neighbors, k):
    '''Find the label of an instance based on its neighbors'''
    prediction = {}
    
    # iterate through the number of neighbors
    for i in range(k):
        index = neighbors[i] # get neighbors' indexes
        label = train.iloc[index,-1] # get neighbors' label
        if label in prediction:
            prediction[label] += 1
        else:
            prediction[label] = 1
            
    label = max(prediction, key = prediction.get) # find key with largest value
    return label

def predict_input(train, distance, k):
    '''Predict labels of all instances in the test set'''
    test_num = distance.shape[0]
    pred = np.empty(test_num, dtype = train.iloc[:,-1].dtype)
    
    for i in range(test_num):
        neighbors = np.argsort(distance[i,:]) # sort the neighbors by distance
        label = find_label(train, neighbors, k) # find the label of each instance
        pred[i] = label # save the label to a list
    return pred  

def evaluate(pred, y):
    tp = 0.1
    tn = 0.1
    fp = 0.1
    fn = 0.1
    
    m = pred.shape[0]
    
    for i in range(m):
        if (pred[i] == y.iloc[i,-1]):
            if (y.iloc[i,-1] == 1):
                tp += 1
            else: 
                tn += 1
        else:
            if (y.iloc[i,-1] == 0):
                fp += 1
            else: 
                fn += 1

    M = np.array([[tp, fn], [fp, tn]])
    accuracy = (M[0,0] + M[1,1])/ M.sum()
    precision = (M[0,0])/ (M[0,0] + M[1,0])
    recall = (M[0,0])/ (M[0,0] + M[0,1])
    f1_score = 2*(precision*recall)/(precision+recall)
    
    return (M, accuracy, precision, recall, f1_score)

def execute(train, test, k):
    '''Combine functions together'''
    train_norm = normalize(train, train)
    test_norm = normalize(train, test)
    
    dist = distance(train_norm, test_norm)
    pred = predict_input(train_norm, dist, k)   
    (M, accuracy, precision, recall, f1_score) = evaluate(pred, test_norm)
    return accuracy, f1_score


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

def implement(df, col_name):  
    dataset = cross_validation(df, col_name)
    
    accuracy_mean = []
    accuracy_sd = []
    f1_mean = []
    f1_sd = []
    
    # loop through k from 1 to 51
    for i in range(1, 53, 2):
        accuracy_arr = []
        f1_arr = []
        for k in range(10):
            train = pd.concat(dataset[:k] + dataset[k+1:])
            test = dataset[k]
            accuracy, f1_score = execute(train, test, i) # find accuracy using train data
            accuracy_arr.append(accuracy)
            f1_arr.append(f1_score)
            
        accuracy_mean.append(np.mean(accuracy_arr))
        accuracy_sd.append(np.std(accuracy_arr))
        f1_mean.append(np.mean(f1_arr))
        f1_sd.append(np.std(f1_arr))
        
    return accuracy_mean, accuracy_sd, f1_mean, f1_sd

def visualize_accuracy(accuracy_mean, accuracy_sd, f1_mean, f1_sd):
    x1 = np.arange(1, 53, 2)
    y1 = np.array(accuracy_mean)
    yerr1 = np.array(accuracy_sd)
    fix, ax = plt.subplots()
    ax.errorbar(x1, y1, yerr = yerr1, fmt = '-o')
    ax.set_xlabel('Value of k')
    ax.set_ylabel('Accuracy')
    plt.show()
    
    x2 = np.arange(1, 53, 2)
    y2 = np.array(f1_mean)
    yerr2 = np.array(f1_sd) 
    fix, ax = plt.subplots()
    ax.errorbar(x2, y2, yerr = yerr2, fmt = '-o')
    ax.set_xlabel('Value of k')
    ax.set_ylabel('F1 Score')
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
    
    
    # #2. Titanic Dataset
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
    
    
    # #4. Parkinsons Dataset
    # df = pd.read_csv('parkinsons.csv')
    # col_name = 'Diagnosis'
    
    #5. Contraceptive Dataset
    df = pd.read_csv('contraceptive.csv', names = ['wife_age', 'wife_edu', 'husband_edu', 
                                        'child','wife_religion', 'wife_work', 
                                        'husband_work', 'living', 'media', 'Method'])
    col_name = 'Method'

    accuracy_mean, accuracy_sd, f1_mean, f1_sd = implement(df, col_name)
    visualize_accuracy(accuracy_mean, accuracy_sd, f1_mean, f1_sd)
    print(accuracy_mean[0])
    print(accuracy_mean[10])
    print(accuracy_mean[20])
    print(f1_mean[0])
    print(f1_mean[10])
    print(f1_mean[20])

    
main()