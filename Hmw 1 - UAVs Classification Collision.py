# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 10:13:02 2022

@author: Simone
"""

# Import libraries that contains the implementations of the functions used in the rest of the program.

import random
import numpy as np
from sklearn import svm
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
import sklearn.metrics 
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt


def import_dataset():
    # This function open the dataset to store in a dictonary
    
    # 1. Delcaration of varible
    X = []
    T = []
    # Column name of dataset
    column_dataset_X = ['UAV_1_track', 'UAV_1_x', 'UAV_1_y', 'UAV_1_vx', 'UAV_1_vy', 'UAV_1_target_x', 'UAV_1_target_y', 'UAV_2_track', 'UAV_2_x', 'UAV_2_y', 'UAV_2_vx', 'UAV_2_vy', 'UAV_2_target_x', 'UAV_2_target_y', 'UAV_3_track', 'UAV_3_x', 'UAV_3_y', 'UAV_3_vx', 'UAV_3_vy', 'UAV_3_target_x', 'UAV_3_target_y', 'UAV_4_track', 'UAV_4_x', 'UAV_4_y', 'UAV_4_vx', 'UAV_4_vy', 'UAV_4_target_x', 'UAV_4_target_y', 'UAV_5_track', 'UAV_5_x', 'UAV_5_y', 'UAV_5_vx', 'UAV_5_vy', 'UAV_5_target_x', 'UAV_5_target_y']
    # 2. Open the file
    with open("C:\\Users\\Simone\\Desktop\\MS Computer Science\\Machine Learning - Primo Anno\\Homework Simone\\Hmw 1\\train_set.tsv", "r") as file:
        # Read the dataset
        dataset = []
        for num_line, line in enumerate(file.readlines()):
            # Remove de first Line (only with columns name)
            if num_line != 0:
                # Transform line from string of number to list of number
                dataset.append(line.split())
        # All rows contain de feature from UAV_1_track to min_CPA
        for num_r, row in enumerate(dataset):
            # Create a list of feature for num_r-row
            List_of_feature = []
            for num_f, feature in enumerate(row):
                # Select the Output T element
                if num_f == len(column_dataset_X):
                    T.append(float(feature))
                elif num_f < len(column_dataset_X):
                    # Assign feature value to row_r and feature_f
                    List_of_feature.append(float(feature))
            normalized_array = np.array(List_of_feature)
            # normalized_array = preprocessing.normalize([np.array(List_of_feature)])
            X.append(normalized_array) #----------> NORMALIZING ARRAY?
        file.close()  # To change file access modes
    return X, np.array(T)

    
def split_dataset(X,T):
    # This function split the dataset in train set e test set
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, T, test_size=0.333, 
                                                    random_state=117)
    return X_train, X_test, y_train, y_test


class LeastSquare:
    
    def __init__(self):
        self.w = [0, 0, 0]

    def fit(self,X,t):
        n = len(X) # nr. of examples
        t2 = np.c_[t, 1-t] # t2 is T: 1-of-K encoding
        phi = np.c_[np.ones(n), X] # design matrix
        self.w = np.matmul(np.linalg.pinv(phi),t2) # Least square solution
        print("Least square solution: %s" %(str(self.w.transpose())))

    def predict(self,x):
        xn = np.array((1, x[0][0], x[0][1]))
        yn = np.matmul(self.w.transpose(),xn)
        if yn[0]>yn[1]:
            return 1
        else:
            return -1   
        

class SimplePerceptron:

    def __init__(self, eta=0.01, niter=100):
        self.eta = eta
        self.niter = niter
        self.w = np.zeros(3)
    
    def fit(self,X,t):
        print('Perceptron model - eta: %f, niter: %d' %(self.eta, self.niter))
        n = len(X)
        # initial solution
        self.w = np.random.random()*np.ones(3)
        # niter iterations
        for i in range (0,self.niter):
            # select an instance
            k = int(np.random.random()*n)
            xk = np.array([1,X[k][0],X[k][1]])
            if (t[k]==1):
                tk = 1
            else:
                tk = -1
            # output
            o = np.sign(np.dot(self.w,xk))  # thresholded
            # update weigths
            self.w = self.w + self.eta * (tk-o) * xk
        print("Perceptron solution: %s" %str(self.w.transpose()))

    def predict(self,x):
        xn = np.array((1, x[0][0], x[0][1]))
        yn = np.matmul(self.w.transpose(),xn)        
        return np.sign(yn)
    
    
class FisherDiscriminant:

    def __init__(self):
        self.w = [0, 0, 0]
        self.label = "Fisher Discriminant"

    def fit(self,X,t):
        n = len(X)  # num of examples
        # group the two subsets 
        # C1 = positive samples, C2 = negative samples
        C1 = np.ndarray((0,2))
        C2 = np.ndarray((0,2))
        for i in range(0,len(X)):
            if (t[i][0] == 1):
                C1 = np.vstack([C1, [X[i,0],X[i,1]]])
            else:
                C2 = np.vstack([C2, [X[i,0],X[i,1]]])			
        
        # compute means m1, m2
        m1 = np.mean(C1, axis=0)
        m2 = np.mean(C2, axis=0)
        
        # compute covariances S1, S2
        S1 = np.zeros((2,2))
        d = np.array(())
        for c in C1:
            d = np.subtract(c,m1).reshape(2,1)
            dt = d.transpose()
            S1 = S1 + np.matmul(d,dt)
        
        S1 = S1/len(C1);
        
        S2 = np.zeros((2,2))
        for c in C2:
            d = np.subtract(c,m2).reshape(2,1)
            dt = d.transpose()
            S2 = S2 + np.matmul(d,dt)
        S2 = S2/len(C2);
        
        # compute Sw matrix
        Sw = S1+S2
        
        # compute solution w 
        wt = np.matmul(np.linalg.inv(Sw),(m1-m2))
        
        # global mean
        mu = m1 * 0.5 + m2 * 0.5
        
        # compute constant term
        w0 = np.dot(wt,mu)
        
        # format the final solution
        self.w = np.array([-w0, wt[0], wt[1]])
        print("Fisher discriminant solution: %s" %str(self.w.transpose()))

    
    def predict(self,x):
        xn = np.array((1, x[0][0], x[0][1]))
        yn = np.matmul(self.w.transpose(),xn)
        if yn>0:
            return 1
        else:
            return -1
       
   
def CodeProf():     
    # Param: n=size of data set, outliers=True/False
    def generateData(n, outliers=False):
        X = np.ndarray((n,2))
        t = np.ndarray((n,1))
        n1 = int(n*0.5)
    
        # define random centers of disctributions far away
        
        for i in range(0,n1):
            X[i,:] = np.random.normal((2.0,8.5),0.5,size=(1,2))
            t[i] = -1
        for i in range(n1,n):
            X[i,:] = np.random.normal((4.0,5.0),0.3,size=(1,2))
            t[i] = 1
        
        if (outliers):
            no=int(n*0.9)
            for i in range (no,n):
                X[i,:] = np.random.normal((9.0,3.0),0.2,size=(1,2))
                t[i] = 1
    
        return [X,t]
    
    n = 100
    outliers = True
    np.random.seed(123)
    X, t = generateData(n, outliers=outliers)
    return X, t
    
if __name__ == '__main__':
    print('Run the program ...\n')
    
    # Format the dataset
    X,t = import_dataset()
    

    # X,t = CodeProf()
    
    # Adjust the dataset
    X_train, X_test, y_train, y_test = split_dataset(X,t)
    
    # Chose the model
    #classifier_name = 'S'
    ClassifierMap = {
        'L': [LeastSquare, 'Least Square'], 
        'F': [FisherDiscriminant, 'Fisher Discriminant'], 
        'p': [SimplePerceptron, 'Simple Perceptron'], 
        'P': [Perceptron, 'Perceptron'], 
        'S': [svm.LinearSVC, 'SVM']
        }
    
    choosed_model = input('Chose the model beetween: \n Least Square: L; \n Fisher Discriminant: F; \n Simple Perceptron: p; \n Perceptron: P; \n SVM: S. \n --> ')
    
    # Train the model choosed
    classifier = ClassifierMap[choosed_model][0]()
    
    #  Set further parameters if using Simple Perceptron
    ETA = 0.001
    NITER = 1000
    if (choosed_model == 'p'):  # SimplePerceptron
        classifier.eta = ETA
        classifier.niter = NITER
        
    # Fit Classifier
    # train the classifier
    classifier.fit(X_train,y_train)
    
    # Prediction for Least Square
    y_pred = classifier.predict(x)
    
    # evaluate Accuracy
    acc = classifier.score(X_test, y_test)    
    print("\nAccuracy %.3f" %acc)
    
    # evluate Precision and Recall
    #print(classification_report(y_test, y_pred, labels=None, target_names=t, digits=3))
