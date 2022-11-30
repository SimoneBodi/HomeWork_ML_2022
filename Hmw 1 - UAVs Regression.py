# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 18:10:55 2022

@author: Simone
"""

# Import libraries that contains the implementations of the functions used in the rest of the program.
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.utils.multiclass import unique_labels
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# import random
import pandas as pd

# Regression Warnings
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

import warnings
warnings.filterwarnings('ignore')
# Modify the dataset: https://towardsdatascience.com/classification-of-unbalanced-datasets-8576e9e366af


def import_dataset_3():
    # https://www.youtube.com/watch?v=4SivdTLIwHc
    dataset = pd.read_csv("train_set.tsv", sep='\t', header=0)
    X = dataset.drop(['min_CPA'], axis=1)
    X = X.drop(['num_collisions'], axis=1)
    y = dataset['min_CPA']
    
    # normalizing the dataset
    x = X.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    x_scaled = (x_scaled*2)-1
    X = pd.DataFrame(x_scaled)
    
    y = y.values.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    y_scaled = min_max_scaler.fit_transform(y)
    y_scaled = (y_scaled*2)-1
    y = pd.DataFrame(y_scaled)
    return X,y


def plot_distribution(outcomes):
    plt.hist(outcomes, bins=25, color='b')
    plt.show()


def split_dataset(X,T):
    # This function split the dataset in train set e test set

    # note for testing
    # svm
    # test_size 0.333
    # random state 117
    # accuracy 0.55

    # test_size 0.350
    # random state 110
    # accuracy 0.56

    # Split the data
    X_train = X[:-20]
    X_test = X[-20:]
    
    # Split the targets into training/testing sets
    y_train = t[:-20]
    y_test = t[-20:] 
    return X_train, X_test, y_train, y_test



if __name__ == '__main__':  # Main Programm
    print('Run the program ...\n')


    # 1. Extract and format the dataset
    X, t = import_dataset_3()
    

    # 2. list of class
    class_names = np.array([str(c) for c in range(0,5)])

    # 3. Split the dataset
    X_train, X_test, y_train, y_test = split_dataset(X, t)

    # 6. Chose the model

    model_type = "linear_svm"  # "linear_regression", "linear_svm", "poly_svm"

    if model_type == "linear_regression":
      # Create linear regression object
      model = linear_model.LinearRegression()
      # Train the model using the training sets
      model.fit(X_train, y_train)
    
    elif model_type == "linear_svm":
      # SVM regression
      model = SVR(kernel='linear', C=1.5)
      # Train the model using the training sets
      model.fit(X_train, y_train)
    
    elif model_type == "poly_svm":
      # SVM polynomial regression
      model = SVR(kernel='poly', C=1.5, degree=3, gamma='scale')
      # Train the model using the training sets
      model.fit(X_train, y_train)
      
    #Train
    model.fit(X_train, y_train)

    #Test
    y_pred = model.predict(X_test)
    
    # Plot outputs
    for i in range(1,34):
        plt.scatter(X_test.iloc[:,i:i+1], y_test,  color='black')
        plt.scatter(X_test.iloc[:,i:i+1], y_pred, color='red', linewidth=3)
        plt.xticks(())
        plt.yticks(())
        plt.show()

    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))
    
    # R2 regression score: 1 is perfect prediction
    print('Regression score: %.2f' % r2_score(y_test, y_pred))