# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 18:10:55 2022

@author: Simone
"""

# Import libraries that contains the implementations of the functions used in the rest of the program.
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
# import random
import pandas as pd

# Regression model
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingClassifier
from sklearn.model_selection  import GridSearchCV

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR

import warnings
warnings.filterwarnings('ignore')
# Modify the dataset: https://towardsdatascience.com/classification-of-unbalanced-datasets-8576e9e366af


def import_dataset_3():
    path = 'C:/Users/Simone/Desktop/MS Computer Science/Machine Learning - Primo Anno/Homework/Hmw 1/Hmw_1_code/train_set.tsv' # insert you path
    # https://www.youtube.com/watch?v=4SivdTLIwHc
    dataset = pd.read_csv(path, sep='\t', header=0)
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
    X_train = X[:-25]
    X_test = X[-25:]
    
    # Split the targets into training/testing sets
    y_train = t[:-25]
    y_test = t[-25:] 
    return X_train, X_test, y_train, y_test


# DEPRECATED
# def plot(X_test, y_pred, y_test):
#         # Plot outputs # is a feature
#         for i in range(1,34):
#                 plt.scatter(X_test.iloc[:,i:i+1], y_test,  color='black', linewidth=0.0001)
#                 plt.scatter(X_test.iloc[:,i:i+1], y_pred, color='green', linewidth=0.0001)
#                 plt.xticks(())
#                 plt.yticks(())
#         plt.ylabel('Y')
#         plt.xlabel('X')
#         plt.show()
        

def plot_predicted_vs_true(y_pred, y_test):
    ## Plot predicted vs true
    fig, ax = plt.subplots(nrows=1, ncols=2)
    from statsmodels.graphics.api import abline_plot
    ax[0].scatter(y_pred, y_test, color="black")
    abline_plot(intercept=0, slope=1, color="red", ax=ax[0])
    # ax[0].vlines(x=max_pred, ymin=max_true, ymax=max_true-max_error, color='red', linestyle='--', alpha=0.7, label="max error")
    ax[0].grid(True)
    ax[0].set(xlabel="Predicted", ylabel="True", title="Predicted vs True")
    ax[0].legend()
        
if __name__ == '__main__':  # Main Programm
    print('Run the program ...\n')


    # 1. Extract and format the dataset
    X, t = import_dataset_3()
    
    plot_distribution(t)
    
 
   

    # 3. Print the information of dataset
    print("Input shape: %s" %str(X.shape))
    print("Output shape: %s" %str(t.shape))
    print("Number of attributes/features: %d" %(X.shape[1]))
    
    # 3. Print inofs on dataset
    # sample_t = dict()
    # for x in t[0]:
    #      if x in sample_t.keys():
    #          sample_t[x] += 1
    #      else:
    #          sample_t[x]=1
         
    # print(sample_t,'\n')
     
    # 3. Split the dataset
    X_train, X_test, y_train, y_test = split_dataset(X, t)

    # 6. Chose the model

    model_type = "poly_svm"  # "linear_regression", "linear_svm", "poly_svm", "random_forest"

    all_model = ["linear_regression", "linear_svm", "poly_svm", "random_forest"]
    
    for x in ["poly_svm"]:
        print('\nModel :',x)
        if x == "linear_regression":
                # Create linear regression object
                model = linear_model.LinearRegression()
                # Train the model using the training sets
                model.fit(X_train, y_train)
            
        elif x == "linear_svm":
                # SVM regression
                model = SVR(kernel='linear', C=0.01 )
                # Train the model using the training sets
                model.fit(X_train, y_train)
            
        elif x == "poly_svm":
                # params = {'kernel': ['poly'], 'C': [0.01,0.1,1], 'degree': [1,2,3,4,5], 'gamma': ['scale', 'auto']
                #               }
                # model = SVR()
                # model = GridSearchCV(model, params, cv=3)
                # SVM polynomial regression
                model = SVR(kernel='poly', C=0.1, degree=2, gamma='scale') #c=1.5, degree=3
                # Train the model using the training sets
                model.fit(X_train, y_train)
              
        elif x == "random_forest":
                # random forest
                model = RandomForestRegressor()
                # Train the model using the training sets
                model.fit(X_train, y_train)
        
        #Test
        y_pred = model.predict(X_test)
        
        plot_predicted_vs_true(y_pred, y_test)
        
        
        # The mean squared error
        print("Mean squared error: %.2f"
                  % mean_squared_error(y_test, y_pred))
            
        # R2 regression score: 1 is perfect prediction
        print('Regression score: %.2f' % r2_score(y_test, y_pred))
            
        # R2 regression score: 1 is perfect prediction
        print('Regression MAE : %.2f' % mean_absolute_error(y_test, y_pred))
