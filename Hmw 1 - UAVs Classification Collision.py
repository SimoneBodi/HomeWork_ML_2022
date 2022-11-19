# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 10:13:02 2022

@author: Simone

#In this program we use GaussianNaiveBayes
"""

# Import libraries that contains the implementations of the functions used in the rest of the program.

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import *
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV

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
            #normalized_array = np.array(List_of_feature)
            normalized_array = preprocessing.normalize([np.array(List_of_feature)])
            X.append(normalized_array) #----------> NORMALIZING ARRAY?
        file.close()  # To change file access modes
    return X, np.array(T)


def split_dataset(X,T):
    # This function split the dataset in train set e test set
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, T, test_size=0.333, 
                                                    random_state=117)
    return X_train, X_test, y_train, y_test
    # svm
    # test_size 0.333
    # random state 117
    # accuracy 0.55
    
    # test_size 0.350
    # random state 110
    # accuracy 0.56

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



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

    
    
if __name__ == '__main__':  # Main Programm
    print('Run the program ...\n')
    
    
    # Format the dataset
    X,t = import_dataset()
    
    # list of class
    class_names = np.array([str(c) for c in set(t)])
    # X,t = CodeProf()
    
    # Adjust the dataset
    X_train, X_test, y_train, y_test = split_dataset(X,t)
    
    # resahpe all dataset
    dimX1, dimX2, dimX3 = np.array(X_train).shape
    X_train = np.reshape(np.array(X_train), (dimX1*dimX2, dimX3))
    
    dimX1, dimX2, dimX3 = np.array(X_test).shape
    X_test = np.reshape(np.array(X_test), (dimX1*dimX2, dimX3))
    
    dimY2 = y_train.shape
    y_train = np.reshape(np.array(y_train), dimY2)
    
    # model = Perceptron() # accuracy 0.53
    # eta = 0.001
    # niter = 1000
    # model.eta = eta
    # model.niter = niter    
    # model = BernoulliNB() # accuracy 0.51
    # model = GaussianNB() # accuracy 0.48
    # model = LogisticRegression() #accuracy 0.53
    model = svm.SVC(kernel='linear', C=4) # accuracy 0.55 best model
    
    

    #Train
    model.fit(X_train, y_train)
    
    #Test
    y_pred = model.predict(X_test)
    
    # print evaluation matrics
    print(classification_report(y_test, y_pred, labels=None, target_names= class_names, digits=2))
    
    # print confusion matrix    
    cm = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
    print(cm)
    plt.rcParams["figure.figsize"] = (10,10)
    plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=False)
    
    #K-Fold Cross Validation 
    # cv = ShuffleSplit(n_splits=5, test_size=0.333, random_state=15)
    # scores = cross_val_score(model, X, t, cv=cv)
    # print(scores)
    # print("Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
   
