# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 10:13:02 2022

@author: Simone

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
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
import random
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# Modify the dataset: https://towardsdatascience.com/classification-of-unbalanced-datasets-8576e9e366af


def import_dataset():
    # https://www.youtube.com/watch?v=4SivdTLIwHc
    dataset = pd.read_csv("train_set.tsv", sep='\t', header=0)
    X = dataset.drop(['num_collisions'], axis=1)
    y = dataset['num_collisions']
    X = X.drop(['min_CPA'], axis=1)
    # normalizing the dataset
    # normalization explaind https://towardsdatascience.com/preprocessing-with-sklearn-a-complete-and-comprehensive-guide-670cb98fcfb9
    X = preprocessing.normalize(X,norm ='max')
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
    X_train, X_test, y_train, y_test = train_test_split(X, T, test_size=0.333,
                                                    random_state=117)
    return X_train, X_test, y_train, y_test


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


def balance_dataset_undersampling(X,t, sampling_strategy):
    #random unsumpling - balance the dataset
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy='majority')
    X, t = rus.fit_resample(X, t)
    return X,t


def balance_dataset_oversampling(X,t, sampling_strategy = None, random_state = None):
    # original distirbution {3: 30, 0: 538, 1: 333, 2: 96, 4: 3} 
    # random unsumpling - balance the dataset
    from imblearn.over_sampling import RandomOverSampler
    if sampling_strategy != None:
        rus = RandomOverSampler(sampling_strategy=sampling_strategy) #1:300, 2:260, 3:250, 4:300
    if random_state != None:
        rus = RandomOverSampler(random_state=random_state)
    X, t = rus.fit_resample(X, t)
    return X,t


def balance_dataset_SMOTE(X_train, y_train):
    from imblearn.over_sampling import SMOTE
    over_sampler = SMOTE(sampling_strategy='minority', random_state=7)
    X, t = over_sampler.fit_resample(X_train, y_train)
    return X,t


def balance_dataset_ADASYN(X,t, sampling_strategy = None, random_state = None):
    # How to banance, with focus on ADAYSN https://medium.com/dataman-in-ai/sampling-techniques-for-extremely-imbalanced-data-part-ii-over-sampling-d61b43bc4879
    from imblearn.over_sampling import ADASYN 
    if sampling_strategy != None:
        ada = ADASYN(sampling_strategy=sampling_strategy)
    if random_state != None:
        ada = ADASYN(random_state=random_state)
    X, t = ada.fit_resample(X, t)
    return X,t

    
if __name__ == '__main__':  # Main Programm
    print('Run the program ...\n')
    print('Paramenter: ')
    sampling_strategy = {2:250, 3: 200, 4: 150} #2:250, 3: 200, 4: 150
    # sampling_strategy = {2: 150, 3:125, 4:50} 
    print('Sampling strategy: ',sampling_strategy)
    random_state = 4
    print('random_state: ', random_state)

    # 1. Read the dataset and Normalize
    X, t = import_dataset()
    
    # a resampled copy
    X_resampled, t_resampled = balance_dataset_oversampling(X, t, sampling_strategy=sampling_strategy)
    # 2. list of traget calsses
    class_names = np.array([str(c) for c in range(0,5)])

    # plot_distribution(t)
    
    # 3. Print the information of dataset
    print("Input shape: %s" %str(X.shape))
    print("Output shape: %s" %str(t.shape))
    print("Number of attributes/features: %d" %(X.shape[1]))
    print("Number of classes: %d %s" %(len(class_names), str(class_names)))
    print("Number of samples: %d" %(X.shape[0]))
    
    # 4. Print the distirbution of output
    sample_t = dict()
    for x in t:
        if x in sample_t.keys():
            sample_t[x] += 1
        else:
            sample_t[x]=1
        
    print(sample_t,'\n')
    
    # 4. Split the dataset into train and test set
    X_train, X_test, y_train, y_test = split_dataset(X_resampled,t_resampled)
    
    # 5. balance first of all minority with Oversampling
    # X_train_resampled, y1_train_resampled = balance_dataset_oversampling(X_train, y_train, sampling_strategy=sampling_strategy)
    # X_train_resampled, y_train_resampled = balance_dataset_undersampling(X_train_resampled, y_train_resampled,sampling_strategy='auto')
    # plot_distribution(t_resampled)
    
    # 6. Balance the dataset with ADASYN
    # X_train_resampled, y_train_resampled = balance_dataset_ADASYN(X_train_resampled, y_train_resampled, sampling_strategy)
    # plot_distribution(y_train_resampled)

    # 7. Choose the model
    # model = RandomForestClassifier()
    #  model = tree.DecisionTreeClassifier() 
    #  model = BernoulliNB() # accuracy 0.51
    # model = GaussianNB() # accuracy 0.48
    # model = LogisticRegression()#accuracy 0.53
    #  SVM method
    params = {
                'C': [0.1, 0.001, 1.0],
                'gamma': [0.1, 1.0, 10],
                'kernel': ['rbf', 'poly', 'sigmoid', 'linear']
              }
    model_svm = svm.SVC()
    # Grid Search
    model_method = model_svm
    model = GridSearchCV(model_method, params, cv=5) # accuracy 0.55 best model
    # model = Perceptron()

    
    # 8. Train
    model.fit(X_train, y_train)

    # 9. Test
    y_pred = model.predict(X_test)

    # # print evaluation matrics
    print(classification_report(y_test, y_pred, labels=None, target_names= class_names, digits=2))
    warnings.filterwarnings('ignore')


    # 10. # print confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
    print(cm)
    plt.rcParams["figure.figsize"] = (10,10)
    plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=False)
    

    # 11. K-Fold Cross Validation
    print('\nK-Fold validation: ')
    cv = ShuffleSplit(n_splits=5, test_size=0.333, random_state=7)
    scores = cross_val_score(model, X, t, cv=cv)
    print(scores)
    print("Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
