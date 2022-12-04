# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 19:00:21 2022
@author: Simone
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
from sklearn.preprocessing import StandardScaler
import random
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# Modify the dataset: https://towardsdatascience.com/classification-of-unbalanced-datasets-8576e9e366af


def import_dataset_3():
    # https://www.youtube.com/watch?v=4SivdTLIwHc
    dataset = pd.read_csv("train_set.tsv", sep='\t', header=0)
    X = dataset.drop(['num_collisions'], axis=1)
    y = dataset['num_collisions']
    X = X.drop(['min_CPA'], axis=1)
    # normalizing the dataset
    X = preprocessing.normalize(X)
    return X,y


def plot_distribution(outcomes):
    plt.hist(outcomes, bins=25, color='b')
    plt.show()


def split_dataset(X,T):
    test_size = 0.333
    random_state = 7
    print('\ntest size',test_size)
    print('random state', random_state)
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
    X_train, X_test, y_train, y_test = train_test_split(X, T, test_size=test_size,
                                                    random_state=random_state)
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


def balance_dataset_undersampling(X,t):
    #random unsumpling - balance the dataset
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy='not minority')
    X, t = rus.fit_resample(X, t)
    return X,t


def balance_dataset_Oversampling(X,t):
    simple_strategy = {2: 190, 3: 150, 4: 100}
    print('\nSimple strategy: ',simple_strategy)
    
    #random unsumpling - balance the dataset
    from imblearn.over_sampling import RandomOverSampler
    rus = RandomOverSampler(sampling_strategy=simple_strategy) #{1:450, 2:380, 3:370, 4:320}) 
    X, t = rus.fit_resample(X, t)
    return X,t


def balance_dataset_SMOTE(X,t):
    from imblearn.over_sampling import SMOTE
    over_sampler = SMOTE(k_neighbors=1)
    X, t = over_sampler.fit_resample(X, t)
    return X,t


if __name__ == '__main__':  # Main Programm
    print('Run the program ...\n')


    # 1. Format the dataset
    X, t = import_dataset_3()
    # X,t = balance_dataset_Oversampling(X, t)

    # 2. list of class
    class_names = np.array([str(c) for c in range(0,5)])


    # 3. Adjust the dataset
    X_train, X_test, y_train, y_test = split_dataset(X,t)

    # 4. plot y distrribution
    # plot_distribution(y_train)
    # plot_distribution(y_test)

    # 5. Balance the dataset
    # 5.1 Balance the dataset with oversampling
    X_train, y_train = balance_dataset_Oversampling(X_train, y_train)
    # X_test, y_test = balance_dataset_Oversampling(X_test, y_test)

    # 5.2 Balance the dataset with undersampling
    # X_train, y_train = balance_dataset_undersampling(X_train, y_train)
    # X_test, y_test = balance_dataset_undersampling(X_test, y_test)

    # 5.3 Balance the dataset with SMOTE
    # X_train, y_train = balance_dataset_SMOTE(X_train, y_train)

    # 6. plot y distrribution
    plot_distribution(y_train)
    # plot_distribution(y_test)



    # model = BernoulliNB() # accuracy 0.51
    # model = GaussianNB() # accuracy 0.48
    # model = LogisticRegression() #accuracy 0.53
    # SVM method
    params = {
                'C': [0.1, 0.001, 1.0],
                'gamma': [0.1, 1.0, 10],
                'kernel': ['rbf', 'poly', 'sigmoid', 'linear']
              }
    model_svm = svm.SVC()
    # Grid Search
    model_method = model_svm
    model = GridSearchCV(model_method, params, cv=3) # accuracy 0.55 best model
    #model = SimplePerceptron()


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
    cv = ShuffleSplit(n_splits=5, test_size=0.333, random_state=15)
    scores = cross_val_score(model, X, t, cv=cv)
    print(scores)
    print("Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
