import numpy as np
import dill as pickle
import sklearn
import matplotlib.pyplot as plt

from numpy import loadtxt

#from Link Alfeld Gave us

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
#SVC is for Support Vector Classifier -- we called it SVM in class
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import train_test_split

# Uncomment the following 3 lines if you're getting annoyed with warnings from sklearn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def runTests(X_train, X_test, y_train, y_test):
    zeroSums = np.zeros((13))
    count = 0
    #Knn
    neighborList = [1,3,5,7,9]
    for value in neighborList:
        kNeighbors = KNeighborsClassifier(n_neighbors = value)
        kNeighbors.fit(X_train, y_train)
        predTest = kNeighbors.predict(X_test)

        zeroSums[count] = zero_one_loss(y_test, predTest)
        count +=1                                        
    #dTree
    depthList = [1,2,3,4,None]
    for depth in depthList:
        dTree = DecisionTreeClassifier(max_depth = depth)
        dTree.fit(X_train,y_train)
        predTest = dTree.predict(X_test)

        zeroSums[count] = zero_one_loss(y_test, predTest)
        count +=1
    #svms
    linSVM = SVC(kernel = 'linear')
    linSVM.fit(X_train, y_train)
    predTest = linSVM.predict(X_test)

    zeroSums[count] = zero_one_loss(y_test, predTest)
    count +=1
    
    rbfSVM = SVC(kernel = 'rbf')
    rbfSVM.fit(X_train, y_train)
    predTest = rbfSVM.predict(X_test)

    zeroSums[count] = zero_one_loss(y_test, predTest)
    count +=1
    
    polySVM = SVC(kernel = 'poly', degree = 3)
    polySVM.fit(X_train, y_train)
    predTest = polySVM.predict(X_test)

    zeroSums[count] = zero_one_loss(y_test, predTest)
    count +=1
    
    #print(zeroSums)
    return zeroSums
    




if __name__ == "__main__":


    X_test = loadtxt("posts_test.txt", skiprows=1, delimiter=",", unpack=False)
    X = loadtxt("posts_train.txt", skiprows=1, delimiter=",", unpack=False)
    X_train = X[:, [0,1,2,3,6]]
    y_train = X[:, 4:5]
    



        
