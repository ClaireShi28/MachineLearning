import LoadData
import sys

import KNN
import DecisionTree as DT
import NeuralNetwork as NN
import AdBoost as AB
import SVM
import Experiments
import random
import time

def Process_wine_quality():

    t0=time.time()
    X, y = LoadData.load_wine_quality_data()
    op=True

    Experiments.Models_Comparison(X,y,"Wine Quality")
    '''
    DT.DecisionTree(X, y, title="Wine Quality Decision Tree",optimize=op)
    AB.AdaBoost(X, y, title="Wine Quality AdaBoost",optimize=op)
    KNN.KNN(X, y, title="Wine Quality KNN",optimize=op)
    NN.NeuralNetwork(X, y, title="Wine Quality Neural Network",optimize=op)
    SVM.SVM(X, y, title="Wine Quality SVM",optimize=op)
    '''
    t1=time.time()
    print("total TIME")
    print(t1-t0)


def Process_air_quality():
    X, y = LoadData.load_ozone_data()

    Experiments.Models_Comparison(X, y, "Air Pollution")
    op=True
    '''
    DT.DecisionTree(X, y, title="Air Pollution Decision Tree",optimize=op)
    AB.AdaBoost(X, y, title="Air Pollution AdaBoost",optimize=op)
    KNN.KNN(X, y, title="Air Pollution KNN",optimize=op)
    NN.NeuralNetwork(X, y, title="Air Pollution Neural Network",optimize=op)
    SVM.SVM(X, y, title="Air Pollution SVM",optimize=op)
    '''
if __name__ == "__main__":
    Process_air_quality()
    Process_wine_quality()

