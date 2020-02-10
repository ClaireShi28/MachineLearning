import numpy as np
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


import LoadData
import Experiments
import time
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

def NeuralNetwork(X,y, title,optimize):
    print("Neural Network Results")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
    clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(100,), random_state=0, activation='logistic',
                        max_iter=10000)
    cval_score = (cross_val_score(clf, X_train, y_train, cv=10)).mean()
    t0 = time.time()
    clf.fit(X_train, y_train)
    t1 = time.time()
    train_time=t1-t0
    print('Completed training in %f seconds' % train_time)
    ytest_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, ytest_pred)

    print("Mean cross validation score: %.2f%%" % (cval_score * 100))
    print("Accuracy score before tuning: %.2f%%" % (accuracy * 100))
    #print("Confusion matrix for neural network is:")
    #Experiments.ConfusionMatrix(y_test, ytest_pred)
    Experiments.plot_l_curve_website(clf, title, X, y, axes=None, ylim=None, cv=10)

    cv = 10

    if optimize==True:
    #GridSearchCV for NN
        param_grid = {
            'hidden_layer_sizes': [(5, 2), (50,), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive'],
        }
    

        best_params = Experiments.Tuning(clf, tuned_params=param_grid, cv=cv, Xtrain=X_train,
                                     Ytrain=y_train, Xtest=X_test, Ytest=y_test)

        best_clf = MLPClassifier(solver=best_params['solver'], alpha=best_params['alpha'],
                             hidden_layer_sizes=best_params['hidden_layer_sizes'], random_state=0,
                             activation=best_params['activation'],learning_rate=best_params['learning_rate'],
                             max_iter=5000)

        clf=best_clf
        clf.fit(X_train, y_train)
        Experiments.plot_l_curve_website(clf, title + " Tuned", X, y, axes=None, ylim=None, cv=10)


    '''
    best_clf = MLPClassifier(solver='adam', alpha=0.0001,
                             hidden_layer_sizes=(5,2), random_state=0,
                             activation='tanh', learning_rate='constant',
                             max_iter=10000)
    '''


    #best_clf=clf
    print("Confusion matrix for NN is:")
    clf.fit(X_train, y_train)
    t0=time.time()
    ytest_pred = clf.predict(X_test)
    t1=time.time()
    pred_time=t1-t0
    print('Inference time on data is %f seconds' % pred_time)
    Experiments.ConfusionMatrix(y_test, ytest_pred)
    class_names = [0, 1]
    disp = plot_confusion_matrix(clf, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues)
    disp.ax_.set_title(title + " Confusion Matrix")

    print(title)
    print(disp.confusion_matrix)
    plt.savefig(title + " Confusion Matrix")
    plt.show()

    print("#######################################################")

    #train_size_array = np.linspace(0.1, 1, 10)
    #Experiments.plot_learning_curve(best_clf, X, y, cv, train_size_array, title=title)

    #best_clf=clf
    # Learning Curve based on best tuned Neural Network
    # train_size_array=([0.1, 0.33, 0.55, 0.78, 1.])
    #train_size_array = np.linspace(0.1, 1, 10)
   # plot = Experiments.plot_learning_curve(estimator=best_clf, title=title + " Neural Network Learning Curve", X=X, y=y,
                                           #ylim=None, cv=10,
                                           #n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))



