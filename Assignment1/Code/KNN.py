import numpy as np
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import Experiments
import time
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

def KNN(X, y, title, optimize):
    print("KNN Results")
    #X, y= LoadData.load_wine_quality_data('./Data/winequality-red.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

    #Use default K first
    clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
    cval_score=(cross_val_score(clf, X_train,y_train, cv=10)).mean()
    clf.fit(X_train, y_train)
    ytest_pred=clf.predict(X_test)
    accuracy=accuracy_score(y_test, ytest_pred)

    print("Mean cross validation score: %.2f%%" % (cval_score * 100))
    print("Accuracy score before tuning: %.2f%%" % (accuracy * 100))

    Experiments.plot_l_curve_website(clf, title, X, y, axes=None, ylim=None, cv=10)

    print("Confusion matrix for KNN is:")
    t0=time.time()
    clf.fit(X_train, y_train)
    t1=time.time()
    train_time=t1-t0
    print('Completed training in %f seconds' % train_time)

    t0 = time.time()
    ytest_pred = clf.predict(X_test)
    t1 = time.time()
    pred_time = t1 - t0
    print('Inference time on data is %f seconds' % pred_time)

    if optimize == True:
        k_range = list(range(1, 51))
        param_grid = dict(n_neighbors=k_range)
        cv = 10
        best_params = Experiments.Tuning(clf, tuned_params=param_grid, cv=cv, Xtrain=X_train,
                                         Ytrain=y_train, Xtest=X_test, Ytest=y_test)
        best_clf = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights="distance")
        clf = best_clf
        clf.fit(X_train, y_train)
        Experiments.plot_l_curve_website(clf, title + " Tuned", X, y, axes=None, ylim=None, cv=10)

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

    # GridSearchCV for KNN

    # Learning Curve based on best tuned KNN
    # train_size_array=([0.1, 0.33, 0.55, 0.78, 1.])
    #train_size_array = np.linspace(0.1, 1, 10)
    #plot=Experiments.plot_learning_curve(estimator=best_clf, title=title + " KNN Learning Curve", X=X, y=y, ylim=None, cv=10,
                                         #n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))

    print("##############################################")
    #train_size_array = np.linspace(0.1, 1, 10)
    #Experiments.plot_learning_curve(best_clf, X, y, cv, train_size_array, title=title)
    '''
    for K in range(1, 50):
        clf = KNeighborsClassifier(K, weights="distance")
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        clf = clf.fit(X_train, y_train)
        test_predict = clf.predict(X_test)
        KNN.append(accuracy_score(y_test, test_predict))
        list2.append(sum(scores) / len(scores))

    plt.plot(range(len(KNN)), KNN)
    plt.plot(range(len(list2)), list2)
    plt.show()

    best_clf = KNeighborsClassifier(3, weights="distance")
    train_size = list(np.arange(0.1, 1, 0.1))
    Y=y

    train_size_array = ([0.1, 0.33, 0.55, 0.78, 1.])


    Experiments.plot_learning_curve(best_clf, X, Y, cv=5, train_size_array=train_size_array)
    '''
