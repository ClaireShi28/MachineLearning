import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
import Experiments
import time
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

def SVM(X,y, title, optimize=True):
    print('SVM results')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
    '''
    for kernel in ('linear', 'rbf','sigmoid'):
        clf = svm.SVC(kernel=kernel, gamma=2, random_state=0)
        clf.fit(X_train, y_train)
        test_predict = clf.predict(X_test)
        print(kernel+":"+str(accuracy_score(y_test, test_predict)))
    '''

    #Choose RBF as the preferred kernel function
    clf = svm.SVC(random_state=0)
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
    accuracy=accuracy_score(y_test,ytest_pred)

    print("Accuracy score of SVM with RBF kernel before tuning: %.2f%%"% (accuracy * 100))
    Experiments.plot_l_curve_website(clf, title, X, y, axes=None, ylim=None, cv=10)
    #print("Confusion Matrix for SVM is:")
    #Experiments.ConfusionMatrix(y_test, clf.predict(X_test))

    cv = 10
    #GridSearchCV for SVM
    if optimize==True:
        Cs=[0.001, 0.01, 0.1, 1, 10]
        gammas=[0.001, 0.01, 0.1, 1]

        #kernels=['rbf','linear','poly']
        param_grid = {'C': Cs, 'gamma': gammas}


        best_params=Experiments.Tuning(clf, tuned_params=param_grid,cv=cv, Xtrain=X_train,
                       Ytrain=y_train, Xtest=X_test, Ytest=y_test)
        best_clf = svm.SVC(C=best_params['C'], gamma=best_params['gamma'],random_state=100)
        clf=best_clf
        clf.fit(X_train, y_train)
        Experiments.plot_l_curve_website(clf, title + " Tuned", X, y, axes=None, ylim=None, cv=10)
    #best_clf = svm.SVC(kernel=best_params[0], random_state=0)

    print("Confusion matrix for SVN is:")
    clf.fit(X_train, y_train)
    ytest_pred = clf.predict(X_test)


    print("Confusion matrix for SVN is:")
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

    #best_clf=clf
    print("#######################################################")

    #train_size_array = np.linspace(0.1, 1, 10)
    #Experiments.plot_learning_curve(best_clf, X, y, cv, train_size_array, title=title)
    #Learning Curve based on best tuned SVM


