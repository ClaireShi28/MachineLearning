import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
import Experiments
import time
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

def AdaBoost(X,y, title, optimize):
    print("AdaBoost Results")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
    clf=AdaBoostClassifier()
    cval_score = (cross_val_score(clf, X_train, y_train, cv=10)).mean()

    t0=time.time()
    clf.fit(X_train, y_train)
    t1=time.time()
    train_time=t1-t0
    print('Completed training in %f seconds' % train_time)

    t0=time.time()
    ytest_pred = clf.predict(X_test)
    t1=time.time()
    pred_time=t1-t0
    print('Inference time on data is %f seconds' % pred_time)
    accuracy = accuracy_score(y_test, ytest_pred)

    Experiments.plot_l_curve_website(clf, title, X, y, axes=None, ylim=None, cv=10)

    #train_size_array = np.linspace(0.1, 1, 10)
    #Experiments.plot_learning_curve(clf, X, y, cv=10, train_size_array=np.linspace(.1, 1.0, 10), title=title)

    print("Mean cross validation score: %.2f%%" % (cval_score * 100))
    print("Accuracy score before tuning: %.2f%%" % (accuracy * 100))
    #print("Confusion matrix for AdaBoost is:")
    #Experiments.ConfusionMatrix(y_test, ytest_pred)

    cv = 10
    # GridSearchCV for AdaBoost
    if optimize==True:
        n_range=list(range(1,101))
        param_grid=dict(n_estimators=n_range)
        best_params = Experiments.Tuning(clf, tuned_params=param_grid, cv=cv, Xtrain=X_train,
                                     Ytrain=y_train, Xtest=X_test, Ytest=y_test)
        best_clf=AdaBoostClassifier(n_estimators=best_params['n_estimators'])
        clf=best_clf
        clf.fit(X_train, y_train)
        Experiments.plot_l_curve_website(clf, title + " Tuned", X, y, axes=None, ylim=None, cv=10)

    print("Confusion matrix for AdaBoostS is:")
    clf.fit(X_train, y_train)
    ytest_pred = clf.predict(X_test)
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
    # Learning Curve based on best tuned AdaBoost
    # train_size_array=([0.1, 0.33, 0.55, 0.78, 1.])
    #plot=Experiments.plot_learning_curve(estimator=best_clf, title=title + " AdaBoost Learning Curve", X=X, y=y, ylim=None, cv=10,
                                    #n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))




