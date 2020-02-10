import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV, learning_curve
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

#To print confusion matrix
def ConfusionMatrix(Ytrue, Ypred):
    #print ("???????????????")
    #cm=confusion_matrix(y_true=Ytrue,y_pred=Ypred)
    #print(cm)
    print (pd.crosstab(Ytrue,Ypred, rownames=['True'], colnames=['Predicted'], margins=True)) #Same as confusion_matrix

#For tuning the models
def Tuning(clf, tuned_params,cv, Xtrain, Ytrain, Xtest, Ytest):
    # 2. Hyperparameter tuning

    num_classifiers = 5
    best_accuracy = np.zeros(num_classifiers)
    train_time = np.zeros(num_classifiers)
    test_time = np.zeros(num_classifiers)

    clf = GridSearchCV(clf, param_grid=tuned_params, iid=True, cv=cv, scoring='accuracy')
    #t0 = time.time()
    clf.fit(Xtrain, Ytrain)
    #t1 = time.time()
    #train_time[0] = t1 - t0
    #print('Completed training in %f seconds' % train_time[0])
    #best_clf = clf

    best_params = clf.best_params_
    print("Best parameters set for decision tree found on development set:")
    print(best_params)
    #t0 = time.time()
    Ypred = clf.predict(Xtest)
    #t1 = time.time()
    #test_time[0] = t1 - t0
    #print('Inference time on test data: %f seconds' % test_time[0])
    best_accuracy[0] = accuracy_score(Ytest, Ypred)
    print('Accuracy score after tunning %.2f%%' % (best_accuracy[0] * 100))

    return best_params

#The below learning curve plot code is from scikit-learn user guide:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_l_curve_website(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 10)):

    #print (train_sizes)
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].set_title(title+" Learning Curve")
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training Size (%)")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    train_size_array=100*np.linspace(0.1, 1, 10)
    train_sizes=train_size_array
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="b")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="r")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="b",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="r",
                 label="Cross-validation score")
    axes[0].legend(loc="best")


    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean,'o-',color='g')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1,color='g')
    axes[1].set_xlabel("Training Size (%)")
    axes[1].set_ylabel("Fit Times (s)")
    axes[1].set_title("Scalability of the model")

    plt.savefig(title+" Learning Curve")
    plt.show()
#The above learning curve plot code is from scikit-learn user guide:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

#To plot comparison graphs of 5 models
def Models_Comparison(X, y, title):
    train_time=[]
    pred_time=[]
    classifiers=['Decision Tree', 'Neural Network', 'AdaBoost', 'KNN', 'SVM']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
    accuracy=[]

    #DT time and accuracy
    t0 = time.time()
    clf=DecisionTreeClassifier(max_depth=1, min_samples_leaf=10, random_state=0)
    clf.fit(X_train, y_train)
    t1 = time.time()
    train_time.append(t1 - t0)

    t0 = time.time()
    ytest_pred = clf.predict(X_test)
    t1 = time.time()
    pred_time.append(t1 - t0)
    accuracy.append(accuracy_score(y_test, ytest_pred))

    #NN time and accuracy
    t0 = time.time()
    clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(100,), random_state=0, activation='logistic',
                        max_iter=10000)
    clf.fit(X_train, y_train)
    t1 = time.time()
    train_time.append(t1 - t0)

    t0 = time.time()
    ytest_pred = clf.predict(X_test)
    t1 = time.time()
    pred_time.append(t1 - t0)
    accuracy.append(accuracy_score(y_test, ytest_pred))

    #AdaBoost time and accuracy
    t0 = time.time()
    clf=AdaBoostClassifier()
    clf.fit(X_train, y_train)
    t1 = time.time()
    train_time.append(t1 - t0)

    t0 = time.time()
    ytest_pred = clf.predict(X_test)
    t1 = time.time()
    pred_time.append(t1 - t0)
    accuracy.append(accuracy_score(y_test, ytest_pred))

    #KNN time and accuracy
    t0 = time.time()
    clf=KNeighborsClassifier(n_neighbors=5, weights="distance")
    clf.fit(X_train, y_train)
    t1 = time.time()
    train_time.append(t1 - t0)

    t0 = time.time()
    ytest_pred = clf.predict(X_test)
    t1 = time.time()
    pred_time.append(t1 - t0)
    accuracy.append(accuracy_score(y_test, ytest_pred))

    #SVM time and accuracy
    t0 = time.time()
    clf = svm.SVC(random_state=0)
    clf.fit(X_train, y_train)
    t1 = time.time()
    train_time.append(t1 - t0)

    t0 = time.time()
    ytest_pred = clf.predict(X_test)
    t1 = time.time()
    pred_time.append(t1 - t0)
    accuracy.append(accuracy_score(y_test, ytest_pred))

    #Plot accuracy on test data comparison (tuned)
    #Based on results before and after tuning
    wine_accuracy_tuned=[0.7208, 0.7375, 0.7125, 0.7333, 0.7042]
    air_accuracy_tuned = [0.9150, 0.9145, 0.9203, 0.9065, 0.9224]

    if title=="Wine Quality":
        plt.bar(classifiers, wine_accuracy_tuned, color='pink')
        plt.title("Wine Quality Prediction Accuracy Comparison")
        plt.ylabel('Accuracy')
        plt.xlabel('Models (Tuned)')
        plt.ylim(bottom=0, top=1.1)
        plt.tight_layout()
        plt.savefig("Wine Quality Tuned Prediction Accuracy Comparison")
        plt.show()

    if title=="Air Pollution":
        plt.bar(classifiers, air_accuracy_tuned, color='pink')
        plt.title("Air Quality Prediction Accuracy Comparison")
        plt.ylabel('Accuracy')
        plt.xlabel('Models (Tuned)')
        plt.ylim(bottom=0, top=1.1)
        plt.tight_layout()
        plt.savefig("Air Quality Tuned Prediction Accuracy Comparison")
        plt.show()

    #Plot accuracy on test data comparison (untuned models)
    plt.bar(classifiers, accuracy, color='g')
    plt.title(title+" Prediction Accuracy Comparison")
    plt.ylabel('Accuracy')
    plt.xlabel('Models')
    plt.ylim(bottom=0,top=1.1)
    plt.tight_layout()
    plt.savefig(title+" Prediction Accuracy Comparison")
    plt.show()

    #Plot training time comparison (untuned models)
    plt.bar(classifiers, train_time)
    plt.title(title + " Training Time Comparison among Models")
    plt.ylabel('Time (s')
    plt.xlabel('Models')
    # plt.ylim(bottom=0,top=1.1)
    # plt.plot(train_size, score, 'o-', label='score')
    plt.tight_layout()
    plt.savefig(title + " Training Time Comparison among Models")
    plt.show()

    #plot inference time comparison
    plt.bar(classifiers, pred_time, color='orange')
    plt.title(title + " Inference Time Comparison among Models")
    plt.ylabel('Time (s')
    plt.xlabel('Models')
    plt.tight_layout()
    plt.savefig(title + " inference Time Comparison among Models")
    plt.show()

