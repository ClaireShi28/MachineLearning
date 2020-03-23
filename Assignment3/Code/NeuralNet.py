import numpy as np
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import LoadData
import time
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA,TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import time
from sklearn.cluster import KMeans
from sklearn import metrics
import LoadData

X1, y1 = LoadData.load_wine_quality_data()
labels1 = y1
features1 = X1


scaler = StandardScaler()
data_scaled1 = scaler.fit_transform(features1)
X1 = data_scaled1



def NeuralNetwork(X, y, title, optimize):
    print("Neural Network Results")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)
    clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(100,), random_state=0, activation='logistic',
                        max_iter=10000)
    cval_score = (cross_val_score(clf, X_train, y_train, cv=10)).mean()
    t0 = time.time()
    clf.fit(X_train, y_train)
    t1 = time.time()
    train_time = t1 - t0
    print('Completed training in %f seconds' % train_time)
    ytest_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, ytest_pred)

    print("Mean cross validation score: %.2f%%" % (cval_score * 100))
    print("Accuracy score before tuning: %.2f%%" % (accuracy * 100))
    # print("Confusion matrix for neural network is:")
    # Experiments.ConfusionMatrix(y_test, ytest_pred)
    plot_l_curve_website(clf, title, X, y, axes=None, ylim=None, cv=10)

    cv = 10

    if optimize == True:
        # GridSearchCV for NN
        param_grid = {
            'hidden_layer_sizes': [(5, 2), (50,), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive'],
        }

        #best_params = Experiments.Tuning(clf, tuned_params=param_grid, cv=cv, Xtrain=X_train,
                                         #Ytrain=y_train, Xtest=X_test, Ytest=y_test)

        best_clf = MLPClassifier(solver='adam', alpha=0.0001,
                                 hidden_layer_sizes=(5, 2), random_state=0,
                                 activation='tanh', learning_rate='constant',
                                 max_iter=10000)
        '''
        best_clf = MLPClassifier(solver=best_params['solver'], alpha=best_params['alpha'],
                                 hidden_layer_sizes=best_params['hidden_layer_sizes'], random_state=0,
                                 activation=best_params['activation'], learning_rate=best_params['learning_rate'],
                                 max_iter=5000)
        '''
        clf = best_clf
        clf.fit(X_train, y_train)
        plot_l_curve_website(clf, title + " Tuned", X, y, axes=None, ylim=None, cv=10)

    '''
    best_clf = MLPClassifier(solver='adam', alpha=0.0001,
                             hidden_layer_sizes=(5,2), random_state=0,
                             activation='tanh', learning_rate='constant',
                             max_iter=10000)
    '''


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
                 label="Testing score")
    axes[0].legend(loc="best")


    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean,'o-',color='g')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1,color='g')
    axes[1].set_xlabel("Training Size (%)")
    axes[1].set_ylabel("Fit Times (s)")
    axes[1].set_title("Scalability of the model")
    plt.tight_layout(True)
    plt.savefig(title+" Learning Curve")
    plt.show()
#The above learning curve plot code is from scikit-learn user guide:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

'''
start_time = time.time()
#km = KMeans(**km_dic)

NeuralNetwork(X1, y1, "Dataset1 NN", optimize=False)
end_time = time.time()
print("NN total time: "+ str(end_time-start_time))


#PCA
start_time = time.time()
#km = KMeans(**km_dic)
proj = PCA(n_components=7).fit_transform(X1)
NeuralNetwork(proj, y1, "Dataset1 PCA + NN", optimize=False)
end_time = time.time()
print("PCA + NN total time: "+ str(end_time-start_time))

start_time = time.time()
proj = GaussianRandomProjection(n_components=7).fit_transform(X1)
NeuralNetwork(proj, y1, "Dataset1 RP + NN", optimize=False)
print("RP + NN total time: "+ str(end_time-start_time))

start_time = time.time()
proj = TruncatedSVD(n_components=7).fit_transform(X1)
NeuralNetwork(proj, y1, "Dataset1 SVD  + NN", optimize=False)
end_time = time.time()
print("SVD + total time: "+ str(end_time-start_time))

start_time = time.time()
proj =FastICA(n_components=7).fit_transform(X1)
NeuralNetwork(proj, y1, "Dataset1 ICA  + NN", optimize=False)
end_time = time.time()
print("ICA + total time: "+ str(end_time-start_time))
'''

#KM + Reduction
#PCA
km_dic = {"n_clusters": 7, "init": "k-means++", "max_iter": 500}
em_dic = {"n_components": 10, "init_params": "kmeans", "max_iter": 500}

KMdf=pd.DataFrame(columns=['homogeneity','completness','v_measure','silhoutette','time(s)'])


print ("KM ########################")
start_time = time.time()
km = KMeans(**km_dic)
proj=PCA(n_components=7).fit_transform(X1)
preds1 = km.fit_predict(proj)

X=np.column_stack((preds1, X1))

NeuralNetwork(X, y1, "Dataset1 KM PCA + NN", optimize=False)
end_time = time.time()

start_time = time.time()
km = KMeans(**km_dic)
proj=FastICA(n_components=7).fit_transform(X1)
preds1 = km.fit_predict(proj)

X=np.column_stack((preds1, X1))

NeuralNetwork(X, y1, "Dataset1 KM ICA + NN", optimize=False)
end_time = time.time()

start_time = time.time()
km = KMeans(**km_dic)
proj=GaussianRandomProjection(n_components=7).fit_transform(X1)
preds1 = km.fit_predict(proj)

X=np.column_stack((preds1, X1))

NeuralNetwork(X, y1, "Dataset1 KM RP + NN", optimize=False)
end_time = time.time()

start_time = time.time()
km = KMeans(**km_dic)
proj=TruncatedSVD(n_components=7).fit_transform(X1)
preds1 = km.fit_predict(proj)

X=np.column_stack((preds1, X1))
NeuralNetwork(X, y1, "Dataset1 KM SVD + NN", optimize=False)
end_time = time.time()


print("EM###############################")

start_time = time.time()
km = GaussianMixture(**em_dic)
proj=PCA(n_components=7).fit_transform(X1)
preds1 = km.fit_predict(proj)

X=np.column_stack((preds1, X1))

NeuralNetwork(X, y1, "Dataset1 EM PCA + NN", optimize=False)
end_time = time.time()

start_time = time.time()
km = GaussianMixture(**em_dic)
proj=FastICA(n_components=7).fit_transform(X1)
preds1 = km.fit_predict(proj)

X=np.column_stack((preds1, X1))

NeuralNetwork(X, y1, "Dataset1 EM ICA + NN", optimize=False)
end_time = time.time()

start_time = time.time()
km = GaussianMixture(**em_dic)
proj=GaussianRandomProjection(n_components=7).fit_transform(X1)
preds1 = km.fit_predict(proj)

X=np.column_stack((preds1, X1))

NeuralNetwork(X, y1, "Dataset1 EM RP + NN", optimize=False)
end_time = time.time()

start_time = time.time()
km = GaussianMixture(**em_dic)
proj=TruncatedSVD(n_components=7).fit_transform(X1)
preds1 = km.fit_predict(proj)

X=np.column_stack((preds1, X1))
NeuralNetwork(X, y1, "Dataset1 EM SVD + NN", optimize=False)
end_time = time.time()
