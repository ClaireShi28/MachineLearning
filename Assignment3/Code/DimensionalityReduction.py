from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.random_projection import GaussianRandomProjection as GRP, SparseRandomProjection as RCA
from sklearn.ensemble import RandomForestClassifier as RFC
from itertools import product
from collections import defaultdict
from sklearn import random_projection

from sklearn.mixture import GaussianMixture

import time
from sklearn.cluster import KMeans


from sklearn.metrics import adjusted_mutual_info_score,adjusted_rand_score,homogeneity_completeness_v_measure

from sklearn.metrics.pairwise import pairwise_distances
import LoadData

X1, y1 = LoadData.load_wine_quality_data()
labels1 = y1
features1 = X1

X2, y2 = LoadData.load_ozone_data()
labels2 = y2
features2 = X2

scaler = StandardScaler()
data_scaled1 = scaler.fit_transform(features1)
features1 = data_scaled1

scaler = StandardScaler()
data_scaled2 = scaler.fit_transform(features2)
features2 = data_scaled2

features1=pd.DataFrame(features1)
features2=pd.DataFrame(features2)

def PCA_exp(X, y, title):
    pca = PCA(random_state=6)
    pca.fit(X)  # for all components
    cum_var = np.cumsum(pca.explained_variance_ratio_)


    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.bar(list(range(len(pca.singular_values_))), pca.singular_values_, color='orange')
    ax1.set_ylabel('Eigenvalues', color='orange')
    ax1.tick_params('y', colors='orange')

    ax2.plot(list(range(len(pca.explained_variance_ratio_))), cum_var, 'b-')
    ax1.set_xlabel('Principal components')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax2.set_ylabel('Cumulative Explained Variance Ratio', color='b')
    ax2.tick_params('y', colors='b')

    plt.title("PCA cumulative explained variance and eigenvalues: " + title)
    fig.tight_layout()
    plt.savefig("PAC "+title)
    plt.show()


def ICA_exp(X, y, title):
    from scipy.stats import kurtosis, entropy

    ica = FastICA()
    temp = ica.fit_transform(X)
    order = [-abs(kurtosis(temp[:, i])) for i in range(temp.shape[1])]
    temp = temp[:, np.array(order).argsort()]
    ica_res = pd.Series([abs(kurtosis(temp[:, i])) for i in range(temp.shape[1])])
    plt.figure()
    ax = ica_res.plot(kind='bar', logy=True)
    ticks = ax.xaxis.get_ticklocs()
    ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
    ax.xaxis.set_ticks(ticks[::10])
    ax.xaxis.set_ticklabels(ticklabels[::10])

    plt.title("Featured Kurtosis Distribution: " +title)
    plt.xlabel("Features ordered by kurtosis")
    plt.ylabel("Kurtosis")
    plt.savefig("ICA "+title)
    plt.show()


def RP_exp(X, y, title):
    ncomp= [i+1 for i in range(X.shape[1]-1)]
    stdev=[]
    mean=[]
    for n in ncomp:
        repeats = []
        for i in range(5):
            rp = GaussianRandomProjection(n_components=n)
            temp = rp.fit_transform(X)
            repeats.append(temp)

        diffs = []
        for (i, j) in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]:
            diffs.append(repeats[i] - repeats[j])
        stdev.append(np.std(diffs))
        mean.append(np.mean(diffs))

    comp_arr=np.array(ncomp)
    mean_arr=np.array(mean)
    stdev_arr=np.array(stdev)

    plt.fill_between(comp_arr, mean_arr-stdev_arr,
                    mean_arr + stdev_arr, alpha=0.1,
                         color="b", label="Stdev")
    plt.plot(ncomp, mean, 'o-', color="b", label="Mean")
    plt.title("Mean pairwise difference of RP: "+ title)
    plt.legend(loc='best')
    plt.xlabel("n_components")
    plt.ylabel("Pairwise difference")
    plt.savefig("RP "+title)
    plt.show()


def SVD_exp(X,y, title):
    svd = TruncatedSVD(n_components=X.shape[1] - 1)
    proj = svd.fit_transform(X)
    cum_var = np.cumsum(svd.explained_variance_ratio_)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.bar(list(range(len(svd.explained_variance_))), svd.explained_variance_, color='orange')
    ax1.set_ylabel('Variance', color='orange')
    ax1.tick_params('y', colors='orange')

    ax2.plot(list(range(len(svd.explained_variance_ratio_))), cum_var, 'b-')
    ax1.set_xlabel('Features')

    ax2.set_ylabel('Cumulative Explained Variance Ratio', color='b')
    ax2.tick_params('y', colors='b')
    plt.title("Variance Distribution of SVD: "+title)
    plt.xlabel("Features")
    plt.ylabel("Variance")
    plt.show()
    plt.savefig("SVD "+title)


def ReconstructData(X,y,title):
    n_comp = [1+ i for i in range(X.shape[1])]
    n_comp2 = [1 + i for i in range(X.shape[1]-1)]

    reconstruct_ICA_mean = []
    reconstruct_PCA_mean=[]

    reconstruct_ICA_std = []
    reconstruct_PCA_std = []

    reconstruct_SVD_mean = []
    reconstruct_SVD_std = []
    for n in list(n_comp):
        pca = PCA(n_components=n)
        projp = pca.fit_transform(X)
        diffp = X.values - pca.inverse_transform(projp)
        reconstruct_PCA_mean.append(np.mean(diffp))
        reconstruct_PCA_std.append(np.std(diffp))

        #print (np.mean(diffp),np.std(diffp))
        ica = FastICA(n_components=n, tol=0.005)
        proji = ica.fit_transform(X)
        diffi = X.values - ica.inverse_transform(proji)
        reconstruct_ICA_mean.append(np.mean(diffi))
        reconstruct_ICA_std.append(np.std(diffi))
        #print(np.mean(diffi), np.std(diffi))
        #print ("$$$$$$$$$$$$$$$$$$$$$$$$$")

    plt.figure()
    plt.fill_between(n_comp, np.array(reconstruct_ICA_mean) - np.array(reconstruct_ICA_std),
                     np.array(reconstruct_ICA_mean) + np.array(reconstruct_ICA_std), alpha=0.1,
                     color="b", label="Stdev")
    plt.plot(n_comp, np.array(reconstruct_ICA_mean), label="mean")
    plt.title("Reconstruction difference of ICA: " + title)
    plt.legend(loc='best')
    plt.xlabel("n_components")
    plt.ylabel("Difference")
    plt.savefig("Reconstruct diff ICA " + title)
    plt.show()

    plt.figure()
    plt.fill_between(n_comp, np.array(reconstruct_PCA_mean) - np.array(reconstruct_PCA_std),
                     np.array(reconstruct_PCA_mean) + np.array(reconstruct_PCA_std), alpha=0.1,
                     color="b", label="Stdev")
    plt.plot(n_comp, np.array(reconstruct_PCA_mean), label="mean")
    plt.title("Reconstruction difference of PCA: " + title)
    plt.legend(loc='best')
    plt.xlabel("n_components")
    plt.ylabel("Difference")
    plt.savefig("Reconstruct diff PCA " + title)
    plt.show()

    for n in list(n_comp2):
        svd = TruncatedSVD(n_components=n)
        projs = svd.fit_transform(X)
        diffs = X.values - svd.inverse_transform(projs)
        reconstruct_SVD_mean.append(np.mean(diffs))
        reconstruct_SVD_std.append(np.std(diffs))

    plt.figure()
    plt.fill_between(n_comp2, np.array(reconstruct_SVD_mean) - np.array(reconstruct_SVD_std),
                     np.array(reconstruct_SVD_mean) + np.array(reconstruct_SVD_std), alpha=0.1,
                     color="b", label="Stdev")
    plt.plot(n_comp2, np.array(reconstruct_SVD_mean), label="mean")
    plt.title("Reconstruction difference of SVD: " + title)
    plt.legend(loc='best')
    plt.xlabel("n_components")
    plt.ylabel("Difference")
    plt.savefig("Reconstruct diff SVD " + title)
    plt.show()


if __name__ == '__main__':
    PCA_exp(features1, labels1, "Dataset1")
    PCA_exp(features2,labels2, "Dataset2")

    ICA_exp(features1, labels1, "Dataset1")
    ICA_exp(features2, labels2, "Dataset2")

    RP_exp(features1, labels1, "Dataset1")
    RP_exp(features2, labels2, "Dataset2")

    SVD_exp(features1, labels1, "Dataset1")
    SVD_exp(features2, labels2, "Dataset2")

    ReconstructData(features1, labels1,"Dataset1")
    ReconstructData(features2, labels2, "Dataset2")






