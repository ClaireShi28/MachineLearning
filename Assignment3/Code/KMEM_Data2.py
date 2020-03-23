import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

import time
from sklearn.cluster import KMeans


from sklearn.metrics import adjusted_mutual_info_score,adjusted_rand_score,homogeneity_completeness_v_measure


import LoadData
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


np.random.seed(6)

if __name__ == '__main__':
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

    #For Wine Data Set 1
    km_dic = {"n_clusters": 6, "init": "k-means++",  "max_iter": 500}
    EM_dic = {"n_components": 6, "init_params": "kmeans", "max_iter": 500}

    #n_candidates = [i*5 + 1 for i in range(1,20)]
    n_candidates=[2,3,4,5,6,7,8,9,10,15,20,30,40,50]

    nclusters=[]
    KM_score=[]
    KM_homo=[]
    KM_com=[]
    KM_v=[]
    KM_s=[]

    EM_score = []
    EM_homo = []
    EM_com = []
    EM_v = []
    EM_s = []

    KM_time=[]
    EM_time=[]
    AIC=[]
    BIC=[]
    Inertia=[]
    round_start = time.time()
    for n in n_candidates:
        km_dic["n_clusters"] = n
        EM_dic["n_components"] = n

        row = [n]
        nclusters.append(n)
        #KMeans clustering
        model = KMeans(**km_dic)
        start_time = time.time()
        model.fit(features2)
        preds2 = model.predict(features2)
        #model.fit(X_train)
        #y_test_pred = model.predict(X_test)
        end_time = time.time()

        # Silhoutette score
        #sil = metrics.silhouette_score(X_test, y_test_pred, metric='euclidean')

        # Variance explained by the cluster
        #var = clf.score(X_test)
        #array_var.append(var)
        #homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_test, y_test_pred)
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(labels2, preds2)
        sil=metrics.silhouette_score(features2, preds2, metric='euclidean')
        KM_homo.append(homogeneity)
        KM_com.append(completeness)
        KM_v.append(v_measure)
        KM_s.append(sil)
        KM_time.append(end_time-start_time)
        KM_score.append(model.score(features2))
        Inertia.append(model.inertia_)


        #row += [-model.score(features), adjusted_mutual_info_score(labels, preds), adjusted_rand_score(labels, preds),
                #homogeneity, completeness, v_measure, end_time - start_time]

        model = GaussianMixture(**EM_dic)
        start_time = time.time()
        model.fit(features2)
        preds2 = model.predict(features2)
        #model.fit(X_train)
        #y_test_pred = model.predict(X_test)
        end_time = time.time()
        #homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_test, y_test_pred)
        #sil = metrics.silhouette_score(X_test, y_test_pred, metric='euclidean')
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(labels2, preds2)
        sil = metrics.silhouette_score(features2, preds2, metric='euclidean')
        EM_homo.append(homogeneity)
        EM_com.append(completeness)
        EM_v.append(v_measure)
        EM_s.append(sil)
        AIC.append(model.aic(features2))
        BIC.append(model.bic(features2))

        EM_time.append(end_time-start_time)
        EM_score.append(model.score(features2))

    round_end = time.time()

    print("Totoal Time:", round_end - round_start)

    # score
    plt.figure()
    plt.grid(True)
    plt.plot(nclusters, KM_homo, color='blue',  label="KM homogeneity")
    plt.plot(nclusters, KM_com, color='green', label="KM completeness")
    plt.plot(nclusters, KM_v, color='red', label="KM v-measure")
    plt.plot(nclusters, KM_s, color='orange', label="KM silhouette")


    plt.legend(loc='best')
    plt.title("KM performance evaluation on dataset2")
    plt.xlabel("Number of clusters")
    plt.ylabel("Score")
    plt.savefig("KM performance Data2")

    plt.figure()
    plt.grid(True)
    plt.plot(nclusters, EM_homo, linestyle='dashed', color='blue', label="EM homogeneity")
    plt.plot(nclusters, EM_v, linestyle='dashed', color='red', label="EM v-measure")
    plt.plot(nclusters, EM_com, linestyle='dashed', color='green', label="EM completeness")
    plt.plot(nclusters, EM_s, linestyle='dashed', color='orange', label="EM silhouette")
    plt.legend(loc='best')
    plt.title("EM performance evaluation on dataset2")
    plt.xlabel("Number of clusters")
    plt.ylabel("Score")
    plt.savefig("EM performance Data2")

    #Runtime
    plt.figure()
    plt.plot(nclusters,KM_time, label="KM")
    plt.plot(nclusters, EM_time, label="EM", linestyle='dashed')
    plt.title("KM/EM running time on dataset2")
    plt.legend(loc='best')
    plt.xlabel("Number of clusters")
    plt.ylabel("Running time (s)")
    plt.grid(True)
    plt.savefig("KMvsEM Time Data2")

    #KM score
    plt.figure()
    plt.plot(nclusters, KM_score,marker='o')
    plt.title("KM score for dataset2")
    plt.grid(True)
    plt.xlabel("Number of clusters")
    plt.ylabel("Sum of squared distance")
    plt.savefig("KM Score Data2")

    # KM inertia
    plt.figure()
    plt.plot(nclusters, Inertia, marker='o')
    plt.title("KM Inertia Data2")
    plt.grid(True)
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.savefig("KM Inertia Data2")

    #EM score
    plt.figure()
    plt.plot(nclusters, EM_score, marker='o',linestyle='dashed')
    plt.title("EM score for dataset2")
    plt.grid(True)
    plt.xlabel("Number of clusters")
    plt.ylabel("Likelihood")
    plt.tight_layout(True)
    plt.savefig("EM Score Data2")

    #BIC.AIC score
    plt.figure()
    plt.plot(nclusters,BIC, marker='o',linestyle='dashed')
    #plt.plot(nclusters, AIC, label="AIC")
    plt.xlabel("Number of clusters")
    plt.title("EM BIC score for dataset2")
    plt.grid(True)
    plt.ylabel("BIC score")
    plt.savefig("EM BIC Data2")