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

X2, y2 = LoadData.load_ozone_data()
labels2 = y2
features2 = X2

scaler = StandardScaler()
data_scaled1 = scaler.fit_transform(features1)
features1 = data_scaled1

scaler = StandardScaler()
data_scaled2 = scaler.fit_transform(features2)
features2 = data_scaled2

#7 for wine data, 6 for PM2.5 data
#data1 km em
#data2 km 15 em 20
def KmeansEM(X,y,title,km,em, dr, str):

    temp = dr(n_components=2).fit_transform(X)

    km_dic = {"n_clusters": km, "init": "k-means++", "max_iter":1000}
    res = KMeans(**km_dic).fit_predict(temp)

    plt.figure()
    plt.scatter(temp[:, 0], temp[:, 1], c=res, alpha=0.5)
    plt.title(str+" and KM:" +title)
    plt.savefig(str + "and KM:" +title)

    EM_dic = {"n_components": em, "init_params": "kmeans","max_iter":1000}
    res = GaussianMixture(**EM_dic).fit(temp).predict(temp)

    plt.figure()
    plt.scatter(temp[:, 0], temp[:, 1], c=res, alpha=0.5)
    plt.title(str+" and EM:" +title)
    plt.savefig(str+" and EM:" + title)


    plt.show()

def CompareData1(X,y,title,km,em):
    algs = ["PCA", "ICA", "RP ", "SVD"]
    km_dic = {"n_clusters": 6, "init": "k-means++", "max_iter": 500}
    em_dic = {"n_components": 6, "init_params": "kmeans", "max_iter": 500}

    km_dic["n_clusters"] = km
    em_dic["n_components"] = em

    KMdf=pd.DataFrame(columns=['homogeneity','completness','v_measure','silhoutette','time(s)'])


    print ("KM ########################")
    start_time = time.time()
    km = KMeans(**km_dic)
    proj=PCA(n_components=7).fit_transform(X)
    preds1 = km.fit_predict(proj)
    end_time = time.time()

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y, preds1)
    sil = metrics.silhouette_score(X, preds1, metric='euclidean')

    print("PCA (n=7)")
    print(title+":")
    print("homogeneity:" +"%0.2f"%homogeneity)
    print("completnes:"+"%0.2f"%completeness)
    print("v_measure:"+"%0.2f"%v_measure)
    print("sil:"+"%0.2f"%sil)
    print("time"+"%0.2f"%(end_time-start_time))

    start_time = time.time()
    km = KMeans(**km_dic)
    proj = FastICA(n_components=7).fit_transform(X)
    preds1 = km.fit_predict(proj)
    end_time = time.time()

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y, preds1)
    sil = metrics.silhouette_score(X, preds1, metric='euclidean')

    print("ICA (n=7)")
    print(title + ":")
    print("homogeneity:" + "%0.2f" % homogeneity)
    print("completnes:" + "%0.2f" % completeness)
    print("v_measure:" + "%0.2f" % v_measure)
    print("sil:" + "%0.2f" % sil)
    print("time" + "%0.2f" % (end_time - start_time))

    start_time = time.time()
    km = KMeans(**km_dic)
    proj = GaussianRandomProjection(n_components=7).fit_transform(X)
    preds1 = km.fit_predict(proj)
    end_time = time.time()

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y, preds1)
    sil = metrics.silhouette_score(X, preds1, metric='euclidean')

    print("RP (n=7)")
    print(title + ":")
    print("homogeneity:" + "%0.2f" % homogeneity)
    print("completnes:" + "%0.2f" % completeness)
    print("v_measure:" + "%0.2f" % v_measure)
    print("sil:" + "%0.2f" % sil)
    print("time" + "%0.2f" % (end_time - start_time))

    start_time = time.time()
    km = KMeans(**km_dic)
    proj = TruncatedSVD(n_components=7).fit_transform(X)
    preds1 = km.fit_predict(proj)
    end_time = time.time()

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y, preds1)
    sil = metrics.silhouette_score(X, preds1, metric='euclidean')

    print("SVD (n=7)")
    print(title + ":")
    print("homogeneity:" + "%0.2f" % homogeneity)
    print("completnes:" + "%0.2f" % completeness)
    print("v_measure:" + "%0.2f" % v_measure)
    print("sil:" + "%0.2f" % sil)
    print("time" + "%0.2f" % (end_time - start_time))

    start_time = time.time()
    km = KMeans(**km_dic)
    preds1 = km.fit_predict(X)
    end_time = time.time()

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y, preds1)
    sil = metrics.silhouette_score(X, preds1, metric='euclidean')

    print("Original data")
    print(title + ":")
    print("homogeneity:" + "%0.2f" % homogeneity)
    print("completnes:" + "%0.2f" % completeness)
    print("v_measure:" + "%0.2f" % v_measure)
    print("sil:" + "%0.2f" % sil)
    print("time" + "%0.2f" % (end_time - start_time))

    print("EM ########################")
    start_time = time.time()
    km = GaussianMixture(**em_dic)
    proj = PCA(n_components=7).fit_transform(X)
    preds1 = km.fit_predict(proj)
    end_time = time.time()

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y, preds1)
    sil = metrics.silhouette_score(X, preds1, metric='euclidean')

    print("PCA (n=7)")
    print(title + ":")
    print("homogeneity:" + "%0.2f" % homogeneity)
    print("completnes:" + "%0.2f" % completeness)
    print("v_measure:" + "%0.2f" % v_measure)
    print("sil:" + "%0.2f" % sil)
    print("time" + "%0.2f" % (end_time - start_time))

    start_time = time.time()
    km = GaussianMixture(**em_dic)
    proj = FastICA(n_components=7).fit_transform(X)
    preds1 = km.fit_predict(proj)
    end_time = time.time()

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y, preds1)
    sil = metrics.silhouette_score(X, preds1, metric='euclidean')

    print("ICA (n=7)")
    print(title + ":")
    print("homogeneity:" + "%0.2f" % homogeneity)
    print("completnes:" + "%0.2f" % completeness)
    print("v_measure:" + "%0.2f" % v_measure)
    print("sil:" + "%0.2f" % sil)
    print("time" + "%0.2f" % (end_time - start_time))

    start_time = time.time()
    km = GaussianMixture(**em_dic)
    proj = GaussianRandomProjection(n_components=7).fit_transform(X)
    preds1 = km.fit_predict(proj)
    end_time = time.time()

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y, preds1)
    sil = metrics.silhouette_score(X, preds1, metric='euclidean')

    print("RP (n=7)")
    print(title + ":")
    print("homogeneity:" + "%0.2f" % homogeneity)
    print("completnes:" + "%0.2f" % completeness)
    print("v_measure:" + "%0.2f" % v_measure)
    print("sil:" + "%0.2f" % sil)
    print("time" + "%0.2f" % (end_time - start_time))

    start_time = time.time()
    km = GaussianMixture(**em_dic)
    proj = TruncatedSVD(n_components=7).fit_transform(X)
    preds1 = km.fit_predict(proj)
    end_time = time.time()

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y, preds1)
    sil = metrics.silhouette_score(X, preds1, metric='euclidean')

    print("SVD (n=7)")
    print(title + ":")
    print("homogeneity:" + "%0.2f" % homogeneity)
    print("completnes:" + "%0.2f" % completeness)
    print("v_measure:" + "%0.2f" % v_measure)
    print("sil:" + "%0.2f" % sil)
    print("time" + "%0.2f" % (end_time - start_time))

    start_time = time.time()
    km = GaussianMixture(**em_dic)
    preds1 = km.fit_predict(X)
    end_time = time.time()

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y, preds1)
    sil = metrics.silhouette_score(X, preds1, metric='euclidean')

    print("Original data")
    print(title + ":")
    print("homogeneity:" + "%0.2f" % homogeneity)
    print("completnes:" + "%0.2f" % completeness)
    print("v_measure:" + "%0.2f" % v_measure)
    print("sil:" + "%0.2f" % sil)
    print("time" + "%0.2f" % (end_time - start_time))


if __name__ == '__main__':
    CompareData1(features1, labels1, "Dataset1", km=7, em=7)
    CompareData1(features2, labels2, "Dataset2", km=10, em=10)










