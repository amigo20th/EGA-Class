import numpy as np
from scipy.spatial import distance
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(url, names=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid',
                             'target'])
target = df.target
df = df.drop(['target'], axis=1)

points = df.values
matrix_dist = np.ndarray(shape=(df.shape[0], df.shape[0]), dtype=np.float)
for row in range(df.shape[0]):
    for col in range(df.shape[0]):
        matrix_dist[row][col] = distance.euclidean(points[row], points[col])


def dmin(x_index_point, clus_index):
    # This function returns the minimum distance between two points
    # the distance is euclidean
    # x exists in y_array because they are from the same cluster
    list_index_c = list(clus_index)
    del(list_index_c[list_index_c.index(x_index_point)])
    dist = []
    for ind in list_index_c:
        dist.append(matrix_dist[x_index_point][ind])
    return min(dist)



def prom_dmin_c(cluster):
    list_dist = []
    for elem_c in cluster:
        list_dist.append(dmin(elem_c, cluster))
    dist_sum = sum(list_dist)
    return dist_sum / len(cluster)

def variance_cluster(cluster):
    list_var = []
    p_dmin_c = prom_dmin_c(cluster)
    for elem_c in cluster:
        list_var.append((dmin(elem_c, cluster) - p_dmin_c) ** 2)
    return (1 / (len(cluster) - 1)) * sum(list_var)

def fitness(index_cluster):
    target_str = list(index_cluster)
    target = [int(x) for x in target_str]
    df_tmp = df
    df_tmp['target'] = target
    clus_0 = pd.DataFrame(df_tmp.loc[df_tmp['target'] == 0])
    clus_1 = pd.DataFrame(df_tmp.loc[df_tmp['target'] == 1])
    clus_2 = pd.DataFrame(df_tmp.loc[df_tmp['target'] == 2])
    clus_0_np = list(clus_0.index)
    clus_1_np = list(clus_1.index)
    clus_2_np = list(clus_2.index)
    vnnd = variance_cluster(clus_0_np) + variance_cluster(clus_1_np) + variance_cluster(clus_2_np)
    return vnnd

def plot_cluster(index_cluster):
    target_str = list(index_cluster)
    target = [int(x) for x in target_str]
    df['target'] = target
    sns.pairplot(df, hue='target')
    plt.show()

def populationInitial(n_class):
    kmeans = KMeans(n_clusters=n_class, max_iter=300).fit(df)
    return kmeans.labels_

