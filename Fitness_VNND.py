import numpy as np
from scipy.spatial import distance
import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(url, names=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid',
                             'target'])
df = df.drop(['target'], axis=1)

points = df.values
matrix_dist = np.ndarray(shape=(df.shape[0], df.shape[0]), dtype=np.float)
#print(points)
for row in range(df.shape[0]):
    for col in range(df.shape[0]):
        matrix_dist[row][col] = distance.euclidean(points[row], points[col])
#print(matrix_dist)

def dmin(x_index_point, clus_index):
    # This function returns the minimum distance between two points
    # the distance is euclidean
    # x exists in y_array because they are from the same cluster

    #pasamos a lista los indices del cluster
    list_index_c = list(clus_index)
    del(list_index_c[list_index_c.index(x_index_point)])
    dist = []
    for ind in list_index_c:
        dist.append(matrix_dist[x_index_point][ind])
    return min(dist)

    # y_a_tmp = np.array(spam)
    # list_dist = []
    # for y_point in y_a_tmp:
    #     list_dist.append(distance.euclidean(x_point, y_point))
    # return min(list_dist)


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

# def VNND():
# pendent
np.random.seed(10)

def fitness(index_cluster):
    list_var_clus = []
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
