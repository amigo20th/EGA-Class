import numpy as np
from scipy.spatial import distance
import pandas as pd
from sklearn.model_selection import train_test_split

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(url, names=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid',
                             'target'])
df = df.drop(['target'], axis=1)


def dmin(x_point, y_array):
    # This function returns the minimum distance between two points
    # the distance is euclidean
    # x exists in y_array because they are from the same cluster
    spam = y_array.tolist()
    del(spam[spam.index(list(x_point))])
    y_a_tmp = np.array(spam)
    list_dist = []
    for y_point in y_a_tmp:
        list_dist.append(distance.euclidean(x_point, y_point))
    return min(list_dist)


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
    clus_0 = clus_0.drop(['target'], axis=1)
    clus_1 = clus_1.drop(['target'], axis=1)
    clus_2 = clus_2.drop(['target'], axis=1)
    clus_0_np = clus_0.values
    clus_1_np = clus_1.values
    clus_2_np = clus_2.values
    vnnd = variance_cluster(clus_0_np) + variance_cluster(clus_1_np) + variance_cluster(clus_2_np)
    return vnnd
