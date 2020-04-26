import numpy as np
from scipy.spatial import distance
import pandas as pd
from sklearn.model_selection import train_test_split

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(url, names=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid',
                             'target'])
df.loc[df['target'] == 'Iris-setosa'] = 0
df.loc[df['target'] == 'Iris-versicolor'] = 1
df.loc[df['target'] == 'Iris-virginica'] = 2
target = df.target
df = df.drop(['target'], axis=1)
print(df)
print(target)


def dmin(x_point, y_array):
    # This function returns the minimum distance between two points
    # the distance is euclidean
    # x exists in y_array because they are from the same cluster
    y_a_list = list(y_array)
    y_a_list.remove(x_point)
    list_dist = []
    for y_point in y_a_list:
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


fitness_VNND(index_cluster)_
