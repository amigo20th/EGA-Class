import numpy as np
from scipy.spatial import distance

y_array = [(1,2), (-35, 98), (90, -7), (10, -80), (7, 6)]


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
    return dist_sum/len(cluster)

def variance_cluster(cluster):
    list_var = []
    p_dmin_c = prom_dmin_c(cluster)
    for elem_c in cluster:
        list_var.append((dmin(elem_c, cluster) - p_dmin_c)**2)
    print(list_var)
    return (1 / (len(cluster) - 1)) * sum(list_var)

def VNND()

print(variance_cluster(y_array))

