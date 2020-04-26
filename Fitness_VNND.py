import numpy as np
from scipy.spatial import distance

y_array = [(1,2), (7, 6), (-35, 98), (90, -7), (10, -80)]


def dist_min(x_point, y_array):
    # This function returns the minimum distance between two points
    # the distance is euclidean
    # x exists in y_array because they are from the same cluster
    y_a_list = list(y_array)
    y_a_list.remove(x_point)
    list_dist = map(lambda x, y: distance.euclidean(x, y), x_point, y_a_list)

    return list(list_dist)


print(dist_min((1, 2), y_array))


for i in y_array:
    print(distance.euclidean((1, 2), i))