# https://github.com/lars76/kmeans-anchor-boxes/blob/master/kmeans.py
import numpy as np
from utils import *
from sklearn.cluster import KMeans
import math

#returns average number of objects detected per frame for a given set of frames
#This is used later for the k in k-means clustering
def get_k_from_avg(frames):
    total_detection_count = 0
    for frame in frames:
        total_detection_count += frame.get_num_detections()

    return total_detection_count / len(frames)

def get_k_from_avg_type_filter(frames, given_class):
    total_detection_count = 0
    for frame in frames:
        total_detection_count += frame.get_num_detections_type_filter(given_class)

    return total_detection_count / len(frames)


#need to determine of object type is the same
def k_means(objects, k, dist=np.median):
    centroids = np.array([obj.get_centroid for obj in objects])
    kmeans_approx = Kmeans(n_clusters=k, random_state=0).fit_predict(centroids)
    grouped_object_list = []
    for i in range(k):
        grouped_object_list.append([])
    for i in range(kmeans_approx.shape[0]):
        grouped_object_list[kmeans_approx[i]].append(objects[i])
    combined_objects = []
    for group_list in grouped_object_list:
        for i in range(1, group_list):
            group_list[0].add_box(group_list[i].boxes[0], group_list[i].score)

    return [group_list[0] for group_list in combined_objects]
