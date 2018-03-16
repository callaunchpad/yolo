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

def get_obj_arr_type(frames, type_class):
    objects = []
    for frame in frames:
        objects.extend(frame.get_objects_of_type(type_class))
    return objects

def compile_joint_typeset(frames):
    typeset = set()
    for frame in frames:
        typeset.union(frame.class_dict.keys())
    return typeset

def k_means_type_split(frames):
    types = frames.compile_joint_typeset()
    final_objects = []
    for type_class in types:
        k = get_k_from_avg_type_filter(frames, type_class)
        objs = get_obj_arr_type(frames, type_class)
        final_objects.extend(k_means(objs, k))
    return final_objects

#need to determine of object type is the same
def k_means(objects, k):
    centroids = np.array([obj.get_centroid for obj in objects])
    kmeans_approx = Kmeans(n_clusters=k, random_state=0).fit_predict(centroids)
    grouped_object_list = [None for i in range(k)]
    for i in range(kmeans_approx.shape[0]):
        if grouped_object_list[i] is None:
            grouped_object_list[kmeans_approx[i]] = objects[i]
        else:
            grouped_object_list[kmeans_approx[i]].combine_objects(objects[i])
    return grouped_object_list
