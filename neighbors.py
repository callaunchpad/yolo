import numpy as np
from utils import *
import math
from sklearn.cluster import DBSCAN

EPS = 80

#UTILITY FUNCTIONS
def get_obj_arr_type(frames, type_class):
    objects = []
    for frame in frames:
        objects.extend(frame.get_objects_of_type(type_class))
    return objects

def compile_joint_typeset(frames):
    typeset = []
    for frame in frames:
        typeset.extend(frame.class_dict.keys())
    return list(set(typeset))

def dbscan_type_split(frames):
    types = compile_joint_typeset(frames)
    final_objects = []
    for type_class in types:
        objs = get_obj_arr_type(frames, type_class)
        dbscan_ret = dbscan_objs(objs)
        final_objects.extend(dbscan_ret)
    return final_objects

#need to determine of object type is the same
def dbscan_objs(objects):
    centroids = np.array([obj.get_init_centroid() for obj in objects])
    dbscan_approx = DBSCAN(eps=EPS).fit_predict(centroids)
    dbset = set(dbscan_approx)
    if -1 in dbset:
        dbscanlen = len(dbset) - 1
    else:
        dbscanlen = len(dbset)
    grouped_object_list = [None for i in range(dbscanlen)]
    for i in range(dbscan_approx.shape[0]):
        if dbscan_approx[i] != -1:
            if grouped_object_list[dbscan_approx[i]] is None:
                grouped_object_list[dbscan_approx[i]] = objects[i]
            else:
                grouped_object_list[dbscan_approx[i]].combine_objects(objects[i])
    return grouped_object_list
