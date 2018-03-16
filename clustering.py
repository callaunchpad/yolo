# https://github.com/lars76/kmeans-anchor-boxes/blob/master/kmeans.py
import numpy as np
from utils import *
import math

def intersect(box1, box2):
    xA = max(box1.xmin, box2.xmin)
    yA = max(box1.ymin, box2.ymin)
    xB = min(box1.xmax, box2.xmax)
    yB = min(box1.ymax, box2.ymax)
    return BoundingBox(xA, yA, xB, yB, None)

# boxes is an array of BoundingBox objects
def iou(boxes):
    numboxes = len(boxes)
    basebox = boxes[0]
    netIntersectBox = None
    unionArea = basebox.get_area()
    for i in range(1,nunboxes):
        netIntersectBox = intersect(netIntersectBox, boxes[i])
        unionArea = unionArea + boxes[i].get_area() - netIntersectBox.get_area

    return netIntersectBox.get_area() / unionArea

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
