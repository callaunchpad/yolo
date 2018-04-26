#from main import IMAGE_HEIGHT, IMAGE_WIDTH
from matplotlib import pyplot as plt
from PIL import Image
from visualization_utils import *
from yolo_utils import read_classes, generate_colors
import numpy as np
from functools import total_ordering
import math

###########
# CONSTANTS
##########
DETECTION_GAP_THRESHOLD = 3


class BoundingBox:
    '''
    Constructor for taking an RCNN box variable.
    Assumes image shape is a tuple of form (width, height)
    '''
    def __init__(self, box, classification):
        ymin, xmin, ymax, xmax = box
        self.ymin = ymin
        self.ymax = ymax
        self.xmin = xmin
        self.xmax = xmax
        self.classification = classification

    def get_width(self):
        return self.xmax - self.xmin

    def get_height(self):
        return self.ymax - self.ymin

    def get_centroid(self):
        return (self.xmin + (self.xmax - self.xmin)/2, self.ymin + (self.ymax - self.ymin)/2)

    def get_area(self):
        return self.get_width() * self.get_height()

    def get_as_array(self):
        return np.array([self.ymin, self.xmin, self.ymax, self.xmax])

    def get_top_left_point(self):
        return [self.xmin, self.ymin]

    #TODO: Fill in string method
    def __str__(self):
        return str(self.get_as_array())

class Object:
    id_count = 0
    types = {}

    def __init__(self, classification):
        self.id = id_count
        Object.id_count += 1
        self.classification = classification
        self.boxes = []
        self.scores = []
        self.prediction = None
        self.detection_gap = 0;

    #TAKES IN A BOX AND CLASSIFICATION
    def __init__(self, classification, rcnnbox, score):
        self.id = Object.id_count
        Object.id_count += 1
        self.classification = classification
        self.boxes = [BoundingBox(rcnnbox, classification)]
        self.scores = [score]
        self.score = score;
        self.prediction = None
        self.detection_gap = 0;


    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.id == other.id) and (len(self.boxes) == len(other.boxes))
        else:
            return False

    def __hash__(self):
        return hash((self.id, self.classification, self.score))

    #Takes in bounding box object and score
    def add_box(self, bounding_box, score):
        self.boxes.append(bounding_box)
        self.scores.append(score)

    def get_box(self, ind):
        return self.boxes[ind]

    def get_score(self, ind):
        return self.scores[ind]

    def combine_objects(self, other):
        self.boxes.extend(other.boxes)
        self.scores.extend(other.scores)
        self.detection_gap = 0

    def get_init_centroid(self):
        return  self.boxes[0].get_centroid()

    def form_data_array(self):
        ret_matrix = np.empty((4,))
        for box in self.boxes:
            boxarr = box.get_as_array()
            ret_matrix = np.vstack((ret_matrix, boxarr))
        ret_matrix = ret_matrix[1:, :]
        return ret_matrix.T

    def get_avg_bounding_box(self):
        data = self.form_data_array()
        ymin = data[0]
        xmin = data[1]
        ymax = data[2]
        xmax = data[3]

        box = [np.mean(ymin), np.mean(xmin), np.mean(ymax), np.mean(xmax)]
        return BoundingBox(box, self.classification)

    def predict_box(self, degree, buffer_size=1):
        data = self.form_data_array()

        ymin = data[0]
        xmin = data[1]
        ymax = data[2]
        xmax = data[3]

        minpred = np.polyfit(xmin, ymin, degree)
        minpoly = np.poly1d(minpred)
        maxpred = np.polyfit(xmax, ymax, degree)
        maxpoly = np.poly1d(maxpred)
        print("MIN POLY" + str(minpred))
        print("MAX POLY" + str(maxpred))

        xmindisp = get_avg_displacement(xmin) * math.ceil(buffer_size/2)
        xmaxdisp = get_avg_displacement(xmax) * math.ceil(buffer_size/2)
        print("XIMDISP: " + str(xmindisp))
        print("XMAXDISP: " + str(xmaxdisp))

        xmin_point = xmin[-1] + xmindisp
        xmax_point = xmax[-1] + xmaxdisp

        ymin_point = minpoly(xmin_point)
        ymax_point = maxpoly(xmax_point)

        plt.plot(xmin, ymin, '-', color='orange')
        plt.plot(xmax, ymax, '-', color='purple')
        plt.plot(xmin[0], ymin[0], 'o', color='green', ms=3)
        plt.plot(xmax[0], ymax[0], 'o', color='green', ms=3)

        box = [ymin_point, xmin_point, ymax_point, xmax_point]
        self.prediction = BoundingBox(box, self.classification)
        return self.prediction

    def update_det_gap(self):
        self.detection_gap += 1

    #TODO: Fill in string method
    def __str__(self):
        return ""

class Frame:

    #define detection threshold for Frame, use default for RCNN
    THRESHOLD = 0.5

    #should take in a frame and detect all of the objects
    #save instance variables of objects that have detection threshold of over THRESHOLD
    #Set NUM DETECTIONS
    def __init__(self, image, objects_list):
        self.image = image

        self.objects = objects_list

        self.class_dict = {}
        for obj in self.objects:
            if obj.classification not in self.class_dict:
                self.class_dict[obj.classification] = 1
            else:
                self.class_dict[obj.classification] = self.class_dict[obj.classification] + 1

        #draw_objects_on_image(image, self.objects)
        #print("I SHOULD SHOW AN IMAGE HERE YO")
        #plt.imshow(image)
        #plt.show()

    #returns number of detections over the threshold
    def get_num_detections(self):
        return len(self.objects)

    def get_objects_of_type(self, given_class):
        ret = []
        for obj in self.objects:
            if obj.classification == given_class:
                ret.append(obj)
        return ret

    def get_num_detections_type_filter(self, given_class):
        return self.class_dict.get(given_class, 0)

def list_centroids(objects):
    index = 1
    printstr = ""
    for obj in objects:
        for box in obj.boxes:
            printstr = printstr + " " + str(box.get_centroid())
        print("Object " + str(index) + " " + str(obj.classification) + ":  " + printstr)
        index += 1
    return printstr

def draw_objects_on_image(image, objects_list, ind=-1) :
    #out_scores = []
    #out_boxes = []
    #out_classes = []
    class_names = read_classes("YOLO_example/model_data/coco_classes.txt")
    colors = generate_colors(class_names)

    for obj in objects_list:
        box = obj.get_box(ind).get_as_array()

        pred = obj.prediction

        if pred is not None:
            predbox = pred.get_as_array()
            draw_bounding_box_on_image_array(image, predbox[0], predbox[1], predbox[2], predbox[3], use_normalized_coordinates=False, color='rgb(115, 200, 255)')

        draw_bounding_box_on_image_array(image, box[0], box[1], box[2], box[3], use_normalized_coordinates=False)
        #out_scores.append(obj.get_score(ind))
        #out_boxes.append(box)
        #out_classes.append(obj.classification)

def show_image(image, objects_list, ind=-1):
    draw_objects_on_image(image, objects_list, ind)
    plt.imshow(image)
    for obj in objects_list:
        pt = obj.get_box(ind).get_top_left_point()
        plt.text(pt[0], pt[1], "ID: " + str(obj.id), style='italic',
        bbox={'facecolor':'red', 'alpha':0.5, 'pad':2})
    plt.show()


@total_ordering
class Score:

    def __init__(self, global_object, cluster_object, buffer_size=1):
        self.global_object = global_object
        self.cluster_object = cluster_object
        self.iou_score = iou(global_object.prediction, cluster_object.get_avg_bounding_box())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.iou_score == other.iou_score
        else:
            return False

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not (self.iou_score == other.iou_score)
        else:
            return True

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.iou_score < other.iou_score
        else:
            return True


# runs predictions on global_objects
# maximize iou btwn predictions and average cluster
# add the best association to global_objects
def associate_with_regression(global_objects, objects_cluster, buffer_size=1):
    scores = []
    [obj.predict_box(1, buffer_size) for obj in global_objects]
    [obj.update_det_gap() for obj in global_objects]
    print(global_objects)
    for global_obj in global_objects:
        for cluster_obj in objects_cluster:
            scores.append(Score(global_obj, cluster_obj, buffer_size))

    scores.sort()
    seen_global = set()
    seen_cluster = set()

    while len(scores) > 0:
        curr = scores.pop()
        if (curr.global_object in seen_global) or (curr.cluster_object in seen_cluster):
            continue
        else:
            if curr.iou_score > 0.2:
                seen_global.add(curr.global_object)
                seen_cluster.add(curr.cluster_object)
                curr.global_object.combine_objects(curr.cluster_object)


    to_remove = [obj for obj in global_objects if obj.detection_gap >= DETECTION_GAP_THRESHOLD]

    for obj in to_remove:
        global_objects.remove(obj)

    global_objects.extend([obj for obj in objects_cluster if obj not in seen_cluster])
    #best_association = max(ious)
    #global_objects.append(best_association)


def iou(box1, box2):
    area1 = box1.get_area()
    area2 = box2.get_area()

    xi1 = max(box1.xmin, box2.xmin)
    yi1 = max(box1.ymin, box2.ymin)
    xi2 = min(box1.xmax, box2.xmax)
    yi2 = min(box1.ymax, box2.ymax)
    inter_area = (xi2 - xi1) * (yi2 - yi1)

    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area

    return iou

def get_avg_displacement(arr):
    return np.mean(np.diff(arr))
