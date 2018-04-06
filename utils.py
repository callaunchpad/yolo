from main import IMAGE_HEIGHT, IMAGE_WIDTH
from matplotlib import pyplot as plt
from PIL import Image
from visualization_utils import *

class BoundingBox:

    def __init__(xmin, ymin, xmax, ymax, classification):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.classification = classification

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
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    #TODO: Fill in string method
    def __str__(self):
        return ""

class Object:
    id_count = 0
    types = {}

    def __init__(self, classification):
        self.id = id_count
        Object.id_count += 1
        self.classification = classification
        self.boxes = []
        self.scores = []

    #TAKES IN A BOX AND CLASSIFICATION
    def __init__(self, classification, rcnnbox, score):
        self.id = Object.id_count
        Object.id_count += 1
        self.classification = classification
        self.boxes = [BoundingBox(rcnnbox, classification)]
        self.scores = [score]
        self.score = score;

    #Takes in bounding box object and score
    def add_box(self, bounding_box, score):
        self.boxes.append(bounding_box)
        self.scores.append(score)

    def get_last_box(self):
        return self.boxes[-1]

    def get_last_score(self):
        return self.scores[-1]

    def combine_objects(self, other):
        self.boxes.extend(other.boxes)
        self.scores.extend(other.scores)

    def get_init_centroid(self):
        return  self.boxes[0].get_centroid()

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

        drawObjects(image, self.objects)
        print("I SHOULD SHOW AN IMAGE HERE YO")
        plt.imshow(image)
        plt.show()

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
    for obj in objects:
        printstr = ""
        for box in obj.boxes:
            printstr = printstr + " " + str(box.get_centroid())
        print("Object " + str(index) + " " + str(obj.classification) + ":  " + printstr)
        index += 1
    return printstr

from YOLO_example import yolo_utils
from yolo_utils import draw_boxes, generate_colors, read_classes

def drawObjects(image, objects_list) :
    out_scores = []
    out_boxes = []
    out_classes = []
    class_names = read_classes("YOLO_example/model_data/coco_classes.txt")
    colors = generate_colors(class_names)

    for obj in objects_list :
        box = obj.get_last_box().get_as_array()
        draw_bounding_box_on_image_array(image, box[1], box[0], box[3], box[2], use_normalized_coordinates=False)
        out_scores.append(obj.get_last_score())
        out_boxes.append(box)
        out_classes.append(obj.classification)

    #draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
