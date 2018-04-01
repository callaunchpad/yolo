import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import math
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import queue

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util

from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# number of previous frames that we want to track
frames_to_track = 5
# list of lists - used to store n previous frames' objects
last_n_frames = [[] for x in range(frames_to_track)]
# current frame in the video
frame_count = 0
# ID # for number of objects seen
globalID = 0

#Download Model
#Commented out if already downloaded
'''
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
'''

#Load model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

'''
BEGIN CODE FOR OBJECT TRACKING
'''
##############################################################################
class Obj:
    id_count = 0
    threshold = 0.5

    def __init__(self, image, box, classification, score):
        self.im_height, self.im_width, dim = image.shape
        #self.centroid = getAbsoluteCentroid(image, box)
        self.classification = classification
        self.id = Obj.id_count
        Obj.id_count += 1
        self.score = score
        self.box = box
        self.frames_since_seen = 0

    def get_box_width(self):
        return (self.box[3] - self.box[1]) * self.im_width

    def get_box_height(self):
        return (self.box[2] - self.box[0]) * self.im_height

    def get_centroid(self):
        ymin, xmin, ymax, xmax = self.box
        xmin *= self.im_width
        xmax *= self.im_width
        ymin *= self.im_height
        ymax *= self.im_height
        return (xmin + (xmax - xmin)/2, ymin + (ymax - ymin)/2)

    def __str__(self):
        ret = "Detected Object ID: " + str(self.id)
        ret += " Type= " + str(self.classification) + " "
        cent = self.get_centroid()
        ret += " Location: x="+ str(cent[0]) + "  y=" + str(cent[1])
        return ret

    def distance(self, other):
        curr_cent = self.get_centroid()
        other_cent = other.get_centroid()
        return math.sqrt((curr_cent[0]-other_cent[0])**2 + (curr_cent[1]-other_cent[1])**2)

    def valid_movement_threshold(self, other):
        curr_area = self.get_box_width() * self.get_box_height()
        other_area = other.get_box_width() * other.get_box_height()
        ratio = curr_area / other_area
        if (ratio > Obj.threshold and ratio < 1 / Obj.threshold):
            return True
        else:
            return False

    def same_class(self, other):
        return self.classification == other.classification

    def update_properties(self, identical_obj):
        if identical_obj is None:
            return
        self.score = identical_obj.score
        self.box = identical_obj.box

    ######## magic comparison functions so that python's default PQ works ##########

    #less than
    def __lt__(self, other) :
        return self.frames_since_seen < other.frames_since_seen

    #less than or equal to
    def __le__(self, other) :
        return self.frames_since_seen <= other.frames_since_seen

    #equal to -- might want to change later to reflect boxes but IDK
    def __eq__(self, other) :
        return self.frames_since_seen == other.frames_since_seen

    def __gt__(self, other) :
        return self.frames_since_seen > other.frames_since_seen

    def __ge__(self, other) :
        return self.frames_since_seen >= other.frames_since_seen

    def __ne__(self, other) :
        return self.frames_since_seen != other.frames_since_seen

    def updatePriority(self) :
        self.frames_since_seen += 1

##############################################################################

def createObjectList(image, boxes, classes, scores):
    objlist = []
    for i in range(len(boxes)):
        if scores[i] > Obj.threshold:
            objlist.append(Obj(image, boxes[i], classes[i], scores[i]))
    return objlist

def printObjList(objList):
    print("Object List of length " + str(len(objList)))
    for obj in objList:
        print(obj)
        print(obj.get_box_width(), obj.get_box_height())

def printPQ(pq) :
    print("GLOBALPQQQQ")
    for i in range(pq.qsize()) :
        print(pq.get())

def update_obj_list(new_obj_list):
    global globalObjectsList
    for old_obj in globalObjectsList:
        closest_dist = float("inf")
        closest_obj = None
        for curr_obj in new_obj_list:
            dist = old_obj.distance(curr_obj)
            print("Obj Dist: ", dist)
            if (dist < closest_dist and old_obj.valid_movement_threshold(curr_obj) and old_obj.same_class(curr_obj)):
                closest_dist = dist
                closest_obj = curr_obj
        old_obj.update_properties(closest_obj)


def detect_objects(image_np, sess, detection_graph):
    global last_n_frames
    global frame_count
    global globalID
    n_frames = len(last_n_frames)
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)

    global globalObjectsList

    objList = createObjectList(image_np, boxes, classes, scores)
    globalObjectsList = objList #the objects detected in each frame
    # if len(globalObjectsList) == 0:
    #     globalObjectsList = objList
    # else:
    #     update_obj_list(objList)

    printObjList(globalObjectsList)

    globalObjectsPQ = queue.PriorityQueue(20)

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=12)


    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)

    # goes through every object in global list
    for objIndex in range(len(globalObjectsList)):
        marked = False
        obj = globalObjectsList[objIndex]
        cent = obj.get_centroid()
        # plt.text(cent[0], cent[1], "ID: " + str(obj.id), bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5})

        frame = 0   #index for frame

        #checks last n frames
        while (not marked and frame < n_frames): # marked = boolean for whether we found a match
            lastFrame = frame_count % n_frames - frame

            #checks every obj in the last n frames to see if there's a match
            for image in range(len(last_n_frames[lastFrame])):
                object = last_n_frames[lastFrame][image]
                # if there's a match:
                if abs(object[0][0] - cent[0]) < 50 and abs(object[0][1] - cent[1]) < 50:
                    plt.text(cent[0], cent[1], "ID: " + str(object[1]), bbox={'facecolor': 'blue', 'alpha': 0.5, 'pad': 5})
                    last_n_frames[lastFrame][image][0] = cent
                    updateID = globalObjectsList[objIndex]
                    updateID.id = object[1]
                    globalObjectsList[objIndex] = updateID
                    marked = True

                    # obj.frames_since_seen = 0
                    # globalObjectsPQ.put(obj)
                    break
            frame += 1
            obj.updatePriority

        if not marked:
            # obj.frames_since_seen = 0;
            # globalObjectsPQ.put(obj)
            updateID = globalObjectsList[objIndex]
            updateID.id = globalID
            globalObjectsList[objIndex] = updateID
            globalID += 1
            plt.text(cent[0], cent[1], "ID: " + str(updateID.id), bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5})
                    #plt.text(cent[0], cent[1], "ID: " + str(obj.id), bbox={'facecolor':'red', 'alpha':0.5, 'pad':5})

    last_n_frames[frame_count % n_frames] = []

    for obj in globalObjectsList:
        last_n_frames[frame_count % n_frames].append([obj.get_centroid(), obj.id])

    frame_count += 1
    plt.show()
    printPQ(globalObjectsPQ)
    return image_np, createObjectList(image_np, boxes, classes, scores)


#Declare global variables
global globalObjectsList
globalObjectsList = []

filename = 'videos/people.mp4'
print(filename)

cap = cv2.VideoCapture(filename)
IMAGE_SIZE = (12, 8)

frame_num = 0
frame_gap = 6

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    prev = None
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break;
        elif frame_num % frame_gap == 0:
            prev = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif prev is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #frame_rgb = cv2.addWeighted(prev, 0.5, frame_rgb, 0.5, 0)
            #frame_rgb = cv2.bilateralFilter(frame_rgb,9,75,75)
            image_np, objlist = detect_objects(frame_rgb, sess, detection_graph)
        else:
            continue
        frame_num += 1


cap.release()
cv2.destroyAllWindows()
