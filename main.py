import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import *

from utils import *
#from clustering import *
from neighbors import *

CWD_PATH = os.getcwd()

#NUMWORKERS = 2
FILENAME = 'videos/people.mp4'

IMAGE_WIDTH = 608
IMAGE_HEIGHT = 608
FRAME_SKIP = 40
FRAME_GAP = 1
INITIAL_BUFFER_SIZE = 12
BUFFER_SIZE = 5

'''
CONSTANTS THAT NEED TO BE FILLED OUT
'''
CLASS_NAMES = read_classes("YOLO_example/model_data/coco_classes.txt")
print("read classes")
ANCHORS = read_anchors("YOLO_example/model_data/yolo_anchors.txt")
print("read anchors")
YOLO_MODEL = load_model("YOLO_example/model_data/yolo.h5")
print("loaded model")
sess = K.get_session()

image_shape = (float(IMAGE_HEIGHT), float(IMAGE_WIDTH))

yolo_outputs = yolo_head(YOLO_MODEL.output, ANCHORS, len(CLASS_NAMES))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

def create_object_list(sess, image):
    objects_list = []
    image_data = np.array(image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={YOLO_MODEL.input: image_data, K.learning_phase(): 0})
    print('Found {} boxes '.format(len(out_boxes)))

    for i in range(len(out_scores)):
        print("Detected Object: " + str(out_classes[i]) + " " + str(out_boxes[i]) + " "+ str(out_scores[i]))
        new_obj = Object(out_classes[i], out_boxes[i], out_scores[i])
        objects_list.append(new_obj)

    return objects_list

def run_detection_on_buffer(images):
    print("Detecting for buffer")
    frames = [Frame(image, create_object_list(sess, image)) for image in images]
    objs_after_cluster = dbscan_type_split(frames)
    list_centroids(objs_after_cluster)

    return objs_after_cluster

#INIT global objects List
global OBJECTS_LIST
OBJECTS_LIST = []
buffer_size = INITIAL_BUFFER_SIZE
#TODO Fill out actual main items
if __name__ == '__main__':
    print("Running main")

    cap = cv2.VideoCapture(FILENAME)
    frame_num = 0
    image_buffer = []

    while(cap.isOpened()):
        print("Video Frame ", frame_num)
        ret, frame = cap.read()
        if frame is None:
            break
        if frame_num > FRAME_SKIP and frame_num % FRAME_GAP == 0:
            #add a frame to the current buffer
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image_buffer.append(frame_rgb)
        if len(image_buffer) == buffer_size:
            print("pushing buffer")
            #THIS IS WHERE WE DO STUFF WITH A FULL BUFFER
            clustered_objs = run_detection_on_buffer(image_buffer)
            if len(OBJECTS_LIST) == 0:
                OBJECTS_LIST = clustered_objs
                buffer_size = BUFFER_SIZE
            else:
                associate_with_regression(OBJECTS_LIST, clustered_objs)

            show_image(frame_rgb, OBJECTS_LIST)

            #EMPTY BUFFER
            image_buffer = []
        frame_num += 1

    cap.release()

def createTestData:
    x1 = []
    y1 = []

    x2 = []
    y2 = []

    for i in range(100):
        # first corner's line
        m1 = np.random.uniform(-20, 20)
        b1 = np.random.uniform(-20, 20)
        noise_x1 = np.random.normal()
        noise_y1 = np.random.normal()
        x1.add(np.random.uniform(0,100))
        y1.add(m1 * (x1[-1] + noise_x1) + (b1 + noise_y1))

        # second corner's line
        m2 = np.random.normal(m1, .5)
        b2 = np.np.random.uniform(-20, 20)
        height = abs(b2 - b1)
        noise_x2 = np.random.normal(0, 0.1 * height)
        noise_y2 = np.random.normal(0, 0.1 * height)
        x2.add(np.random.uniform(0,100))
        y2.add(m2 * (x2[-1] + noise_x2) + (b2 + noise_y2))


    print("x1 || " + str(x1))
    print("y1 || " + str(y1))

    print("x2 || " + str(x2))
    print("y2 || " + str(y2))
