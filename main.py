import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import *
from darkflow.net.build import TFNet
import re

from utils import *
#from clustering import *
from neighbors import *

CWD_PATH = os.getcwd()

#NUMWORKERS = 2
FILENAME = 'videos/people.mp4'
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
ap.add_argument("-g", "--gpu", type=float, default=0.0,
	help="gpu usage")
args = vars(ap.parse_args())


IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
FRAME_SKIP = 0
FRAME_GAP = 1
INITIAL_BUFFER_SIZE = 12
BUFFER_SIZE = 5

'''
CONSTANTS THAT NEED TO BE FILLED OUT
'''
#CLASS_NAMES = read_classes("YOLO_example/model_data/coco_classes.txt")
#print("read classes")
#ANCHORS = read_anchors("YOLO_example/model_data/yolo_anchors.txt")
#print("read anchors")
#YOLO_MODEL = load_model("YOLO_example/model_data/yolo.h5")

image_shape = (float(IMAGE_HEIGHT), float(IMAGE_WIDTH))

#yolo_outputs = yolo_head(YOLO_MODEL.output, ANCHORS, len(CLASS_NAMES))
#scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

def create_object_list(image):
	objects_list = []
	#image_data = np.array(image, dtype='float32')
	#image_data /= 255.
	#image_data = np.expand_dims(image_data, 0)

	result = tfnet.return_predict(image)
	# print(result)
	# box = [result[3][1], result[3][0], result[4][1], result[4][0]]
	# out_scores = result[1]
	# out_classes = result[0]
	# out_boxes = box

	#print('Found {} boxes '.format(len(out_boxes)))

	for i in range(len(result)):
		#print("Detected Object: " + str(out_classes[i]) + " " + str(out_boxes[i]) + " "+ str(out_scores[i]))
		box = [result[i].get("topleft").get("y"), result[i].get("topleft").get("x"), result[i].get("bottomright").get("y"), result[i].get("bottomright").get("x")]
		new_obj = Object(result[i].get("label"), box, result[i].get("confidence"))
		objects_list.append(new_obj)

	return objects_list

def run_detection_on_buffer(images):
	print("Detecting for buffer")
	frames = [Frame(image, create_object_list(image)) for image in images]
	objs_after_cluster = dbscan_type_split(frames)
	#list_centroids(objs_after_cluster)
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

	out_name = re.split('/|[.]', FILENAME)[-2] # last item is the extention, second to last is file name

	out = cv2.VideoWriter('out-' + out_name + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 15, (IMAGE_WIDTH,IMAGE_HEIGHT))
	options = {"model": "cfg/yolo.cfg", "load": "yolo.weights", "threshold": 0.5}
	if args.get("video", False):
		options.update({"gpu": args["gpu"]})
	tfnet = TFNet(options)

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
				#BUFFER SIZE IS LAST PARAM SET TO 1
				associate_with_regression(OBJECTS_LIST, clustered_objs, buffer_size)

			show_image(frame_rgb, OBJECTS_LIST)
			# cv2.imwrite('file.jpg', frame_rgb)
			out.write(frame_rgb)

			#EMPTY BUFFER
			image_buffer = []
		frame_num += 1

	cap.release()
	out.release()
