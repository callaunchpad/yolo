def predict(sess, image_file, graph):
    """
    Runs the graph stored in sess to predict boxes for image_file. Prints and plots the preditions.

    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.

    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes

    """
    objects_list = []
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    for i in len(out_scores):
        new_obj = Object(out_classes[i])
        new_obj.boxes = out_boxes[i]
        new_obj.scores = out_scores[i]
        objects_list.append(new_obj)
    return objects_list
