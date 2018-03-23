from sklearn.cluster import KMeans

def n_clusters(frames):
    boxes = 0
    for frame in frames:
        boxes+=len(frames.boxes)
    return boxes/len(frames)

def kmeans(frames):
    bounding_boxes = []
    for frame in frames:
        bounding_boxes.append(frame.boxes)

    #TRYNA USE KMEANS TO FIND OBJECTS
    # Number of clusters
    kmeans = KMeans(n_clusters(frames))
    # Fitting the input data
    kmeans = kmeans.fit(bounding_boxes)
    # Getting the cluster labels
    labels = kmeans.predict(bounding_boxes)
    # Centroid values
    centroids = kmeans.cluster_centers_

    #create list of objects
    objects = []
    for centroid in centroids:
        objects.append()

    #return list of Objects with updated bounding boxes
    return objects

def iou(box1, box2):
  “”"Implement the intersection over union (IoU) between box1 and box2

  Arguments:
  box1 -- first box, list object with coordinates (x1, y1, x2, y2)
  box2 -- second box, list object with coordinates (x1, y1, x2, y2)
  “”"

  #Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
  xi1 = max(box1[0], box2[0])
  yi1 = max(box1[1], box2[1])
  xi2 = min(box1[2], box2[2])
  yi2 = min(box1[3], box2[3])
  inter_area = (xi2 - xi1) * (yi2 - yi1)

  #Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
  box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
  box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
  union_area = box1_area + box2_area - inter_area

  #Compute the IoU
  iou = inter_area / union_area

  return iou
