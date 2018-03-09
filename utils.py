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
    def __init__(rcnnbox, im_shape, classification):
        ymin, xmin, ymax, xmax = box
        self.ymin = ymin * im_shape[1]
        self.ymax = ymax * im_shape[1]
        self.xmin = xmin * im_shape[0]
        self.xmax = xmax * im_shape[0]
        self.classification = classification

    def get_width(self):
        return self.xmax - self.xmin

    def get_height(self):
        return self.ymax - self.ymin

    def get_centroid(self):
        return (self.xmin + (self.xmax - self.xmin)/2, self.ymin + (self.ymax - self.ymin)/2)

    #TODO: Fill in string method
    def __str__(self):
        return ""

class Object:
    id_count = 0

    def __init__(self, classification):
        self.id = id_count
        Object.id_count += 1
        self.classification = classification
        self.boxes = []
        self.scores = []

    #Takes in bounding box object and score
    def add_box(self, bounding_box, score):
        self.boxes.append(bounding_box)
        self.scores.append(score)

    #TODO: Fill in string method
    def __str__(self):
        return ""

class Frame:

    def __init__(self, image):
        self.image = image
