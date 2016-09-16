import cv2
import numpy as np
import math

# a very unlucky implementation of image transformations ;/, sorry

def compute_padding_size(orig_image,size=(500,500)):

    cpimage = orig_image.copy()
    height, width, _ = np.shape(cpimage)
    landscape = True

    if width > height:
        newx, newy = size[1], round(height / (width / size[1]))  # new size (w,h)
        tfimage = cv2.resize(cpimage, (newx, newy), interpolation=cv2.INTER_CUBIC)
        # compute padding area
        padding_size = math.floor((size[0] - np.shape(tfimage)[0]) / 2)
    else:
        newx, newy = round(width / (height / size[0])), size[0]  # new size (w,h)
        tfimage = cv2.resize(cpimage, (newx, newy), interpolation=cv2.INTER_CUBIC)
        # compute padding area
        padding_size = math.floor((size[1] - np.shape(tfimage)[1]) / 2)
        landscape = False

    return padding_size, landscape, newx, newy


# resizes images and saves them, and resizes bounding boxes
def resize_image_and_bboxes(image_data, bounding_boxes, size=(500,500)):

    new_image = image_data.copy()
    new_bounding_boxes = []
    # if bounding_boxes:
    #     for (xmin, ymin, xmax, ymax) in bounding_boxes:
    #         image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    #
    # cv2.imshow('img', image)

    old_height,old_width,_ = np.shape(new_image)

    if old_width > old_height:
        newx, newy = size[1], round(old_height/ (old_width/size[1])) # new size (w,h)
        new_image = cv2.resize(new_image, (newx, newy),interpolation=cv2.INTER_CUBIC)
       #compute padding area
        padding_size = math.floor((size[0] - np.shape(new_image)[0])/2)
        new_image = cv2.copyMakeBorder( new_image, padding_size, padding_size, 0, 0, cv2.BORDER_CONSTANT,value=[0,0,0])
        new_height = np.shape(new_image)[0]
        if (new_height != size[0]):  # sometimes it can be 499
            new_padding_size = size[0] - new_height
            new_image = cv2.copyMakeBorder(new_image, new_padding_size, 0, 0, 0, cv2.BORDER_CONSTANT,
                                           value=[0, 0, 0])

        if bounding_boxes:

            for (xmin, ymin, xmax, ymax) in bounding_boxes:
                new_xmin = float(round(newx/ old_width * xmin))
                new_xmax = float(round(newx / old_width * xmax))
                new_ymin = float(round(newy / old_height * ymin + padding_size))
                new_ymax = float(round(newy / old_height * ymax + padding_size))
                new_bounding_boxes.append((new_xmin, new_ymin, new_xmax, new_ymax))


    else:
        newx, newy = round(old_width/ (old_height/size[0])), size[0] # new size (w,h)
        new_image = cv2.resize(new_image, (newx, newy),interpolation=cv2.INTER_CUBIC)
       #compute padding area
        padding_size = math.floor((size[1] - np.shape(new_image)[1])/2)
        new_image = cv2.copyMakeBorder( new_image, 0,0, padding_size, padding_size, cv2.BORDER_CONSTANT,value=[0,0,0])
        new_width = np.shape(new_image)[1]
        if (new_width != size[1]): #sometimes it can be 499, because of the floor up
            new_padding_size = size[1] - new_width
            new_image = cv2.copyMakeBorder(new_image, 0, 0, 0, new_padding_size, cv2.BORDER_CONSTANT,
                                           value=[0, 0, 0])

        if bounding_boxes:

            for (xmin, ymin, xmax, ymax) in bounding_boxes:
                new_xmin = float(round(newx / old_width * xmin + padding_size))
                new_xmax = float(round(newx / old_width * xmax + padding_size))
                new_ymin = float(round(newy / old_height * ymin))
                new_ymax = float(round(newy / old_height * ymax))
                new_bounding_boxes.append((new_xmin, new_ymin, new_xmax, new_ymax))

    # # Now we preview new image and adjusted bounding boxes
    # for (xmin, ymin, xmax, ymax) in new_bounding_boxes:
    #     cv2.rectangle(new_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    # cv2.imshow('img', new_image)
    #

    return new_image, new_bounding_boxes

def transform_bounding_boxes(old_size,old_bounding_boxes,new_size,new_image,padding=False):

    old_height, old_width = old_size
    new_height, new_width = new_size

    new_bounding_boxes = []
    landscape = True

    if padding:
        padding_size, landscape, oldx, oldy = compute_padding_size(new_image) #update _old values for padding

    else:
        padding_size = 0
        oldx = old_width
        oldy = old_height


    if landscape: #only important for padding
        for (xmin, ymin, xmax, ymax) in old_bounding_boxes:
            new_xmin = max(0,float(round(new_width/oldx * xmin )))
            new_xmax = min(float(round(new_width/oldx * xmax)),new_width)
            new_ymin = max(0,float(round((new_height/oldy *(ymin- padding_size)))))
            new_ymax = min(float(round((new_height/oldy * (ymax- padding_size)))),new_height)
            new_bounding_boxes.append((new_xmin, new_ymin, new_xmax, new_ymax))
    else:
        for (xmin, ymin, xmax, ymax) in old_bounding_boxes:
             new_xmin = max(0,float(round(new_width/oldx * (xmin - padding_size))))
             new_xmax = min(float(round(new_width/oldx * (xmax - padding_size))),new_height)
             new_ymin = max(0,float(round(new_height/oldy*ymin)))
             new_ymax = min(float(round(new_height/oldy*ymax)),new_height)
             new_bounding_boxes.append((new_xmin, new_ymin, new_xmax, new_ymax))

    return new_bounding_boxes



def main():

    image = cv2.imread("../_old/389679202_326a1884d4_183_33403234@N00.xml.jpg")

    bounding_boxes = [[302, 115, 401,231],
    [477, 124, 576,231],
    [626, 81, 709,197]]

    new_image, _ = resize_image_and_bboxes(image,bounding_boxes)
    cv2.imwrite("../_old/389679202_326a1884d4_183_33403234@N00_.xml.jpg",new_image)


if __name__ == "__main__":
    main()