import json
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.misc import imread, imresize

from vgg16_model import VGG16
from face_detector import add_rectangles,  build_forward, init_model
from utils.tensorbox import googlenet_load
from utils.tensorbox.annolist import AnnotationLib as al
from utils.transform_images import transform_bounding_boxes, resize_image_and_bboxes


def unpackAnnoRect(rects):

    bounding_boxes = []
    for rect in rects:
        if rect.score > 0.4: #if confidence rate higher than 0.4
            # minx = rect.cx - int(rect.width / 2)
            # miny = rect.cy - int(rect.height / 2)
            # maxx = rect.cx + int(rect.width / 2)
            # maxy = rect.cy + int(rect.height / 2)
            bounding_boxes.append([rect.x1,rect.y1,rect.x2,rect.y2])

    return bounding_boxes

def get_bounding_boxes(image_dir,imagefiles):

    # Initialize face-detector
    iteration = 25000
    hypes_file = './data/models/face-detector-hypes.json'
    x_in, pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas, rezoom_features = init_model('./data/models/face-detector-hypes.json', 25000)

    # Open hyperparameter file
    with open(hypes_file, 'r') as f:
        H = json.load(f)

    saver = tf.train.Saver()

    # Now, initialize session where a network will predict bounding boxes, that we store in the dictionary predicted_bounding_boxes = {}

    predicted_bounding_boxes = {}

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        # Load stored model
        saver.restore(sess, './data/models/face-detector.ckpt-%d' % iteration)
        print('Detecting faces on images...')
        annolist = al.AnnoList()

        for idx, imagefile in enumerate(imagefiles):  # try last 5 images


            orig_image= imread(image_dir+ '/%s' % imagefile)
            #resize to 500x500
            image, _ = resize_image_and_bboxes(orig_image, [])

            # Resize image to the shape of the input (640x480)
            img = imresize(image, (H["arch"]["image_height"], H["arch"]["image_width"]), interp='cubic')
            feed = {x_in: img}
            (np_pred_boxes, np_pred_confidences, rz_feat) = sess.run([pred_boxes, pred_confidences, rezoom_features],
                                                                     feed_dict=feed)

            pred_anno = al.Annotation()
            #this part is for visualization :)
            new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                                           H["arch"], use_stitching=True, rnn_len=H['arch']['rnn_len'], min_conf=0.3)

            pred_anno.rects = rects
            annolist.append(pred_anno)

            old_bounding_boxes = unpackAnnoRect(rects)

            # Now, we apply two transformations of bounding boxes, first from 600*480 to 500*500, and then from 500*500 to orig_image size
            old_bounding_boxes1 = transform_bounding_boxes((H["arch"]["image_height"], H["arch"]["image_width"]),
                                                           old_bounding_boxes, (image.shape[0], image.shape[1]), image,
                                                           padding=False)
            new_bounding_boxes = transform_bounding_boxes((image[0], image[1]),
                                                          old_bounding_boxes1,
                                                          (orig_image.shape[0], orig_image.shape[1]),
                                                          orig_image, padding=True)

            predicted_bounding_boxes[imagefile] = new_bounding_boxes

    print('Detection done!')
    # Reset TF graph
    tf.reset_default_graph()
    return predicted_bounding_boxes


def get_face_happiness(image_dir, predicted_bounding_boxes):

    faceModel = VGG16()
    features = {}

    with tf.Session() as sess:
        # initialize Saver
        saver = tf.train.Saver(tf.trainable_variables())
        tf.initialize_all_variables().run()
        print('Computing happiness on detected faces')
        saver.restore(sess, './data/models/vgg16.ckpt-6601')
        for imagefile in predicted_bounding_boxes:

            imagedata = imread(image_dir+ '/%s' % imagefile)
            bounding_boxes = predicted_bounding_boxes[imagefile]
            happiness_vectors = []

            for idx, (xmin, ymin, xmax, ymax) in enumerate(bounding_boxes):
                face_data = imagedata[int(ymin):int(ymax),int(xmin):int(xmax)]
                prob = faceModel.infer(sess, face_data)
                happiness_vectors.append(prob)

            features[imagefile] = happiness_vectors

    print('Done!')
    # Reset TF graph
    tf.reset_default_graph()
    return features


