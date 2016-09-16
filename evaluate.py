import argparse
import json
import os
from itertools import chain
from datetime import datetime
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
import csv
import compute_features as feat
from utils.data_loader import Loader
from face_centrist_model import FaceCentrist
import glob
import utils.eval_metrics as evl
import numpy as np
from utils.data_loader import Loader, load_labels


def preprocess_features(data_dir,mode):


    feat_dir = './data/features'
    # generate filename for bounding boxes
    bb_filename = os.path.join(feat_dir,mode+'_bounding_boxes.json')
    # generate filename for face happiness file
    face_happiness_filename = os.path.join(feat_dir, mode + '_face_happiness.json')

    # if both files exist, return
    if os.path.isfile(bb_filename) and os.path.isfile(face_happiness_filename):
        return

    if not os.path.isfile(bb_filename):

        #find csv file (either training or validation)
        label_filename = ''
        for name in glob.glob(os.path.join(data_dir,'*.csv')):
            if mode in name:
                label_filename = name
                break

        #load imagefiles and labels
        imagefiles,_ = load_labels(label_filename)

        #Now, compute bounding boxes and save it into the file
        bb = feat.get_bounding_boxes(os.path.join(data_dir,'images'),imagefiles)
        json.dump(bb, open(bb_filename, 'w'))

    if not os.path.isfile(face_happiness_filename):

        #load bounding boxes, so faces from images can be extracted
        bb = json.load(open(bb_filename, 'r'))
        hscores = feat.get_face_happiness(os.path.join(data_dir,'images'), bb)
        #save individual face happiness to the file
        json.dump(hscores, open(face_happiness_filename, 'w'))




def evaluate(data_dir):

    data_loader_val = Loader(data_dir,'validation')

    # Launch the graph
    # Launch the graph
    with tf.Session() as sess:

        model = FaceCentrist(3,5)
        # Initializing the variables
        init = tf.initialize_all_variables()

        saver = tf.train.Saver(tf.all_variables())
        sess.run(init)
        # Restore the saved model
        saver.restore(sess, './data/models/face-centrist.ckpt-47')

        # Get all data
        test_x, test_label, test_seqlen, test_c , imagefiles = data_loader_val.get_all_data()

        # Feed it to the graph, fetch the error
        error, predictions = sess.run([model.rmse, model.pred], feed_dict={model.x: test_x, model.y: test_label,
                                                                           model.seqlen: test_seqlen, model.dropout_keep_prob: 1, model.c:test_c})

        # Evaluate the outputs
        evl.evaluate_linear_regression("./output/", predictions, test_label)
        predictions = list(chain.from_iterable(predictions))
        eval_dict = {}
        for image, pred, label in zip(imagefiles, predictions, test_label):
            eval_dict[image] = [float(pred), float(label)] # [predicted, label] values are store in the dictionary

        # Dump out the results
        json.dump(eval_dict, open('./output/face-cen-predictions.json', 'w'))

        # Compute results with RMSE higher than > 1
        bad_res = {}
        for image, pred, label in zip(imagefiles, predictions, test_label):
            if abs(float(pred) - label) > 1.0:
                bad_res[image] = [float(pred), float(label)]

        json.dump(bad_res, open('./output/face-cen-predictions_RMSE_higher_1.json', 'w'))




def main(**kwargs):

    data_dir = kwargs["data_dir"]

    #get current folder
    #cwd = os.getcwd()

    if not os.path.exists(data_dir):
        print('Please provide a proper path to data_dir')
        exit(0)

    preprocess_features(data_dir, 'validation')
    evaluate( data_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get a program and run it with input')
    parser.add_argument('--data_dir', type=str, default='', help='name of data dir file')
    args = parser.parse_args()
    main(**vars(args))