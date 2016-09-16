import json
import os
import tensorflow as tf
import glob
import compute_features as feat
from utils.data_loader import Loader, load_labels
from face_centrist_model import FaceCentrist
import matplotlib.pyplot as plt
from scipy.misc import imread
import numpy as np


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
                # load imagefiles and labels
                imagefiles, _ = load_labels(label_filename)
                break

        #otherwise, it's demo
        if label_filename is '':
            image_dir = os.path.join(data_dir,'images')
            imagefiles = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

        #Now, compute bounding boxes and save it into the file
        bb = feat.get_bounding_boxes(os.path.join(data_dir,'images'),imagefiles)
        json.dump(bb, open(bb_filename, 'w'))

    if not os.path.isfile(face_happiness_filename):

        #load bounding boxes, so faces from images can be extracted
        bb = json.load(open(bb_filename, 'r'))
        hscores = feat.get_face_happiness(os.path.join(data_dir,'images'), bb)
        #save individual face happiness to the file
        json.dump(hscores, open(face_happiness_filename, 'w'))

def infer_FaceCentrist(data_dir):

    # we use data loader lcass, as it computes face_features for the LSTMs
    data_loader = Loader(data_dir,"demo")

    # Launch the graph
    with tf.Session() as sess:

        model = FaceCentrist(3, 5)

        saver = tf.train.Saver(tf.trainable_variables())

        saver.restore(sess, './data/models/face-centrist.ckpt-47')
        # saver.save(sess, args.init_from+'_')

        # get data fot Face-CENTRIST model
        face_features, _, seqlength, centrist, imagefiles = data_loader.get_all_data()

        # get predictions
        predictions = model.infer(sess,face_features, seqlength, centrist)
        eval_dict = {}
        for image, label in zip(imagefiles, predictions):
            eval_dict[image] = [float(label)]

        json.dump(eval_dict, open('./output/face-centrist-predictions-demo.txt', 'w'))

        #plot results
        fig, ax = plt.subplots()
        plt.ion()

        for idx, imagefile in enumerate(imagefiles):
            try:
                imagedata = imread(os.path.join(data_dir,'images',imagefile))
                length, width, ch = np.shape(imagedata)
                ax.imshow(imagedata, extent=[-width / 2, width / 2, -length / 2, length / 2])
                ax.set_title('Predicted group happiness: ' + str(predictions[idx]))
                plt.show()
                plt.pause(3)
            except IOError:
                print('Can not open the image')



        #This portion od code outputs predictions in separate .txt files

        # test_data = json.load(open('./output/face-centrist-predictions-demo.txt', 'r'))
        # values = []
        #
        # sorted_keys = sorted(test_data)
        # if not os.path.exists('./output/face-centrist/'):
        #     os.makedirs('./output/face-centrist/')
        #
        # for key in sorted_keys:
        #     values.append(test_data[key][0])
        #     txt_file = os.path.splitext(os.path.basename(key))[0] + '.txt'
        #     with open('./output/face-centrist/' + txt_file, "w", encoding='utf-8') as infile:
        #         infile.write("{:.6f}".format(test_data[key][0]))


def infer_group_happiness(data_dir, model):


    if model == 'face-centrist':
        infer_FaceCentrist(data_dir)


def main(**kwargs):

    data_dir = './data/fake_HAPPEI/'
    # At the moment, we only provide Face_CENTRIST model
    model = 'face-centrist'

    preprocess_features(data_dir,'demo')
    #preprocess_features('./data/HAPPEI/HAPPEI_val.txt', 'val')
    infer_group_happiness(data_dir, model)


if __name__ == '__main__':

    main()