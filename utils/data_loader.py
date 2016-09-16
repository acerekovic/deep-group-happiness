import json
import math
import os
import random
from itertools import chain
import scipy.io as sio
import numpy as np
from scipy.misc import imread
import csv
import glob

# def compute_face_size(xmin, ymin, xmax, ymax):
#     return math.sqrt(abs(math.pow(ymax - ymin,2) + math.pow(xmax - xmin,2)))

def load_labels(csv_file):
    with open(csv_file, "rt", encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter=',')
        next(reader, None)  # skip the header
        imagefiles = []
        labels = []
        for row in reader:
            imagefiles.append(row[0])
            labels.append(int(row[1]))

    return imagefiles, labels


def load_CENTRIST(centrist_dir, imagefiles):

    # load centrist
    centrist_dict = {}

    for image in imagefiles:
        matfilename = os.path.splitext(os.path.basename(image))[0]
        CENTRIST_mat = os.path.join(centrist_dir + '/' + str(image) + '.mat')
        try:
            CENTRIST_content = sio.loadmat(CENTRIST_mat)
        except IOError as e:
            print('Could not read:', CENTRIST_mat, ':', e, '- it\'s ok, skipping.')
        else:
            try:
                CENTRIST_struct = CENTRIST_content['var']
            except KeyError:  # a dirty trick to overcome different structures of mat file
                CENTRIST_struct = CENTRIST_content['data']
            centrist_dict[image] = CENTRIST_struct

    return centrist_dict


def compute_center(xmin, ymin, xmax, ymax):
    return (xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2)


def compute_size(xmin, ymin, xmax, ymax):
    return (xmax - xmin, ymax - ymin)

class Loader():

    def __init__(self, data_dir, mode, batch_size = 0, SEED=10034):

        self.max_seq_len = 25
        self.vector_size = 10
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.mode = mode
        self.seed = SEED

        self.feature_file = os.path.join("./data/features/"+self.mode+"_face_features.json")

        if not os.path.exists(self.feature_file):
            print("Creating features...")
            self.preprocess()

        print("Loading features")
        self.load_preprocessed()
        if self.batch_size != 0:
            self.create_batches()


    def preprocess(self):

        #define where are bounding boxes and face happiness intensities
        bb_json = './data/features/'+self.mode+'_bounding_boxes.json'
        face_json = './data/features/'+self.mode+'_face_happiness.json'

        face_happiness = json.load(open(face_json, 'r'))
        bbs = json.load(open(bb_json, 'r'))

        face_features = {}
        for image, faces in face_happiness.items():
            #compute features to the model
            face_features[image] = self._compute_features(face_happiness[image], bbs[image], image)

        filename = './data/features/'+self.mode+'_face_features.json'
        json.dump(face_features, open(filename, 'w'))

    def _compute_features(self,face_happiness, bounding_boxes, imagefile):

        imagedata = imread(os.path.join(self.data_dir,'images')+'/%s' % imagefile)
        features = []
        height, width, _ = np.shape(imagedata)


        for idx, _ in enumerate(bounding_boxes):
            features.append([])

        for idx, (xmin, ymin, xmax, ymax) in enumerate(bounding_boxes):
            #compute center of face (bounding box)
            xcen, ycen = compute_center(xmin, ymin, xmax, ymax)
            #normalize it
            features[idx].append([xcen / width, ycen / height])
            #compute size of the bounding box
            fwidth, fheight = compute_size(xmin, ymin, xmax, ymax)
            #normalize it
            features[idx].append([fwidth / width, fheight / height])

        for idx, face_vector in enumerate(face_happiness):
            #unpack probabilities from face happiness prediction
            normal_list = list(chain.from_iterable(face_vector))
            normal_list = list(chain.from_iterable(normal_list))
            #append it to feature vector
            features[idx].append(normal_list)

        return features



    def load_preprocessed(self):

        imagefiles = []
        # load features from feature file
        with open(self.feature_file, 'r') as f:
            features_dict = json.load(f)

        # find csv file (either training or validation)
        label_filename = ''
        for name in glob.glob(os.path.join(self.data_dir, '*.csv')):
            if self.mode in name:
                label_filename = name
                # load imagefilenames and labels
                imagefiles, labels = load_labels(label_filename)
                label_dict = dict(zip(imagefiles, labels))
                break

        # otherwise, it's demo
        if label_filename is '':
            image_dir = os.path.join(self.data_dir, 'images')
            imagefiles = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

        # load CENTRIST
        centrist_dir = os.path.join(self.data_dir, 'CENTRIST')
        centrist_dict = load_CENTRIST(centrist_dir,imagefiles)

        input_features = []
        input_size = []
        input_centrist = []
        input_imagefiles = []
        input_labels = []
        # iterate in seeded random order
        random.seed(self.seed)
        imagefiles = list(features_dict.keys())
        random.shuffle(imagefiles)

        for image in imagefiles:
            features = features_dict[image]
            input_features.append(list(chain.from_iterable(chain.from_iterable(features))))  # unpack
            input_size.append(len(list(chain.from_iterable(chain.from_iterable(features))))) # size
            if 'label_dict' in locals(): #if there are labels for files
                input_labels.append(label_dict[image]) #get label
            input_centrist.append(centrist_dict[image]) #get centrist
            input_imagefiles.append(image) #get imagefile

        #store computed values
        self.input_features = input_features
        self.input_labels = input_labels
        self.input_size = len(self.input_features)
        self.input_centrist = input_centrist
        self.input_imagefiles = input_imagefiles

        self.seq_length = np.ndarray((len(self.input_features), 1), dtype=np.int32)

        #compute sequence length
        for idx, feature_vector in enumerate(self.input_features):
            self.seq_length[idx] = len(feature_vector) / self.vector_size

    def create_batches(self):

        # Compute number of batches.
        # Note that this is not perfect - if the batch_size is not a divisor of an input_size, some images will be discarded in this epoch
        self.num_batches = int(self.input_size / (self.batch_size))
        self.pointer = 0

        # When the data (tesor) is too small, let's give them a better error message
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."
        vector_size = self.vector_size

        self.tensor = np.zeros((self.num_batches * self.batch_size, self.max_seq_len, vector_size)) #is the maximum seq len

        for input_seq in range(0,self.num_batches * self.batch_size):
            vectors = []
            #for offset in range(0,int(len(self.input_features[input_seq])/vector_size)):
            for offset in range(0, self.seq_length[input_seq]):
                vectors.append(self.input_features[input_seq][offset*vector_size:offset*vector_size+vector_size])
            try:
                self.tensor[input_seq,:len(vectors),:] = np.array(vectors)
            except IndexError and ValueError:
                print('Index is out of bonds')

        xdata = self.tensor

        seq_len = np.array(self.seq_length[:self.num_batches * self.batch_size])
        centrist = np.array(self.input_centrist[:self.num_batches * self.batch_size])
        imagefiles = np.array(self.input_imagefiles[:self.num_batches * self.batch_size])

        self.x_batches = np.split(xdata, self.num_batches)

        self.seq_len_batches = np.split(seq_len, self.num_batches)
        self.centrist_batches = np.split(centrist, self.num_batches)
        self.imagefiles_batches = np.split(imagefiles, self.num_batches)

        # add labels, if exist
        if len(self.input_labels) != 0:
            ydata = np.array(self.input_labels[:self.num_batches * self.batch_size])
            self.y_batches = np.split(ydata, self.num_batches)
        else: #it's zero
            self.y_batches = np.split(np.zeros(self.num_batches * self.batch_size),self.num_batches)

    def next_batch(self):

        if (self.pointer == self.num_batches):
            self.reset_batch_pointer()
        x, y, z, c = self.x_batches[self.pointer], self.y_batches[self.pointer].reshape(self.batch_size,), self.seq_len_batches[self.pointer].reshape(self.batch_size,), self.centrist_batches[self.pointer].reshape(self.batch_size,4064,)
        im = self.imagefiles_batches[self.pointer]

        self.pointer += 1
        return x, y, z, c, im

    def reset_batch_pointer(self):
        #re-shuffle data and compute new batches
        self.load_preprocessed()
        self.create_batches()


    def get_all_data(self):

        vector_size = self.vector_size
        self.tensor = np.zeros((self.input_size, self.max_seq_len, vector_size)) #is the maximum seq len

        for idx, face_vector in enumerate(self.input_features):
            vectors = []
            for offset in range(0,self.seq_length[idx]):
                vectors.append(face_vector[offset*vector_size:offset*vector_size+vector_size])
            try:
                self.tensor[idx,:len(vectors),:] = np.array(vectors)
            except IndexError and ValueError:
                print('Index is out of bonds')


        xdata = self.tensor
        if self.input_labels != 0:
            ydata = np.array(self.input_labels[:self.input_size]).reshape(None,)
        else:
            ydata = np.zeros(self.num_batches * self.batch_size).reshape(None,)

        seq_len = np.array(self.seq_length[:self.input_size]).reshape(self.input_size,)
        centrist = np.array(self.input_centrist[:self.input_size]).reshape(self.input_size,4064,)
        imagefiles = np.array(self.input_imagefiles[:self.input_size])

        return xdata, ydata, seq_len, centrist, imagefiles





