########################################################################################
#                                                                                      #
# Original code written by Davi Frossard, 2016                                         #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
# NOTE: Code rewritten by Aleksandra Cerekovic, 2016 for the purpose of training       #
# and testing with Tensorflow
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################



import numpy as np
import tensorflow as tf
import cv2
from itertools import chain

def to_rgb1a(im):
    # This should be fsater than 1, as we only
    # truncate to uint8 once (?)
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.float32)
    ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] =  im
    return ret

class VGG16():

    def __init__(self, image_size=224, num_channels=3,  num_classes=6, batch_size=1, seed=66478):

        #Store important features for the graph
        self.NUM_CLASSES = num_classes
        self.SEED = seed
        self.IMAGE_SIZE = image_size
        self.NUM_CHANNELS = num_channels
        self.BATCH_SIZE = batch_size

        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.
        with tf.name_scope('input'):
            self.x = tf.placeholder(
                tf.float32,
                shape=(batch_size, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS))
            self.y = tf.placeholder(tf.int64, shape=(batch_size,))
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")


        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(self.x, [-1, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS])
            tf.image_summary('input', image_shaped_input, 6)

        self.def_model()
        #self.load_weights('./data/vgg16_weights.npz')

        #propagate layers
        conv = self.convlayers(self.x)
        logits = self.fc_layers(conv)

        with tf.name_scope('loss'):
            with tf.name_scope('total'):
                self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.y))
                # L2 regularization for the fully connected parameters.
                # regularizers = 0
                # for wb in self.reg:
                #     regularizers += tf.nn.l2_loss(wb)
                # # Add the regularization term to the loss.
                # self.loss += 1e-3 * regularizers
            tf.scalar_summary('loss', self.loss)

        with tf.name_scope('train'):
            self.learning_rate = tf.Variable(1e-5)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                # Predictions (probabilities)
                self.predictions = tf.nn.softmax(logits)
                # Compute number of correct predictions
                correct_prediction = tf.equal(tf.argmax(self.predictions, 1), self.y)
            with tf.name_scope('accuracy'):
                self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('accuracy', self.acc)

    def def_model(self):

        self.parameters = {}
        self.fin = []

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                             stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                         trainable=True, name='biases')
            self.parameters['conv1_1_W'] = kernel
            self.parameters['conv1_1_b'] = biases

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                             stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                         trainable=True, name='biases')
            self.parameters['conv1_2_W'] = kernel
            self.parameters['conv1_2_b'] = biases

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                             stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                         trainable=True, name='biases')
            self.parameters['conv2_1_W'] = kernel
            self.parameters['conv2_1_b'] = biases

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                             stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                         trainable=True, name='biases')
            self.parameters['conv2_2_W'] = kernel
            self.parameters['conv2_2_b'] = biases

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                             stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                         trainable=True, name='biases')
            self.parameters['conv3_1_W'] = kernel
            self.parameters['conv3_1_b'] = biases

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                             stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                         trainable=True, name='biases')
            self.parameters['conv3_2_W'] = kernel
            self.parameters['conv3_2_b'] = biases

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                             stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                         trainable=True, name='biases')
            self.parameters['conv3_3_W'] = kernel
            self.parameters['conv3_3_b'] = biases

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                             stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                         trainable=True, name='biases')
            self.parameters['conv4_1_W'] = kernel
            self.parameters['conv4_1_b'] = biases

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                             stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                         trainable=True, name='biases')
            self.parameters['conv4_2_W'] = kernel
            self.parameters['conv4_2_b'] = biases

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                             stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                         trainable=True, name='biases')
            self.parameters['conv4_3_W'] = kernel
            self.parameters['conv4_3_b'] = biases

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                             stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                         trainable=True, name='biases')
            self.parameters['conv5_1_W'] = kernel
            self.parameters['conv5_1_b'] = biases

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                             stddev=1e-1), name='weights')

            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                         trainable=True, name='biases')
            self.parameters['conv5_2_W'] = kernel
            self.parameters['conv5_2_b'] = biases

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                             stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                         trainable=True, name='biases')
            self.parameters['conv5_3_W'] = kernel
            self.parameters['conv5_3_b'] = biases

        with tf.name_scope('fc1') and tf.device('/cpu:0') as scope:
            fc1w = tf.Variable(tf.truncated_normal([25088, 4096],
                                                        dtype=tf.float32,
                                                        stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                    trainable=True, name='biases')
            self.parameters['fc6_W'] = fc1w
            self.parameters['fc6_b'] = fc1b

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                        dtype=tf.float32,
                                                        stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                    trainable=True, name='biases')
            self.parameters['fc7_W'] = fc2w
            self.parameters['fc7_b'] = fc2b

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                        dtype=tf.float32,
                                                        stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                    trainable=True, name='biases')
            self.parameters['fc8_W'] = fc3w
            self.parameters['fc8_b'] = fc3b

        # fc_final
        with tf.name_scope('fc_final') as scope:
            fcwf = tf.get_variable("fc4_w",  # fully connected 4
                                        shape=[1000, self.NUM_CLASSES],
                                        initializer=tf.contrib.layers.xavier_initializer(seed=self.SEED))
            fcfb = tf.Variable(tf.constant(0., shape=[self.NUM_CLASSES]))
            self.parameters['fc9_W'] = fcwf
            self.parameters['fc9_b'] = fcfb

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            #print('Before...')
            #print(self.parameters[k].eval())
            sess.run(self.parameters[k].assign(weights[k]))
            #print('After...')
            #print(self.parameters[k].eval())


    def variable_summaries(self,var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.scalar_summary('sttdev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)

    def convlayers(self,input):

        # conv1_1
        conv = tf.nn.conv2d(input, self.parameters['conv1_1_W'], [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.parameters['conv1_1_b'])
        self.conv1_1 = tf.nn.relu(out)
        self.variable_summaries(self.conv1_1, 'conv1_1/out')

        # conv1_2
        conv = tf.nn.conv2d(self.conv1_1, self.parameters['conv1_2_W'], [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.parameters['conv1_2_b'])
        self.conv1_2 = tf.nn.relu(out)

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')
        self.variable_summaries(self.pool1, 'pool1/out')

        # conv2_1
        conv = tf.nn.conv2d(self.pool1, self.parameters['conv2_1_W'], [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.parameters['conv2_1_b'])
        self.conv2_1 = tf.nn.relu(out)

        # conv2_2
        conv = tf.nn.conv2d(self.conv2_1, self.parameters['conv2_2_W'], [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.parameters['conv2_2_b'])
        self.conv2_2 = tf.nn.relu(out)

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        conv = tf.nn.conv2d(self.pool2, self.parameters['conv3_1_W'], [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.parameters['conv3_1_b'])
        self.conv3_1 = tf.nn.relu(out)

        # conv3_2
        conv = tf.nn.conv2d(self.conv3_1, self.parameters['conv3_2_W'], [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.parameters['conv3_2_b'])
        self.conv3_2 = tf.nn.relu(out)

        # conv3_3
        conv = tf.nn.conv2d(self.conv3_2, self.parameters['conv3_3_W'], [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.parameters['conv3_3_b'])
        self.conv3_3 = tf.nn.relu(out)

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        conv = tf.nn.conv2d(self.pool3, self.parameters['conv4_1_W'], [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.parameters['conv4_1_b'])
        self.conv4_1 = tf.nn.relu(out)

        # conv4_2
        conv = tf.nn.conv2d(self.conv4_1, self.parameters['conv4_2_W'], [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.parameters['conv4_2_b'])
        self.conv4_2 = tf.nn.relu(out)

        # conv4_3
        conv = tf.nn.conv2d(self.conv4_2, self.parameters['conv4_3_W'], [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.parameters['conv4_3_b'])
        self.conv4_3 = tf.nn.relu(out)

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        conv = tf.nn.conv2d(self.pool4, self.parameters['conv5_1_W'], [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.parameters['conv5_1_b'])
        self.conv5_1 = tf.nn.relu(out)

        # conv5_2
        conv = tf.nn.conv2d(self.conv5_1, self.parameters['conv5_2_W'], [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.parameters['conv5_2_b'])
        self.conv5_2 = tf.nn.relu(out)

        # conv5_3
        conv = tf.nn.conv2d(self.conv5_2, self.parameters['conv5_3_W'], [1, 1, 1, 1], padding='SAME')
        out = tf.nn.bias_add(conv, self.parameters['conv5_3_b'])
        self.conv5_3 = tf.nn.relu(out)

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')
        return self.pool5

    def fc_layers(self,pool5):
        # fc1
        with tf.name_scope('fc1') and tf.device('/cpu:0') as scope:
            shape = int(np.prod(pool5.get_shape()[1:]))
            pool5_flat = tf.reshape(pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, self.parameters['fc6_W']), self.parameters['fc6_b'])
            self.fc1 = tf.nn.relu(fc1l)
            fc1 = tf.nn.dropout(self.fc1, self.dropout_keep_prob)
            #tf.histogram_summary('hidden1/pre_activations', pool5_flat)
            #tf.histogram_summary('hidden1/activations', self.fc1)
            #self.variable_summaries(self.parameters['fc6_W'], 'fc1/weights')
            #self.variable_summaries(self.parameters['fc6_b'], 'fc1/biases')

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2l = tf.nn.bias_add(tf.matmul(fc1, self.parameters['fc7_W']), self.parameters['fc7_b'])
            self.fc2 = tf.nn.relu(fc2l)
            fc2 = tf.nn.dropout(self.fc2, self.dropout_keep_prob)
            #tf.histogram_summary('hidden2/activations', self.fc2)

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3l = tf.nn.bias_add(tf.matmul(fc2, self.parameters['fc8_W']), self.parameters['fc8_b'])
            self.fc3 = tf.nn.relu(fc3l)
            fc3 = tf.nn.dropout(self.fc3, self.dropout_keep_prob)
            # fc_final
        with tf.name_scope('fc_final') as scope:
            self.fout = tf.nn.bias_add(tf.matmul(fc3, self.parameters['fc9_W']), self.parameters['fc9_b'])
        return self.fout

    def infer(self, sess, input_imagedata):
        PIXEL_DEPTH = 255.
        pred = []
        # resize the image, use opencv
        gray = cv2.cvtColor(input_imagedata, cv2.COLOR_BGR2GRAY)
        image_data_rs = cv2.resize(gray, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        # standardize the pixels
        # image_data = image_data_rs
        image_data = (image_data_rs -
                      PIXEL_DEPTH / 2.0) / PIXEL_DEPTH
        image_data = to_rgb1a(image_data)
        image_reshaped = image_data.reshape((-1, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS)).astype(np.float32)
        feed = {self.x: image_reshaped, self.dropout_keep_prob: 1}

        pred_ = sess.run([self.predictions], feed_dict=feed)
        for i in pred_:
            pred.append(i.tolist())
        return pred