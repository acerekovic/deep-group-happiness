import argparse
import json
import os
from itertools import chain
from datetime import datetime
import tensorflow as tf
import compute_features as feat
from utils.data_loader import Loader, load_labels
from face_centrist_model import FaceCentrist
import time
import utils.eval_metrics as evl
import numpy as np
import glob

#to make results replicable
SEED = 10125


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


def grid_search(data_dir):

    #tunning parameters
    tuned_parameters ={'rnn_size': [2, 4, 5, 6, 8], 'num_layers': [1, 2, 3],
                         'batch_size': [50]}
    #max num of epochs for training
    num_epochs = 100
    grid_scores = {}
    eval_rate = 1

    for n_hidden in tuned_parameters['rnn_size']:


        for num_layers in tuned_parameters['num_layers']:


            for batch_size in tuned_parameters['batch_size']:

                best_val_rmse = 100000
                rel_epoch = -1

                data_loader = Loader(data_dir, 'training',batch_size, SEED)
                data_loader_val = Loader(data_dir, 'validation', batch_size, SEED)

                model = FaceCentrist(num_layers, n_hidden,SEED)

                init = tf.initialize_all_variables()

                # Launch the graph
                with tf.Session() as sess:

                    sess.run(init)

                    # Keep training until reach max iterations
                    for e in range(num_epochs):
                        data_loader.reset_batch_pointer()
                        for b in range(data_loader.num_batches):
                            batch_x, batch_y, batch_seqlen, batch_c, _ ,  = data_loader.next_batch()
                            # Run optimization op (backprop)
                            _, train_rmse = sess.run([model.train_op, model.rmse], feed_dict={model.x: batch_x, model.y: batch_y,
                                                                                              model.seqlen: batch_seqlen, model.dropout_keep_prob: 0.5, model.c: batch_c})
                        # Calculate val accuracy every epoch and save the best one
                        if e % eval_rate == 0:

                            train_x, train_y, train_seqlen, train_c, _  = data_loader.get_all_data()
                            train_rmse, summary, predictions, loss = sess.run([model.rmse, model.merged_summaries, model.pred, model.cost],
                                                                         feed_dict={model.x: train_x, model.y: train_y,
                                                                                    model.seqlen: train_seqlen,
                                                                                    model.dropout_keep_prob: 1, model.c: train_c})

                            test_data, test_label, test_seqlen, test_c, _  = data_loader_val.get_all_data()
                            val_predictions, val_rmse = sess.run([model.pred, model.rmse], feed_dict={model.x: test_data, model.y: test_label,
                                                                                                      model.seqlen: test_seqlen,
                                                                                                      model.dropout_keep_prob: 1,
                                                                                                      model.c: test_c})


                            train_rmse = float(train_rmse)
                            val_rmse = float(val_rmse)
                            #mean_rmse = evl.mean_RMSE_per_class(val_predictions, test_label)
                            if val_rmse > train_rmse and best_val_rmse > val_rmse:  ## train rmse should be lower than val rmse
                                best_val_rmse = val_rmse
                                rel_epoch = e

                tf.reset_default_graph() #reset
                key ='n_hidden: '+str(n_hidden) + ', num_layers:' + str(num_layers) + ', batch_size:' +str(batch_size)
                grid_scores[key] = [best_val_rmse, rel_epoch]
                print("Model, rnn_size= " + str(n_hidden) + ", num_layers= " + \
                  str(num_layers) + ", batch size= " + str(batch_size) +\
                  ", validation RMSE= " + "{:.5f}".format(best_val_rmse) + ", in epoch: " + str(rel_epoch))

    json.dump(grid_scores, open('./output/grid_search_results.json', 'w'))
    print('the best score is: %.4f', min(grid_scores.items(), key=lambda x: x[1]))

def train(data_dir):

    display_step = 2
    batch_size = 50
    num_epochs = 52
    data_loader = Loader(data_dir,'training',batch_size,SEED)
    data_loader_val = Loader(data_dir,'validation')

    # Launch the graph
    with tf.Session() as sess:

        #parameters of the model
        model = FaceCentrist(3,5,SEED)
        # Initializing the variables
        init = tf.initialize_all_variables()

        saver = tf.train.Saver(tf.all_variables())
        sess.run(init)
        save_dir = './train_dir/face_centrist_' + datetime.now().strftime('%Y_%m_%d_%H.%M')

        train_writer = tf.train.SummaryWriter(save_dir,
                                              sess.graph)

        # if args.init_from is not None:
        #     saver.restore(sess, args.init_from)

        train_scores = []
        test_scores = []


        # Keep training until reach max iterations
        for e in range(num_epochs):
            data_loader.reset_batch_pointer()
            for b in range(data_loader.num_batches):
                batch_x, batch_y, batch_seqlen, batch_c, _  = data_loader.next_batch()
                # Run optimization op (backprop)
                sess.run(model.train_op, feed_dict={model.x: batch_x, model.y: batch_y,
                                                    model.seqlen: batch_seqlen, model.dropout_keep_prob: 0.5, model.c: batch_c})
            if e % display_step == 0:
                # Calculate train accuracy
                train_x, train_y, train_seqlen, train_c, _ = data_loader.get_all_data()
                train_rmse, summary, predictions, loss = sess.run([model.rmse, model.merged_summaries, model.pred, model.cost],
                                                                  feed_dict={model.x: train_x, model.y: train_y,
                                                                             model.seqlen: train_seqlen,
                                                                             model.dropout_keep_prob: 1, model.c: train_c})

                test_data, test_label, test_seqlen, test_c, _  = data_loader_val.get_all_data()
                val_predictions = sess.run([model.pred], feed_dict={model.x: test_data, model.y : test_label,
                                                                    model.seqlen: test_seqlen,
                                                                    model.dropout_keep_prob: 1,
                                                                    model.c: test_c})

                val_predictions = list(chain.from_iterable(val_predictions))
                # test_label[test_label > 4] = 6
                # test_label[test_label < 1] = -1
                print ("Epoch " + str(e + 1) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training RMSE = " + \
                      "{:.5f}".format(train_rmse) + ", Validation RMSE: " + "{:.5f}".format(evl.RMSE(val_predictions,test_label)) )
                train_writer.add_summary(summary, e)
                saver.save(sess, save_dir + '/' + 'model.ckpt',
                           global_step=e + 1)


                #
                val = evl.RMSE(val_predictions, test_label)
                test_scores.append(val)
                tr = evl.RMSE(predictions, train_y)
                train_scores.append(tr)

        print("Optimization Finished, saving model!")
        saver.save(sess, save_dir + '/' + 'model.ckpt',
                   global_step=e + 1)


        evl.plot_learning_curves(np.array(train_scores), np.array(test_scores), display_step, save_dir,
                             title='learning_curves:rnn_size_' + str(8) + ':num_layers_' + str(
                                 3))



        test_data, test_label, test_seqlen, test_c, _  = data_loader_val.get_all_data()
        error, predictions = sess.run([model.rmse, model.pred], feed_dict={model.x: test_data, model.y: test_label,
                                                                           model.seqlen: test_seqlen, model.dropout_keep_prob: 1, model.c:test_c})
        evl.evaluate_linear_regression(save_dir,predictions,test_label)



def main(**kwargs):

    data_dir = kwargs["data_dir"]

    #get current folder
    #cwd = os.getcwd()

    if not os.path.exists(data_dir):
        print('Please provide a proper path to data_dir')
        exit(0)

    preprocess_features(data_dir,'training')
    preprocess_features(data_dir, 'validation')

    #grid_search(data_dir)
    train( data_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get a program and run it with input')
    parser.add_argument('--data_dir', type=str, default='', help='name of data dir file')
    args = parser.parse_args()
    main(**vars(args))