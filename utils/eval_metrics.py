import math
import statistics as stats

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def prob_to_float(predictions):
    classes = np.array([0, 1, 2, 3, 4, 5])
    new_predictions = []

    for idx, row in enumerate(predictions):
        sum = np.sum(row * classes)
        new_predictions.append(sum)
    return new_predictions

def prob_to_float_NN(predictions):

    glob_max = np.argmax(predictions, 1)
    new_predictions = []

    for idx, _ in enumerate(predictions):
        acc= 0
        sum = 0
        for offset in range(-1, 2):
            if 0 <= (glob_max[idx]+offset) <= 5:
                sum = sum + (glob_max[idx]+offset)*predictions[idx][(glob_max[idx]+offset)]
                acc = acc + predictions[idx][(glob_max[idx]+offset)]
        new_predictions.append(sum/acc)

    return new_predictions

def prob_to_class_argmax(predictions):

    return np.argmax(predictions,1)


def RMSE(predictions, labels):

    return math.sqrt(mean_squared_error(labels, np.array(predictions)))

def RMSE_per_class(predictions, labels):

    grouped_pre = []
    rmse_per_class = []

    for i in range(0,6):
        grouped_pre.append([])

    for label,prediction in zip(labels,predictions):
        grouped_pre[label].append(float(prediction))

    for idx,group_p in enumerate(grouped_pre):
        if len(group_p) > 0:
            label = np.full((len(group_p)), idx, dtype=np.int64)
            rmse_per_class.append(math.sqrt(mean_squared_error(label,np.array(group_p))))
        else:
            rmse_per_class.append(0.) #if there is a label of this class

    return rmse_per_class

def mean_RMSE_per_class(predictions, test_label):

    rmse_per_class = RMSE_per_class(predictions, test_label)
    mean_rmse = 0


    for idx, value in enumerate(rmse_per_class):
        #print('Label ' + str(idx) + ', rmse: ' + str(value))
        mean_rmse = mean_rmse + value

    mean_rmse = mean_rmse / 6

    return mean_rmse


def mean_std_RMSE_per_class(predictions, test_label):

    rmse_per_class = RMSE_per_class(predictions, test_label)
    # mean_rmse = 0
    #
    #
    # for idx, value in enumerate(rmse_per_class):
    #     #print('Label ' + str(idx) + ', rmse: ' + str(value))
    #     mean_rmse = mean_rmse + value
    #
    # mean_rmse = mean_rmse / 6
    mean = stats.mean(rmse_per_class)
    std = stats.stdev(rmse_per_class)

    return  mean, std

def plot_learning_curves(train_scores,test_scores,display_step,train_dir,title='learning_curves'):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)

    plt.xlabel("Num epochs")
    plt.ylabel("Score")
    train_sizes = train_scores.shape[0]
    x = [] # x-axis range
    x.append(display_step)
    for i in range(1,int(train_sizes)):
        x.append(x[i-1] + display_step) #batch size


    plt.grid()
    #
    # plt.fill_between(x, train_scores_mean - train_scores_std,
    #                  train_scores_mean + train_scores_std, alpha=0.1,
    #                  color="r")
    # plt.fill_between(x, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(x, train_scores, 'o-', color="r",
             label="Training score")
    plt.plot(x, test_scores, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(train_dir + '/'+ title + '.png', bbox_inches='tight')
    return plt

def evaluate_linear_regression(train_dir, predictions, test_label, title='', plot=True):


    # Calculate accuracy
    # test_data = testset.data
    # test_label = testset.labels
    # test_seqlen = testset.seqlen
    # print "Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={x: test_data, y: test_label,
    #                                   seqlen: test_seqlen})
    print(title + " rmse:", \
          "{:.5f}".format(RMSE(predictions, test_label)))

    print(title + "Mean rmse per class:", \
          "{:.5f}".format(mean_RMSE_per_class(predictions, test_label)))

    # print('RMSE per class is')
    # rmse_per_class = RMSE_per_class(predictions, test_label)
    # mean_rmse = 0
    #
    # for idx, value in enumerate(rmse_per_class):
    #     print('Label ' + str(idx) + ', rmse: ' + str(value))
    #     mean_rmse = mean_rmse + value
    #
    # mean_rmse = mean_rmse / 6
    # print('Mean of rmse per class is ' + str(mean_rmse))

    if plot:

        data = []
        for i in range(0, 6):
            data.append([])

        for idx, label in enumerate(test_label):
            data[label].append((round(float(predictions[idx]),4)))

        data_transposed = [list(i) for i in zip(*data)]
        # basic plot
        df = pd.DataFrame(data_transposed, columns=['0', '1', '2', '3', '4', '5'])
        ax = df.plot.box()
        ax.set_xlabel('Ground truth values')
        ax.set_ylabel('Predicted values')
        plot = ax

        label = "Training score"
        fig = plot.get_figure()
        fig.savefig(train_dir + '/'+ title + 'val_scores.png', bbox_inches='tight')
#        fig.close()


def evaluate_log_regression(train_dir, predictions, test_label, title='',plot=True):

    print('Mapping probabilities with arg max....evaluating')


    new_predictions = prob_to_class_argmax(predictions)
    evaluate_linear_regression(train_dir,new_predictions, test_label, 'arg_max_', plot) #to do, print out some heatmap or something similar

    print('Mapping probabilities to float....evaluating')

    new_predictions = prob_to_float(predictions)
    evaluate_linear_regression(train_dir,new_predictions, test_label, 'float_', plot)

    print('Mapping probabilities to float using NN....evaluating')

    new_predictions = prob_to_float_NN(predictions)
    evaluate_linear_regression(train_dir,new_predictions, test_label, 'float_NN_', plot)



