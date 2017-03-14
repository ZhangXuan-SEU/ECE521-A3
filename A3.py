import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import logging 
import os
import datetime
import time

timestr = time.strftime("%Y%m%d-%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument(
    '--eps',
    type=float,
    default=0.01,
    help='Initial learning rate.'
)
parser.add_argument(
    '--num_epochs',
    type=int,
    default=200,
    help='Total number of epochs'
)
parser.add_argument(
    '--k',
    type=int,
    default=1,
    help='Number of clusters clusters'
)
parser.add_argument(
    '--validate',
    dest='validate',
    action='store_true'
)
parser.add_argument(
    '--no_validate',
    dest='validate',
    action='store_false'
)
parser.set_defaults(validate=False)

def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])

def loadData2D():
    with open("data2D.npy", "rb") as npy:
        data = np.load(npy)

    return data

def setup_logger(logger_name, log_file, level=logging.INFO, stream=False):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter()
    if not os.path.exists(logger_name):
        mode = 'w'
    else:
        mode = 'a'
    fileHandler = logging.FileHandler(log_file, mode=mode)
    fileHandler.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fileHandler)

    if stream==True:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        l.addHandler(streamHandler)


def plotClusters(data, assignment, centroids):
    fig, ax = plt.subplots()
    ax.scatter(data[:,0], data[:,1], marker='o', s=100, linewidths=1, c=assignment, cmap=plt.cm.coolwarm)
    ax.scatter(centroids[:,0], centroids[:,1], marker='x', s=100, linewidths=5, c='k', cmap=plt.cm.coolwarm)


def defineGraph(trainData, clusterCentroids):
    N, D = shape(trainData)

    trainData =  tf.cast(trainData, tf.float32)

    assignment_init = tf.cast(tf.zeros([N]), tf.int32)
    clusterAssignments = tf.Variable(assignment_init)

    clusterCentroidsTiled = tf.reshape(tf.tile(clusterCentroids, [N, 1]), [N, FLAGS.k, D])
    trainDataTiled = tf.reshape(tf.tile(trainData, [1, FLAGS.k]), [N, FLAGS.k, D])

    squareDist = tf.reduce_sum(tf.square(clusterCentroidsTiled - trainDataTiled), reduction_indices=2)
    minSquareDist = tf.reduce_min(squareDist, reduction_indices=1)
    clusterAssignments = tf.argmin(squareDist, axis=1)
    loss = tf.reduce_sum(minSquareDist)

    return loss, clusterAssignments

if __name__ == '__main__':
    FLAGS = parser.parse_args()

    setup_logger('training log', "./train.log", stream=True)
    logger = logging.getLogger('training log')

    if FLAGS.validate == False:
        trainData = tf.constant(loadData2D())
    else:
        Data = loadData2D()
        split = int(len(Data) * (2.0/3.0))
        trainData = tf.constant(Data[:split, :])
        validData = tf.constant(Data[split:, :])

    N, D = shape(trainData)


    centroid_init = tf.cast(tf.random_normal([FLAGS.k, D]), tf.float32)
    clusterCentroids = tf.get_variable('clusterCentroids', initializer=centroid_init)


    trainLoss, trainAssignments = defineGraph(trainData, clusterCentroids)
    if FLAGS.validate == True:
        validLoss, validAssignments = defineGraph(validData, clusterCentroids)

    train_step = tf.train.AdamOptimizer(learning_rate=FLAGS.eps, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(trainLoss)


    sess = tf.Session()

    epoch_num = 0
    with sess.as_default():
        init = tf.global_variables_initializer()
        sess.run(init)
        while epoch_num < FLAGS.num_epochs:
            train_loss = sess.run([trainLoss])
            t = datetime.datetime.utcnow()
            logger.info("Epochs Trained: %d/%d" % (epoch_num, FLAGS.num_epochs))
            logger.info("Train---LOSS: %.4f" %(train_loss[0]))
            if FLAGS.validate == True:
                valid_loss = sess.run([validLoss])
                logger.info("Valid---LOSS: %.4f" %(valid_loss[0]))

            epoch_num+=1
            sess.run(train_step)

        #Done training
        finalTrainData, finalTrainAssignments, finalCentroids = sess.run([trainData, trainAssignments, clusterCentroids])
        if FLAGS.validate == True:
            finalValidData, finalValidAssignments = sess.run([validData, validAssignments])


    #for i in range()
    plotClusters(finalTrainData, finalTrainAssignments, finalCentroids)
    trainIdx, trainCounts = np.unique(finalTrainAssignments, return_counts=True)
    trainPercent = trainCounts/float(len(finalTrainAssignments))
    print "Train:"
    print dict(zip(trainIdx, trainPercent))

    if FLAGS.validate == True:
        plotClusters(finalValidData, finalValidAssignments, finalCentroids)
        validIdx, validCounts= np.unique(finalValidAssignments, return_counts=True)
        validPercent = validCounts/float(len(finalValidAssignments))
        print "Valid:"
        print dict(zip(validIdx, validPercent))

    plt.show()







