import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import pi
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
    default=0.1,
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
    with open("data100D.npy", "rb") as npy:
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
    ax.scatter(data[:,0], data[:,1], marker='o', s=1, linewidths=1, c=assignment, cmap=plt.cm.coolwarm)
    ax.scatter(centroids[:,0], centroids[:,1], marker='x', s=100, linewidths=5, c='k', cmap=plt.cm.coolwarm)

def logsoftmax(input_tensor):
  """Computes normal softmax nonlinearity in log domain.

     It can be used to normalize log probability.
     The softmax is always computed along the second dimension of the input Tensor.     
 
  Args:
    input_tensor: Unnormalized log probability.
  Returns:
    normalized log probability.
  """
  return input_tensor - reduce_logsumexp(input_tensor, keep_dims=True)


def reduce_logsumexp(input_tensor, reduction_indices=1, keep_dims=False):
  """Computes the sum of elements across dimensions of a tensor in log domain.
     
     It uses a similar API to tf.reduce_sum.

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    reduction_indices: The dimensions to reduce. 
    keep_dims: If true, retains reduced dimensions with length 1.
  Returns:
    The reduced tensor.
  """
  max_input_tensor1 = tf.reduce_max(input_tensor, 
                                    reduction_indices, keep_dims=keep_dims)
  max_input_tensor2 = max_input_tensor1
  if not keep_dims:
    max_input_tensor2 = tf.expand_dims(max_input_tensor2, 
                                       reduction_indices)

  # sess = tf.Session()
  # init = tf.global_variables_initializer()
  # sess.run(init)
  # val = sess.run(tf.log(tf.reduce_sum(tf.exp(input_tensor - max_input_tensor2), 
  #                               reduction_indices, keep_dims=keep_dims)) + max_input_tensor1)
  # print "val: ", val

  return tf.log(tf.reduce_sum(tf.exp(input_tensor - max_input_tensor2), 
                                reduction_indices, keep_dims=keep_dims)) + max_input_tensor1

def defineGraph(trainData, clusterCentroids, covarianceMatrix, priors):
    N, D = shape(trainData)
    print D
    trainData =  tf.cast(trainData, tf.float32)

    assignment_init = tf.cast(tf.zeros([N]), tf.int32)
    clusterAssignments = tf.Variable(assignment_init)

    clusterCentroidsTiled = tf.reshape(tf.tile(clusterCentroids, [N, 1]), [N, FLAGS.k, D])
    trainDataTiled = tf.reshape(tf.tile(trainData, [1, FLAGS.k]), [N, FLAGS.k, D])

    normDist = tf.contrib.distributions.MultivariateNormalDiag(clusterCentroids, tf.tile(covarianceMatrix,[1,D]))
    # print clusterCentroids
    # print tf.tile(covarianceMatrix,[1,D])
    """calculating the llf"""

    """constant term"""
    #const_term = tf.log(priors/tf.cast(((2*pi)**(D/2.))*(tf.reduce_prod(covarianceMatrix))**(1./2),tf.float32))
    #print priors
    #const_term = priors + tf.log(tf.cast(((2*pi)**(D/-2.) * ((tf.reduce_prod(covarianceMatrix))**(-0.5))), tf.float32))
    term1 = (2*pi)**(D/-2.)
    term2 = covarianceMatrix**(D/-2.)
    #print trainDataTiled
    powerof = tf.add(tf.transpose(priors),normDist.log_pdf(trainDataTiled))
    #tf.log(tf.cast(term1*term2,tf.float32))

    """exp. term"""
    #exp_term = -1./2 * (tf.divide(tf.subtract(trainDataTiled,clusterCentroids)**2,covarianceMatrix))
    #exp_term = tf.divide(-1./2 * (tf.subtract(trainDataTiled,clusterCentroids)**2), (covarianceMatrix))

    """powerof"""
    #print const_term
    #powerof = tf.transpose(const_term) + tf.reduce_sum(exp_term,2)

    """use logsumexp function log(P(x)) """
    result = reduce_logsumexp(powerof, keep_dims=True)

    """summing across all points"""
    loss = -1 * tf.reduce_sum(result)

    """assignments to each cluster, softmax to determine best"""
    softmax = powerof -  result
    clusterAssignments = tf.argmax(softmax, axis=1)

    return loss, result, clusterAssignments

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

    """constant variables per cluster"""
    phi = tf.cast(tf.Variable(tf.zeros([FLAGS.k, 1]), name='covariance')+0.1, tf.float32)
    covMatrix = tf.exp(phi)
    psi = tf.Variable(tf.ones([1,FLAGS.k])/FLAGS.k,tf.float32)
    #priors = tf.exp(psi)/(tf.reduce_sum(tf.exp(psi)))
    priors = tf.transpose(logsoftmax(psi))

    trainLoss, result, trainAssignments = defineGraph(trainData, clusterCentroids, covMatrix,priors)
    if FLAGS.validate == True:
        trainLoss, result, trainAssignments = defineGraph(validData, clusterCentroids, covMatrix,priors)

    #train_step = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.eps).minimize(trainLoss)
    train_step = tf.train.AdamOptimizer(learning_rate=FLAGS.eps, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(trainLoss)

    sess = tf.Session()

    epoch_num = 0
    with sess.as_default():
        init = tf.global_variables_initializer()
        sess.run(init)
        while epoch_num < FLAGS.num_epochs:
            train_loss = sess.run([trainLoss])
            t = datetime.datetime.utcnow()
            #logger.info("Epochs Trained: %d/%d" % (epoch_num, FLAGS.num_epochs))
            #logger.info("Train---LOSS: %.4f" %(train_loss[0]))
            if FLAGS.validate == True:
                valid_loss = sess.run([validLoss])
                logger.info("Valid---LOSS: %.4f" %(valid_loss[0]))

            epoch_num+=1
            sess.run(train_step)

            a,b,cov, pri, means= sess.run([trainLoss,clusterCentroids, covMatrix, priors, clusterCentroids])
            print "train loss ", a
            # print "prior term ", pri
            # # # print "means ", means
            # print "cov ", cov
            # # print "prior ", pri
            # # # #print "powerof ", c
            # # print "exp_term ", d
            # print "const term ", const
            # # print "covmatrix ", f

            # import sys
            # sys.exit(-1)

        #Done training
        finalTrainData, finalTrainAssignments, finalCentroids,d = sess.run([trainData, trainAssignments, clusterCentroids,result])
        print "final result: ", finalTrainAssignments
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







