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
  return tf.log(tf.reduce_sum(tf.exp(input_tensor - max_input_tensor2), 
                                reduction_indices, keep_dims=keep_dims)) + max_input_tensor1, max_input_tensor1

def logprob(x,priors,means,std,logpdf):
    """log(p(z|x)) = log(p(x|z)p(z)) - log(sum(exp((x-mean)^2/2(std^2) + ln(p(z)/root(2pistd^2)))"""

    """casting types"""
    means = tf.cast(means, tf.float32)
    x = tf.cast(x,tf.float32)

    """tiled data"""
    x = tf.reshape(tf.tile(x,[1,FLAGS.k]),[N,FLAGS.k,D])

    """p(x|z)"""
    likelihood = logpdf(tf.cast(x,tf.float32)) # N x K X D

    """constant term"""
    const_term = tf.log(priors/tf.cast(((2*pi)**(1./D))*std,tf.float32))

    """exp. term"""
    exp_term = tf.divide(-1 * (tf.subtract(x,means)**2), 2*(std**2))

    """powerof"""
    powerof = tf.reduce_sum(tf.add(const_term, exp_term),2)

    """use logsumexp function"""
    denominator, max_tensor = reduce_logsumexp(powerof, keep_dims=True)

    """numerator"""
    numerator = tf.add(likelihood, tf.log(tf.transpose(priors)))

    """result"""
    result = tf.subtract(numerator,denominator)

    return result,numerator,priors,likelihood, denominator



def defineGraph(trainData, clusterCentroids, covarianceMatrix):
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

    """calculate the log prob density function using formula
		f(x) = (2 pi)^(-k/2) |det(C)|^(-1/2) exp(-1/2 (x - mu)^T C^{-1} (x - mu))

		assume std is a [kxd] representing the std's of each cluster with indenpendent dimensions
    """
    normDist = tf.contrib.distributions.MultivariateNormalDiag(clusterCentroids, covarianceMatrix)

    logpdffunction = normDist.log_pdf

    return loss, clusterAssignments, logpdffunction

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
    covMatrix = tf.cast(tf.Variable(tf.zeros([FLAGS.k, D]), name='covariance') + 0.2, tf.float32)
    priors = tf.cast(tf.Variable(tf.ones([FLAGS.k,1]), name = 'priors')/FLAGS.k, tf.float32)

    trainLoss, trainAssignments, logpdffunction = defineGraph(trainData, clusterCentroids, covMatrix)
    if FLAGS.validate == True:
        validLoss, validAssignments, logpdffunction = defineGraph(validData, clusterCentroids, covMatrix)

    """initialize the logprob function"""
    logprobs, numerator, pr, likeli, denominator = logprob(trainData,priors, clusterCentroids, covMatrix, logpdffunction)

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

            logprobability, n, d = sess.run([logprobs, numerator, denominator])
            if epoch_num == 100 or epoch_num == 200:
                print "logprob"
                print logprobability
                print "numerator"
                print n
                print "denomiator"
                print d

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







