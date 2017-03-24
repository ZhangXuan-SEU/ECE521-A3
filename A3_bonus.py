import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import logging 
import os
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.image
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import pylab
from mpl_toolkits.mplot3d import proj3d

'''
The Arrow3D class is adopted from online sources for plotting 3D arrow
'''
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


timestr = time.strftime("%Y%m%d-%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument(
    '--eps',
    type=float,
    default=0.001,
    help='Initial learning rate.'
)
parser.add_argument(
    '--num_epochs',
    type=int,
    default=2000,
    help='Total number of epochs'
)
parser.add_argument(
    '--k',
    type=int,
    default=4,
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


def loadToyData():
    mu = [0, 0, 0.]
    diag_stdev = [1, 1, 1.]
    dist = tf.contrib.distributions.MultivariateNormalDiag(mu, diag_stdev)
    s = dist.sample(sample_shape=(200))
    s = tf.cast(s, tf.float32)
    A = tf.constant(np.array([[1, 0, 0], [1, 0.001, 0], [0, 0, 10]]), dtype=tf.float32)
    x = tf.matmul(A, tf.transpose(s))
    return x

def plot_images(images, ax, ims_per_row=4, padding=4, digit_dimensions=(8, 8),
                cmap=matplotlib.cm.binary, vmin=None, vmax=None):
    """Images should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
                      col_start: col_start + digit_dimensions[1]] = cur_image
    cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    return cax

def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])

def loadData2D():
    npz = open("tinymnist.npz", "rb")
    data = np.load(npz)
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


def plot3D(data, W, U):
    X = data[:,0]
    Y = data[:,1]
    Z = data[:,2]
    Xbar = np.mean(X)
    Ybar = np.mean(Y)
    Zbar = np.mean(Z)
    fig = pylab.figure()
    ax = Axes3D(fig)
    ax.set_aspect('equal')
    #Setting all axis to be the same
    max_range = (np.max(Z)-np.min(Z))/2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.scatter(data[:,0], data[:,1],data[:,2])
    ax.scatter(Xbar, Ybar, Zbar, 'o', s=100, c='r')

    arrow = Arrow3D([Xbar, W[0]], [Xbar, W[1]], 
                    [Zbar, W[2]], mutation_scale=10, 
                    lw=2, arrowstyle="-|>", color="r")
    ax.add_artist(arrow)
    arrow = Arrow3D([Xbar, U[0]], [Xbar, U[1]], 
                    [Zbar, U[2]], mutation_scale=10, 
                    lw=2, arrowstyle="-|>", color="r")
    ax.add_artist(arrow)

def definePCAGraph(x, K, D):
    initial_u = tf.truncated_normal([D, K], stddev=1.0/math.sqrt(D))
    N = shape(x)[1]
    U = tf.Variable(initial_u)
    U = U/tf.matmul(tf.transpose(U), U)
    xbar = tf.expand_dims(tf.reduce_mean(x, reduction_indices=1), 1)
    S = tf.matmul(x-xbar, tf.transpose(x-xbar))/float(N)

    loss = tf.matmul(tf.transpose(U), tf.matmul(S, U))

    return loss, U

def defineFAGraph(x, K, D):
    initial_w = tf.truncated_normal([D, K], stddev=1.0/math.sqrt(D))
    W = tf.Variable(initial_w)

    initial_psi = tf.eye(D, D)
    #Multiply by eye to ensure Psi is always diagonal
    Psi = tf.square(tf.multiply(tf.Variable(initial_psi), tf.eye(D,D)))

    initial_mu = tf.expand_dims(tf.zeros([D]), 1)
    mu = tf.Variable(initial_mu)

    A = tf.add(Psi,tf.matmul(W, tf.transpose(W)))

    firstTerm = -D/2.0 * tf.log(2*math.pi)   #1 x 1
    logDet = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(A)))) # 1 x 1
    secondTerm = -1.0/2.0  * logDet
    thirdTerm =  -1.0/2.0  * tf.matmul(tf.transpose(x-mu), tf.matmul(tf.matrix_inverse(A), x-mu))

    logLikelihood = tf.trace(firstTerm + secondTerm + thirdTerm)

    loss = (-logLikelihood)

    return loss, W

def Pt3():
    setup_logger('training log', "./train.log", stream=True)
    logger = logging.getLogger('training log')

    x = loadToyData()
    K = 1
    D = shape(x)[0]
    lossFA, W = defineFAGraph(x, K, D)
    trainFA_step = tf.train.AdamOptimizer(learning_rate=FLAGS.eps, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(lossFA)
    lossPCA, U = definePCAGraph(x, K, D)
    trainPCA_step = tf.train.AdamOptimizer(learning_rate=FLAGS.eps, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(lossPCA)


    sess = tf.Session()

    epoch_num = 0
    with sess.as_default():
        init = tf.global_variables_initializer()
        sess.run(init)
        
        while epoch_num < FLAGS.num_epochs:
            trainFA_loss, trainPCA_loss = sess.run([lossFA, lossPCA])
            t = datetime.datetime.utcnow()
            logger.info("Epochs Trained: %d/%d" % (epoch_num, FLAGS.num_epochs))
            logger.info("Train---FA  LOSS: %.4f" %(trainFA_loss))
            logger.info("Train---PCA LOSS: %.4f" %(trainPCA_loss))
            epoch_num+=1
            sess.run(trainFA_step)
            sess.run(trainPCA_step)
        W_final = np.asarray(W.eval())
        U_final = np.asarray(U.eval())
        xData = x.eval()
    print W_final
    print U_final
    plot3D(xData.T, W_final*10, U_final*10)
    plt.show()

def Pt2():

    setup_logger('training log', "./train.log", stream=True)
    logger = logging.getLogger('training log')


    Data = loadData2D()

    trainX = Data['x'].T
    trainY = Data['y']
    validX = Data['x_valid'].T
    validY = Data['y_valid']
    testX = Data['x_test'].T
    testY = Data['y_test']

    x = tf.placeholder(tf.float32, shape=[trainX.shape[0], None])
    K = FLAGS.k
    D = trainX.shape[0]

    loss, W= defineFAGraph(x, K, D)
    train_step = tf.train.AdamOptimizer(learning_rate=FLAGS.eps, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.eps).minimize(loss)

    sess = tf.Session()

    epoch_num = 0
    with sess.as_default():
        init = tf.global_variables_initializer()
        sess.run(init)
        
        while epoch_num < FLAGS.num_epochs:
            train_loss = sess.run([loss], feed_dict={x: trainX})
            t = datetime.datetime.utcnow()
            logger.info("Epochs Trained: %d/%d" % (epoch_num, FLAGS.num_epochs))
            logger.info("Train---LOSS: %.4f" %(train_loss[0]))
            valid_loss = sess.run([loss], feed_dict={x: validX})
            logger.info("Valid---LOSS: %.4f" %(valid_loss[0]))
            test_loss = sess.run([loss], feed_dict={x: testX})
            logger.info("Test---LOSS: %.4f" %(test_loss[0]))
            epoch_num+=1
            sess.run(train_step, feed_dict={x: trainX})
        W_final = np.asarray(W.eval())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = plot_images(W_final.T, ax)
    fig.colorbar(cax)
    plt.show()

if __name__ == '__main__':
    FLAGS = parser.parse_args()

    #Pt2()

    Pt3()

