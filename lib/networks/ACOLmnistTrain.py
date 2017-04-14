import tensorflow as tf
from networks.network import Network

#define
n_classes = 2
n_clusters = 5
_feat_stride = [16,]
#anchor_scales = [8, 16, 32]

class ACOLmnistTrain(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.clustcount = n_clusters
        self.data = tf.placeholder(tf.float32, shape=[None, 784])
        self.x_image = tf.reshape(self.data,[-1,28,28,1])
        self.y_ = tf.placeholder(tf.float32, shape=[None,n_classes])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
#        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'im_info':self.im_info})#, 'gt_boxes':self.gt_boxes})
        self.trainable = trainable
        self.setup()

        # create ops and placeholders for bbox normalization process
        #with tf.variable_scope('bbox_pred', reuse=True):
        #    weights = tf.get_variable("weights")
        #    biases = tf.get_variable("biases")

        #    self.bbox_weights = tf.placeholder(weights.dtype, shape=weights.get_shape())
        #    self.bbox_biases = tf.placeholder(biases.dtype, shape=biases.get_shape())

        #    self.bbox_weights_assign = weights.assign(self.bbox_weights)
        #    self.bbox_bias_assign = biases.assign(self.bbox_biases)

    def setup(self):
        (self.feed('x_image')
             .conv(5, 5, 1, 32, name='conv1_1')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv(5, 5, 32, 64, name='conv2_1')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2'))
             
        (self.feed('pool2')
             .reshape_layer([-1, 7*7*64], name='flat1')
             .fc(1024, name='fc7')
             .dropout(0.3, name='drop7'))
        
        (self.feed('drop7')
             .acol(self.clustcount, name='clust1'))
        (self.feed('drop7')
             .acol(self.clustcount, name='clust2'))
        
        (self.feed('clust1','clust2')
             .concat(name='stackedClusts')
             .matrix_softmax(name='softmaxMat')
             .redMax(name='smStacked'))
        
        
