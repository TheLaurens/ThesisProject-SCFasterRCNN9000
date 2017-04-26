# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

from fast_rcnn.config import cfg
import gt_data_layer.roidb as gdl_roidb
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from tensorflow.examples.tutorials.mnist import input_data
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import sys
from tensorflow.python.client import timeline
import time

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, saver, network, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        # For checkpoint
        self.saver = saver

    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

    def _modified_smooth_l1(self, sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma2 = sigma * sigma

        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul

    #helper loss funcs
    def zBar(self,x):
        xshape = x.shape.as_list()
        print '============================================='
        print xshape
        s=[-1,xshape[1]*xshape[2]]
        #return tf.reshape(x,s)
        return tf.maximum(tf.reshape(x,s),0)
    
    def bigU(self,zb):
        return tf.matmul(tf.transpose(zb),zb)

    def selectNonDiag(self,x):
        selection = np.ones(x.shape.as_list()[0],dtype='float32') - np.eye(x.shape.as_list()[0],dtype='float32')
        return tf.reduce_sum(tf.multiply(x,selection))

    def bigV(self,x):
        smallNu=tf.reshape(tf.reduce_sum(x,axis=0),[1,-1])
        return tf.multiply(tf.transpose(smallNu),smallNu)

    def specialNormalise(self,x):
        top = self.selectNonDiag(x)
        bottom = tf.multiply(tf.to_float(x.shape[1]-1),tf.reduce_sum(tf.multiply(x,np.eye(x.shape[1],dtype='float32'))))
        return tf.divide(top,bottom)

    def frobNorm(self,x):
        return tf.sqrt(tf.reduce_sum(tf.square(x)))

    def train_model(self, sess, max_iters):
        """Network training loop."""
        cfg.TRAIN.LEARNING_RATE = 1e-5
        y = {0:[0,1], 1:[1,0]}
        tresh = tf.constant(0.03)
        cc0=1.0
        cc1=1.0
        cc2=1.0
        cc3=0.0003
        cc4=0.000001
        c0 = tf.constant(cc0)
        c1 = tf.constant(cc1)
        c2 = tf.constant(cc2)
        c3val = tf.constant(cc3)
        c3 = lambda affinity: tf.cond(tf.less(affinity,tresh),lambda: c3val,lambda: tf.constant(0.0))
        c4 =tf.constant(cc4)
        
        stackedClusts = self.net.get_output('stackedClusts')
        
        bZ = self.zBar(stackedClusts)
        bU = self.bigU(bZ)
        coact = self.selectNonDiag(bU)
        affinity = self.specialNormalise(bU)

        #balance
        bV=self.bigV(bZ)
        balance = self.specialNormalise(bV)

        #cross entropy
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.net.y_ * tf.log(tf.clip_by_value(self.net.get_output('smStacked'),1e-10,1.0)), reduction_indices=[1]))

        frob = self.frobNorm(stackedClusts)

        loss = c0*cross_entropy + c1*affinity + c2*tf.subtract(tf.constant(1.0),balance) + c3(affinity)*coact + c4*frob
        
        # optimizer and learning rate
        global_step = tf.Variable(0, trainable=False)
        #lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step,
        #                                cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
        #momentum = cfg.TRAIN.MOMENTUM
        #train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss, global_step=global_step)
        lr = 1e-5
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(self.net.get_output('smStacked'),1), tf.argmax(self.net.y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # iintialize variables
        sess.run(tf.global_variables_initializer())
        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):
            # get one batch
            trainbatch = mnist.train.next_batch(128)
            trainbatch = (trainbatch[0],np.array([y[np.argmax(trainbatch[1][j])>4] for j in range(len(trainbatch[1]))]))
            # Make one SGD update
            feed_dict={self.net.data: trainbatch[0], self.net.y_: trainbatch[1], self.net.keep_prob: 0.5}

            run_options = None
            run_metadata = None
            if cfg.TRAIN.DEBUG_TIMELINE:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            timer.tic()

            train_loss, train_acc, _ = sess.run([loss,accuracy, train_op], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)

            timer.toc()

            if cfg.TRAIN.DEBUG_TIMELINE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(str(long(time.time() * 1000)) + '-train-timeline.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()

            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                print 'iter: %d / %d, total loss: %.4f, train_accuracy: %.4f, lr: %f'%\
                        (iter+1, max_iters, train_loss, train_acc, lr)#lr.eval())
                print 'speed: {:.3f}s / iter'.format(timer.average_time)
                #print  self.net.get_output('fc7').eval(feed_dict=feed_dict)
                aff = affinity.eval(feed_dict=feed_dict)
                bal = balance.eval(feed_dict=feed_dict)
                coa = coact.eval(feed_dict=feed_dict)
                entr = cross_entropy.eval(feed_dict=feed_dict)
                frb = frob.eval(feed_dict=feed_dict)
                #print bV.eval(feed_dict=feed_dict)
                print("cross_entropy: %g, affinity: %g, balance: %g, coact: %g, frob: %g"%(cc0*entr,cc1*aff,cc2*(1-bal),cc3*coa,cc4*frb))

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

def train_net(network, output_dir, max_iters=40000):
    """Train a Fast R-CNN network."""
    #roidb = filter_roidb(roidb)
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sw = SolverWrapper(sess, saver, network, output_dir)
        print 'Solving...'
        sw.train_model(sess, max_iters)
        print 'done solving'
