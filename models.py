"""
Please implement a standard AlexNet model here as defined in the paper
    https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Note: Although you will only have to edit a small fraction of the code at the
beginning of the assignment by filling in the blank spaces, you will need to
build on the completed starter code to fully complete the assignment,
We expect that you familiarize yourself with the codebase and learn how to
setup your own experiments taking the assignments as a basis. This code does
not cover all parts of the assignment and only provides a starting point. To
fully complete the assignment significant changes have to be made and new
functions need to be added after filling in the blanks. Also, for your projects
we won't give out any code and you will have to use what you have learned from
your assignments. So please always carefully read through the entire code and
try to understand it. If you have any questions about the code structure,
we will be happy to answer it.

Attention: All sections that need to be changed to complete the starter code
are marked with EDIT!
"""

import os
import numpy as np
import tensorflow as tf

def alexnet_model(inputs, train=True, norm=True, **kwargs):
    """
    Vanilla AlexNet
    """

    # propagate input targets
    outputs = inputs
    dropout = .5 if train else None
    input_to_network = inputs['images']
    weight_decay = 0.0005

    ### YOUR CODE HERE

    # set up all layer outputs
    outputs['conv1'],outputs['conv1_kernel']  = conv(outputs['images'], 96, 11, 4, 
        padding='VALID', layer = 'conv1', weight_decay=weight_decay)
    lrn1 = outputs['conv1']
    if norm:
        lrn1 = lrn(outputs['conv1'], depth_radius=5, bias=1, alpha=.0001, beta=.75, layer='conv1')
    outputs['pool1'] = max_pool(lrn1, 3, 2, layer = 'pool1')
    
    
    outputs['conv2'], outputs['conv2_kernel'] = conv(outputs['pool1'], 256, 5, 1, layer = 'conv2', weight_decay=weight_decay)
    lrn2 = outputs['conv2']
    if norm:
        lrn2 = lrn(outputs['conv2'], depth_radius=5, bias=1, alpha=.0001, beta=.75, layer='conv2')

    outputs['pool2'] = max_pool(lrn2, 3, 2, layer = 'pool2')
    outputs['conv3'],outputs['conv3_kernel'] = conv(outputs['pool2'], 384, 3, 1, layer = 'conv3', weight_decay=weight_decay)
    outputs['conv4'],outputs['conv4_kernel'] = conv(outputs['conv3'], 384, 3, 1, layer = 'conv4', weight_decay=weight_decay)
    outputs['conv5'],outputs['conv5_kernel'] = conv(outputs['conv4'], 256, 3, 1, layer = 'conv5', weight_decay=weight_decay)
    outputs['pool5'] = max_pool(outputs['conv5'], 3, 2, layer = 'pool5')

    outputs['fc6'] = fc(outputs['pool5'], 4096, dropout=dropout, bias=.1, layer = 'fc6', weight_decay=weight_decay)
    outputs['fc7'] = fc(outputs['fc6'],4096, dropout=dropout, bias=.1, layer = 'fc7', weight_decay=weight_decay)
    outputs['fc8'] = fc(outputs['fc7'],1000, activation=None, dropout=None, bias=0, layer = 'fc8', weight_decay=weight_decay)

    outputs['pred'] = outputs['fc8']
    

    for k in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'pool1',
            'pool2', 'pool5', 'fc6', 'fc7', 'fc8', 'conv1_kernel', 'pred']:
        assert k in outputs, '%s was not found in outputs' % k
    return outputs, {}
        #    return outputs['pred'], {}

def rotation_model(inputs, train=True, norm=True, **kwargs):
    """
    Vanilla AlexNet
    """

    # propagate input targets
    outputs = inputs
    dropout = .5 if train else None
    batch_size = inputs['images'].get_shape().as_list()[0]
    weight_decay = 0.0005

    # rotations
    rotation_labels = np.random.randint(0, 4, batch_size,dtype='int32')
    #rotation_labels = np.array([0,1,2,3] * batch_size, dtype = np.int32)
    input_to_network = tf.contrib.image.rotate(
        inputs['images'],
        rotation_labels * 1.5708, # roation in radians
        )
    #rotated_ims = tf.map_fn(
    #    lambda x: (x, tf.image.rot90(x, 1) , tf.image.rot90(x, 2), tf.image.rot90(x, 3)), 
    #    inputs['images'], 
    #    dtype=(tf.float32, tf.float32, tf.float32, tf.float32)
    #    )
    #input_to_network = tf.concat(rotated_ims, 0) # concat the results
    outputs['labels_rotation'] = rotation_labels
    ### YOUR CODE HERE

    # set up all layer outputs
    outputs['conv1'],outputs['conv1_kernel']  = conv(input_to_network, 96, 11, 4, padding='VALID', layer = 'conv1', weight_decay=weight_decay)
    lrn1 = outputs['conv1']
    if norm:
        lrn1 = lrn(outputs['conv1'], depth_radius=5, bias=1, alpha=.0001, beta=.75, layer='conv1')
    outputs['pool1'] = max_pool(lrn1, 3, 2, layer = 'pool1')
    
    
    outputs['conv2'], outputs['conv2_kernel'] = conv(outputs['pool1'], 256, 5, 1, layer = 'conv2', weight_decay=weight_decay)
    lrn2 = outputs['conv2']
    if norm:
        lrn2 = lrn(outputs['conv2'], depth_radius=5, bias=1, alpha=.0001, beta=.75, layer='conv2')

    outputs['pool2'] = max_pool(lrn2, 3, 2, layer = 'pool2')
    outputs['conv3'],outputs['conv3_kernel'] = conv(outputs['pool2'], 384, 3, 1, layer = 'conv3', weight_decay=weight_decay)
    outputs['conv4'],outputs['conv4_kernel'] = conv(outputs['conv3'], 384, 3, 1, layer = 'conv4', weight_decay=weight_decay)
    outputs['conv5'],outputs['conv5_kernel'] = conv(outputs['conv4'], 256, 3, 1, layer = 'conv5', weight_decay=weight_decay)
    outputs['pool5'] = max_pool(outputs['conv5'], 3, 2, layer = 'pool5')

    outputs['fc6'] = fc(outputs['pool5'], 256, dropout=dropout, bias=.1, layer = 'fc6', weight_decay=weight_decay)
    outputs['fc7'] = fc(outputs['fc6'],256, dropout=dropout, bias=.1, layer = 'fc7', weight_decay=weight_decay)
    outputs['fc8'] = fc(outputs['fc7'],4, activation=None, dropout=None, bias=0, layer = 'fc8', weight_decay=weight_decay)

    outputs['pred_rotation'] = outputs['fc8']
    
    for k in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'pool1',
            'pool2', 'pool5', 'fc6', 'fc7', 'fc8', 'conv1_kernel', 'pred_rotation']:
        assert k in outputs, '%s was not found in outputs' % k
    return outputs, {}
        #    return outputs['pred'], {}

def multitask_model(inputs, train=True, norm=True, **kwargs):
    """
    Vanilla AlexNet
    """

    # propagate input targets
    outputs = inputs
    dropout = .5 if train else None
    batch_size = inputs['images'].get_shape().as_list()[0]

    # rotations
    rotation_labels = np.random.randint(0, 4, batch_size,dtype='int32')
    input_to_network = tf.contrib.image.rotate(
        inputs['images'],
        rotation_labels * 1.5708, # roation in radians
        )
    #input_to_network = tf.map_fn(
    #    lambda x: tf.image.rot90(x[0], x[1]), 
    #    (inputs['images'], rotation_labels), 
    #    dtype=tf.float32
    #    )
    outputs['labels_rotation'] = rotation_labels
    ### YOUR CODE HERE

    # set up all layer outputs
    outputs['conv1'],outputs['conv1_kernel']  = conv(outputs['images'], 96, 11, 4, padding='VALID', layer = 'conv1')
    lrn1 = outputs['conv1']
    if norm:
        lrn1 = lrn(outputs['conv1'], depth_radius=5, bias=1, alpha=.0001, beta=.75, layer='conv1')
    outputs['pool1'] = max_pool(lrn1, 3, 2, layer = 'pool1')
    
    
    outputs['conv2'], outputs['conv2_kernel'] = conv(outputs['pool1'], 256, 5, 1, layer = 'conv2')
    lrn2 = outputs['conv2']
    if norm:
        lrn2 = lrn(outputs['conv2'], depth_radius=5, bias=1, alpha=.0001, beta=.75, layer='conv2')

    outputs['pool2'] = max_pool(lrn2, 3, 2, layer = 'pool2')
    outputs['conv3'],outputs['conv3_kernel'] = conv(outputs['pool2'], 384, 3, 1, layer = 'conv3')
    outputs['conv4'],outputs['conv4_kernel'] = conv(outputs['conv3'], 384, 3, 1, layer = 'conv4')
    outputs['conv5'],outputs['conv5_kernel'] = conv(outputs['conv4'], 256, 3, 1, layer = 'conv5')
    outputs['pool5'] = max_pool(outputs['conv5'], 3, 2, layer = 'pool5')

    # rotation head
    outputs['fc6_rot'] = fc(outputs['pool5'], 256, dropout=dropout, bias=.1, layer = 'fc6')
    outputs['fc7_rot'] = fc(outputs['fc6_rot'],256, dropout=dropout, bias=.1, layer = 'fc7')
    outputs['fc8_rot'] = fc(outputs['fc7_rot'],4, activation=None, dropout=None, bias=0, layer = 'fc8')
    outputs['pred_rotation'] = outputs['fc8_rot']

    # classification head
    outputs['fc6_clf'] = fc(outputs['pool5'], 4096, dropout=dropout, bias=.1, layer = 'fc6')
    outputs['fc7_clf'] = fc(outputs['fc6_clf'],4096, dropout=dropout, bias=.1, layer = 'fc7')
    outputs['fc8_clf'] = fc(outputs['fc7_clf'],1000, activation=None, dropout=None, bias=0, layer = 'fc8')
    outputs['pred'] = outputs['fc8_clf']


    return outputs, {}
        #    return outputs['pred'], {}

def conv(inp,
         out_depth,
         ksize=[3,3],
         strides=[1,1,1,1],
         padding='SAME',
         kernel_init='xavier',
         kernel_init_kwargs=None,
         bias=0,
         weight_decay=None,
         activation='relu',
         batch_norm=False,
         name='conv',
         layer = None,
         ):
    with tf.variable_scope(layer):
        # assert out_shape is not None
        if weight_decay is None:
            weight_decay = 0.
        if isinstance(ksize, int):
            ksize = [ksize, ksize]
            
        if isinstance(strides, int):
            strides = [1, strides, strides, 1]            
            
        if kernel_init_kwargs is None:
            kernel_init_kwargs = {}
        in_depth = inp.get_shape().as_list()[-1]

        # weights
        init = initializer(kernel_init, **kernel_init_kwargs)


        kernel = tf.get_variable(initializer=init,
                                shape=[ksize[0], ksize[1], in_depth, out_depth],
                                dtype=tf.float32,
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                name='weights')
        init = initializer(kind='constant', value=bias)
        biases = tf.get_variable(initializer=init,
                                shape=[out_depth],
                                dtype=tf.float32,
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                name='bias')
        # ops
        conv = tf.nn.conv2d(inp, kernel,
                            strides=strides,
                            padding=padding)
        output = tf.nn.bias_add(conv, biases, name=name)

        if activation is not None:
            output = getattr(tf.nn, activation)(output, name=activation)
        if batch_norm:
            output = tf.nn.batch_normalization(output, mean=0, variance=1, offset=None,
                                scale=None, variance_epsilon=1e-8, name='batch_norm')
    return output, kernel

def max_pool(x, ksize, strides,  name='pool', padding='SAME', layer = None):
    with tf.variable_scope(layer):
        if isinstance(ksize, int):
            ksize = [ksize, ksize]
        if isinstance(strides, int):
            strides = [1, strides, strides, 1]
    return tf.nn.max_pool(x, ksize= [1, ksize[0], ksize[1],1],
                        strides = strides,
                        padding = padding, name = name)

def fc(inp,
       out_depth,
       kernel_init='xavier',
       kernel_init_kwargs=None,
       bias=1,
       weight_decay=None,
       activation='relu',
       batch_norm=True,
       dropout=None,
       dropout_seed=None,
       name='fc',
       layer='blah'):
    with tf.variable_scope(layer):
        if weight_decay is None:
            weight_decay = 0.
        # assert out_shape is not None
        if kernel_init_kwargs is None:
            kernel_init_kwargs = {}
        resh = tf.reshape(inp, [inp.get_shape().as_list()[0], -1], name='reshape')
        in_depth = resh.get_shape().as_list()[-1]

        # weights
        init = initializer(kernel_init, **kernel_init_kwargs)
        kernel = tf.get_variable(initializer=init,
                                shape=[in_depth, out_depth],
                                dtype=tf.float32,
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                name='weights')
        init = initializer(kind='constant', value=bias)
        biases = tf.get_variable(initializer=init,
                                shape=[out_depth],
                                dtype=tf.float32,
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                name='bias')

        # ops
        fcm = tf.matmul(resh, kernel)
        output = tf.nn.bias_add(fcm, biases, name=name)

        if activation is not None:
            output = getattr(tf.nn, activation)(output, name=activation)
        if batch_norm:
            output = tf.nn.batch_normalization(output, mean=0, variance=1, offset=None,
                                scale=None, variance_epsilon=1e-8, name='batch_norm')
        if dropout is not None:
            output = tf.nn.dropout(output, dropout, seed=dropout_seed, name='dropout')
    return output

def initializer(kind='xavier', *args, **kwargs):
    if kind == 'xavier':
        init = tf.contrib.layers.xavier_initializer(*args, **kwargs)
    else:
        init = getattr(tf, kind + '_initializer')(*args, **kwargs)
    return init


def lrn(inp,
    depth_radius=5, 
    bias=1, 
    alpha=.0001, 
    beta=.75, 
    name = 'lrn',
    layer = None):
    with tf.variable_scope(layer):
        lrn = tf.nn.local_response_normalization(inp, depth_radius = depth_radius, alpha = alpha,
                                            beta = beta, bias = bias, name = name)
    return lrn