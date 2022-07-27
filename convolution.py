from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math

def conv2d(inputs, filters, strides, padding):
    """
    Performs 2D convolution given 4D inputs and filter Tensors.
    :param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
    :param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
    :param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
    :param padding: either "SAME" or "VALID", capitalization matters
    :return: outputs, NumPy array or Tensor with shape [num_examples, output_height, output_width, output_channels]
    """
    num_examples = inputs.shape[0]
    in_height = inputs.shape[1]
    in_width = inputs.shape[2]
    input_in_channels = inputs.shape[3]

    filter_height = filters.shape[0]
    filter_width = filters.shape[1]
    filter_in_channels = filters.shape[2]
    filter_out_channels = filters.shape[3]

    num_examples_stride = None
    strideY = strides[1]
    strideX = strides[2]
    channels_stride = None
    
    assert input_in_channels == filter_in_channels
    

    # Cleaning padding input
    if padding == 'SAME':
        height = (filter_height - 1) // 2
        width = (filter_width - 1) // 2
        
    else:
        height = 0
        width = 0 
        
    inputs = np.pad(inputs, ((0,0), (height,width), (height,width), (0,0)), 'constant')

    # Calculate output dimensions
    
    width_2 = (in_width - filter_width + width*2) // strideX + 1
    height_2 = (in_height - filter_height + height*2) // strideY + 1
    
    
    result = np.zeros((num_examples, height_2, width_2, filter_out_channels))
    
    for i in range(0, num_examples):
        for height in range(0, height_2):
            for width in range(0, width_2):
                for channels in range(0, filter_out_channels):
                    var = inputs[i, height: height+filter_height, width: width+filter_width, :]
                    result[i, height, width, channels] = np.tensordot(var, filters[:, :, :, channels], ((0,1,2), (0,1,2)))
       
    outputs_array = tf.convert_to_tensor(result, tf.float32)
    return outputs_array
                    

def same_test_0():
    '''
    Simple test using SAME padding to check out differences between 
    own convolution function and TensorFlow's convolution function.

    NOTE: DO NOT EDIT
    '''
    imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
    imgs = np.reshape(imgs, (1,5,5,1))
    filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
                                dtype=tf.float32,
                                stddev=1e-1),
                                name="filters")
    my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="SAME")
    tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="SAME")
    print("SAME_TEST_0:", "my conv2d:", my_conv[0][0][0], "tf conv2d:", tf_conv[0][0][0].numpy())

def valid_test_0():
    '''
    Simple test using VALID padding to check out differences between 
    own convolution function and TensorFlow's convolution function.

    NOTE: DO NOT EDIT
    '''
    imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
    imgs = np.reshape(imgs, (1,5,5,1))
    filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
                                dtype=tf.float32,
                                stddev=1e-1),
                                name="filters")
    my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
    tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
    print("VALID_TEST_0:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_1():
    '''
    Simple test using VALID padding to check out differences between 
    own convolution function and TensorFlow's convolution function.

    NOTE: DO NOT EDIT
    '''
    imgs = np.array([[3,5,3,3],[5,1,4,5],[2,5,0,1],[3,3,2,1]], dtype=np.float32)
    imgs = np.reshape(imgs, (1,4,4,1))
    filters = tf.Variable(tf.random.truncated_normal([3, 3, 1, 1],
                                dtype=tf.float32,
                                stddev=1e-1),
                                name="filters")
    my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
    tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
    print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_2():
    '''
    Simple test using VALID padding to check out differences between 
    own convolution function and TensorFlow's convolution function.

    NOTE: DO NOT EDIT
    '''
    imgs = np.array([[1,3,2,1],[1,3,3,1],[2,1,1,3],[3,2,3,3]], dtype=np.float32)
    imgs = np.reshape(imgs, (1,4,4,1))
    filters = np.array([[1,2,3],[0,1,0],[2,1,2]]).reshape((3,3,1,1)).astype(np.float32)
    my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
    tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
    print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def main():
    # TODO: Add in any tests you may want to use to view the differences between your and TensorFlow's output

    same_test_0() 
    valid_test_0() 
    valid_test_1() 
    valid_test_2() 

if __name__ == '__main__':
    main()
