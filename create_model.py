# This file contains code to define the model for hdsr
# Author: Pinaki Nath Chowdhury <pinakinathc@gmail.com>

# from evaluate_model import *

import tensorflow as tf
import keras
from keras.layers import (Input, Conv2D, MaxPool2D, Concatenate,
                          BatchNormalization, Dropout, Add, Activation)
from keras.models import Model
from keras.engine.topology import Layer
from keras import backend as K
import numpy as np

def conv_block(inputs):
  """This module creates Dense Block units with growth rate 4 to be used by
  dense block module for creation of the Dense Block as mentioned in the paper

  Args:
    inputs: input conv layer with shape [None, None, <no. of feature maps>

  Returns:
    conv: output of Dense Block having the shape: [None, None, 4]
  """
  bn = BatchNormalization()(inputs) # Normalise the inputs to Dense Block
  relu = Activation('relu')(bn) # Apply relu activation for non-linearity
  conv = Conv2D(4, (3,3), padding='same', activation='relu')(relu)
  return conv

def dense_block(inputs):
  """This module uses conv_block units for creating a 5-layer Dense block.
  The Dense Block always returns a 2D Conv Layer with feature maps=4

  Args:
    inputs: input conv layer with shape [None, None, 128]

  Returns:
    conv_5: output from a Dense block with shape [None, None, 4]
  """
  conv_1 = conv_block(inputs)
  concat_1 = Concatenate()([inputs, conv_1])
  
  conv_2 = conv_block(concat_1)
  concat_2 = Concatenate()([concat_1, conv_2])

  conv_3 = conv_block(concat_2)
  concat_3 = Concatenate()([concat_2, conv_3])

  conv_4 = conv_block(concat_3)
  concat_4 = Concatenate()([concat_3, conv_4])

  conv_5 = conv_block(concat_4)
  return conv_5

def transition_layer(inputs, add_drop=True):
  """This module adds the transition layer as mentioned in the hdsr paper

  Args:
    inputs: The input to transition layer is the output from a Dense Block
            with shape [None, None, 4]
    add_drop: This is a boolean variable which if False will not apply dropout

  Returns:
    pool: output from transition layer with shape [None, None, 4]
  """
  bn = BatchNormalization()(inputs)
  relu = Activation('relu')(bn)
  conv = Conv2D(128, (1,1), padding='same', activation='relu')(relu)
  if add_drop:
    drop = Dropout(0.2)(conv)
    pool = MaxPool2D(pool_size=(2,2), strides=2)(drop)
  else:
    pool = MaxPool2D(pool_size=(2,2), strides=2)(conv)
  return pool

class DimTranspose(Layer):
  """This Layer carries our the Feature Dimention Transposition as mentioned
  in the Section II B of the hdsr paper.
  """
  def __init__(self, output_dim, **kwargs):
    """Args:
        output_dim: This is an input from the user specifying the number
                    of alphabets in the CTC Layer. For 0-9 we shall have
                    output_dim = 10+1 = 11
    """
    self.output_dim = output_dim
    super(DimTranspose, self).__init__(**kwargs)

  def build(self, input_shape):
    """Args:
        input_shape: This is the input shape of the layer coming input to this
                      Dimension Transpose Layer
    """
    (a, b, c, d) = input_shape
    self.kernel = self.add_weight(name='kernel',
                                  shape=(c*d, self.output_dim),
                                  initializer='uniform',
                                  trainable=True)
    super(DimTranspose, self).build(input_shape)

  def call(self, x):
    """Args:
        x: This is input layer that comes to the Dimension Transpose module
    """
    N = x.shape[0]
    tmp_x = []
    for i in range(x.shape[-1]):
      tmp_x.append(x[:, :, :, i])
    x = K.concatenate(tmp_x, axis=2)
    return K.dot(x, self.kernel)

  def compute_output_shape(self, input_shape):
    [a, b, c, d] = input_shape
    return (a, b, self.output_dim)

def CTC_loss(y_true, y_pred):
  """This is the Objective function which calculates the CTC Loss
  Args:
    y_true: This is the output matrix storing Final output. The format is 
            NOT one-hot encoded i.e. if the label is 23 it stores [2,3]
            In tensorflow.nn.ctc loss, a sparse tranformation is Labels.
    y_pred: This is what the model predicts after processing the input.
            The information is stored in the fom of one-hot encoding
  """
  y_true = y_true[:, 0, :2]
  y_true = tf.cast(y_true, dtype=tf.int32)
  label_length = tf.multiply(tf.ones(tf.shape(y_true)[0], dtype=tf.int32), 2)
  where = tf.ones(tf.shape(y_true))
  indices = tf.where(where)
  values = tf.gather_nd(y_true, indices)
  sparse = tf.SparseTensor(indices, values,
                            tf.cast(tf.shape(y_true), dtype=tf.int64))
  # Finally calculating the CTC Loss
  loss = tf.nn.ctc_loss(sparse, y_pred, label_length, time_major=False, 
                                                ctc_merge_repeated=False)
  case_true = tf.multiply(tf.ones(tf.shape(loss)), 100)
  loss = tf.where(tf.is_inf(loss), case_true, loss)
  return loss

def create_model():
  """This module calls various other functions to create the model
  and also sets the hyper-parameters.
  """
  inputs = Input(shape=(75, 75, 1), name='input_layer')
  conv = Conv2D(128, (3,3), padding='same',
                activation='relu', name='Convolution_layer')(inputs)

  dense_1 = dense_block(conv)
  transition_1 = transition_layer(dense_1)

  dense_2 = dense_block(transition_1)
  transition_2 = transition_layer(dense_2)

  dense_3 = dense_block(transition_2)
  transition_3 = transition_layer(dense_3, add_drop=False)
   
  dim = DimTranspose(output_dim=11)(transition_3)

  model = Model(inputs=inputs, outputs=dim)

  # Setting the hyper parameters
  opt = keras.optimizers.Adadelta() # As the paper suggested to use Adadelta
  model.compile(loss=CTC_loss, optimizer=opt) 

  model.summary() # To print the final model architecture
  return model
