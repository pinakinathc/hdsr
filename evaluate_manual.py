# This file contains module to evaluate the accuracy of the model after
# every partial training is over.
# Author: Pinaki Nath Chowdhury <pinakinathc@gmail.com>

from create_model import *
from load_data import *
from save_load_weights import *

import numpy as np
import tensorflow as tf
import keras
from keras import backend as K

def evaluate(model, x_test, y_test):
  """This module evaluates the model on the test data and prints accuracy 
  
  Args:
    model: Object of the Model class(in Keras)
    x_test: Numpy storing the Test Data Images
    y_test: Numpy storing the Test Labels

  Returns:
    None.
  """
  y_pred = model.predict(x_test) # Calculates the CTC Layer
  """This is my experiment"""
  loss = CTC_loss(y_test, y_pred)
  with tf.Session() as sess:
    loss = loss.eval()
  print ("This is the loss function: ", loss)
  print ("The length of the loss function: ", len(loss))
  inp = input('check')
  """ End of Experiment"""

  y_test = y_test[:, 0, :2]
  label_length = tf.multiply(tf.ones(tf.shape(y_test)[0], dtype=tf.int32), 2)

  N = y_pred.shape[1]
  inputs = [] # This variable stored y_pred in a rearranged format for CTC

  for i in range(N):
    inputs.append(y_pred[:, i, :])
  inputs = np.array(inputs)
  inputs = tf.constant(inputs, tf.float32)

  # We calculate the prediction from CTC Layer in a Greedy manner
  y_pred = tf.nn.ctc_greedy_decoder(inputs,
                                    label_length,
                                    merge_repeated=False)
  y_pred = tf.sparse_tensor_to_dense(y_pred[0][0])

  with tf.Session() as sess:
    y_pred = y_pred.eval()

  print ("prediction: ", y_pred[:10])
  print ("target: ", y_test[:10])

  accuracy = np.where(y_pred==y_test, 1, 0)
  accuracy = np.sum(accuracy, axis=1)
  accuracy = np.where(accuracy==2, 1, 0)

  accuracy = accuracy.sum()*1./accuracy.shape[0]

  print ("The accuracy is: ", accuracy)
  print ("Total no. of test data: ", y_test.shape[0])

def main():
  model = create_model()
  model = load_model_weight(model, "model_weights.pkl")
  x_train, y_train, _, _, x_test, y_test = load_data()
  print (x_train.shape)
  print (x_test.shape)

  evaluate(model, x_test, y_test)

main()
