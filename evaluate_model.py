# This file contains module to evaluate the accuracy of the model after
# every partial training is over.
# Author: Pinaki Nath Chowdhury <pinakinathc@gmail.com>

import numpy as np
import tensorflow as tf
import keras
from keras import backend as K

from create_model import *
from load_data import load_data
from save_load_weights import *


def evaluate_process(model, x_test, y_test):
  """This module evaluates the model on the test data and prints accuracy 
  
  Args:
    model: Object of the Model class(in Keras)
    x_test: Numpy storing the Test Data Images
    y_test: Numpy storing the Test Labels

  Returns:
    None.
  """
  y_test = y_test[:, 0, :2]

  import matplotlib.pyplot as plt 
  for i in range(x_test.shape[0]):
    plt.imshow(np.squeeze(x_test[i], axis=2))
    plt.title(str(y_test[i]))
    plt.show(block=False)
    inp = input('check')

  label_length = tf.multiply(tf.ones(tf.shape(y_test)[0], dtype=tf.int32), 2)

  y_pred = model.predict(x_test) # Calculates the CTC Layer

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

  accuracy = np.where(y_pred==y_test, 1, 0)
  accuracy = np.sum(accuracy, axis=1)
  accuracy = np.where(accuracy==2, 1, 0)

  return accuracy.sum()*1./accuracy.shape[0]


def evaluate(model=None, x_test=None, y_test=None):
  if model is None:
    model = create_model()
    model = load_model_weight(model, "model_weights.pkl")
  if x_test is None or y_test is None:
    x_train, y_train, _, _, x_test, y_test = load_data()
    out = evaluate_process(model, x_train, y_train)
    print ("The accuracy is : ", out)
    print ("The total number of sample: ", y_test.shape[0])

# evaluate()