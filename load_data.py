# This file loads the data of connected component in the format which
# this program needs. Since, keras demands y_true and y_pred to be of same
# dimension, but in this case, output from model is the CTC layer
# where y_true is something like [[2,3], [3,4], [4,5].....]
# hence we shall use dummy padding for y_true.
# Author: Pinaki Nath Chowdhury <pinakinathc@gmail.com>

from glob import glob
import numpy as np
import cv2

def load_data():
  """This module loads data from the specified path. In our case the images 
  are in tif format, hence this function is written accordingly. If your data 
  is of different format, change the extension hardcoded in glob path.

  Args:
    None. (We can later include arguments for setting the path using FLAGS

  Returns:
    x_train: training data of shape [batch_size*4/5., 75, 75, 1]
    y_train: training labels of shape [batch_size*4/5., 9, 11), due to padding
    x_valid: Empty Array, as we did not use validation set for this dataset
    y_valid: Empty Array, as we did not use validation set for this dataset
    x_test: testing data of shape [batch_size*1/5., 75, 75, 1]
    y_test: testing Labels of shape [batch_size*1/5., 75, 75, 1]
  """
  FILE_PATH = '../data_hdsr/Bangla_connected_numerals_00_99/*/*.tif'
  image_files = glob(FILE_PATH)

  data = [] # for storing all data
  labels = [] # for storing all labels

  for image_file in image_files:
    im = cv2.imread(image_file, 0)
    im = cv2.resize(im, (75, 75))
    im = np.transpose(im) # This is utilised in Dimensional Transpose Layer
    im = np.where(im>200, 0, 1) # Applying a threshold of 200 (HyperParameter)
    im = im[:, :, np.newaxis] # Making im compatible for Keras
    data.append(im)
    labels.append(image_file.split('/')[-2]) # Pathnames have the bangla digit

  N = len(labels) # Getting total number of data/labels

  labels = [[int(x[0]), int(x[1])] for x in labels] # making labels compatible 
  labels = np.array(labels, dtype=np.int32)
  zeros = np.zeros((N, 9, 11), dtype=np.int32)
  zeros[:, 0, :2] = labels
  labels = zeros # Dummy padding of y_true to make it compatible with keras
  
  data = np.array(data, dtype=np.float32)

  x_train = np.concatenate((data[1::4], data[2::4], data[3::4]), axis=0)
  y_train = np.concatenate((labels[1::4], labels[2::4], labels[3::4]), axis=0)

  x_valid = []
  y_valid = []

  x_test = data[0::4]
  y_test = labels[0::4]

  return (x_train, y_train, x_valid, y_valid, x_test, y_test)
