# This file trains the model
# Author: Pinaki Nath Chowdhury <pinakinathc@gmail.com>

from create_model import *
from load_data import *
from save_load_weights import *
from evaluate_model import *

import tensorflow as tf
import keras
from keras.callbacks import TensorBoard

# To utilize only the part of the GPU needed for training,
# instead of reserving the entire Video Memory available
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

JUMP = 1 # No. of epochs after which we save the model


def trainer(model, x_train, y_train, x_valid, y_valid, initial_epoch):
  """This model only focuses on training using the data supplied
  Args:
    x_train: Training Data. (DType: Numpy Array)
    y_train: Testing Labels. (DType: Numpy Array)
    x_valid: Validation Data (empty array in our case)
    y_valid: Validation Labels (empty array in our case)
  
  Returns:
    model: An Object of the Class Model(in Keras) which is partially trained
  """
  global JUMP
  tnsbrd = TensorBoard(log_dir='./logs')

  model.fit(x_train,
            y_train,
            batch_size=128, # As specified in the paper
            initial_epoch=initial_epoch,
            epochs=initial_epoch+JUMP)
  return model


def train(starting_epoch=0):
  """This module sets all hyper-parametes of the model and optimisers,
  creates an instance of the Keras Model class and partially trains it using
  the trainer module.

  Args:
    starting_epoch: Specifies at which epoch do we want to start. (Integer)

  Returns:
    None
  """
  global JUMP

  model = create_model() # Creates an object of Model class

  if starting_epoch: # In case starting_epoch is Non-zero
    model = load_model_weight(model, 'model_weights.pkl')
  
  (x_train, y_train, x_valid, y_valid, x_test, y_test) = load_data()
  print ("Training Data Shape: ", x_train.shape)
  print ("Testing Data Shape: ", x_test.shape)

  for i in range(starting_epoch, 300000, JUMP): # The paper trained to 300000 
    model = trainer(model,
                    x_train,
                    y_train,
                    x_valid,
                    y_valid,
                    initial_epoch=i)
    #try:
    #  save_model_weight(model, 'model_weights.pkl')
    #except:
    #  print ("Cannot save the model")
    evaluate(model=model, x_test=x_test, y_test=y_test)

train(870) # This line will fire-up the training
