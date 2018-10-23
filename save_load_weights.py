# We are trying to avoid the general saving of the model as done by Keras 
# The reason for this is, the server where we are training does not support H5
# Author: Pinaki Nath Chowdhury <pinakinathc@gmail.com>

import pickle as pkl

def save_model_weight(model, filename):
  """This function saves the model weights in a pickle file
  Args:
    model: Object of Class Model (in Keras). Has the model of hdsr
    filename: String. Provides Pathname to store Model.
  Returns:
    None.
  """
  weights = model.get_weights()
  with open (filename, 'wb') as obj:
    pkl.dump(weights, obj, pkl.HIGHEST_PROTOCOL)

def load_model_weight(model, filename):
  """This function loads the model weights from a pickle file
  Args:
    model: Object of Class Model(in Keras).
    filename: String. PathName of the stored pickle file
  """
  with open(filename, 'rb') as obj:
      weights = pkl.load(obj)

  model.set_weights(weights)
  return model
