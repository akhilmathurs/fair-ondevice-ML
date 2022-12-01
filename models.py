import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, ReLU,  GlobalAveragePooling2D, ELU
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Sequential
import tensorflow_addons as tfa
import tensorflow_model_optimization as tfmot
import pdb


#2D model. Works for a fixed input size.
def speech_commands_model_2D(weights=None, num_classes=35, is_training=True):
	inputs = tf.keras.Input(shape=(40,100,1), name='spec')
	x = Conv2D(filters = 64, kernel_size = (8,20),  activation='relu',kernel_regularizer = l2(1e-5))(inputs)
	x = MaxPooling2D(pool_size = (2,2))(x)
	x = Dropout(0.5)(x, training=is_training)

	# x = Conv2D(filters = 128, kernel_size = (4,10), activation='relu',kernel_regularizer = l2(1e-5))(x)
	# x = MaxPooling2D(pool_size = (1,4))(x)
	# x = Dropout(0.5)(x, training=is_training)

	x = Conv2D(filters = 512, kernel_size = (2,2), kernel_regularizer = l2(1e-5))(x)
	x = ReLU()(x)
	x = Dropout(0.5)(x, training=is_training) #remove it for deeper_20

	flattened = Flatten()(x)
	# x = Dense(256, kernel_regularizer = l2(1e-5), activation='relu')(flattened)	
	x = Dense(128, kernel_regularizer = l2(1e-5), activation='relu')(flattened)
	outputs = Dense(num_classes, kernel_regularizer = l2(1e-5))(x)

	model = tf.keras.Model(inputs=inputs, outputs=outputs, name='model')
	if weights is not None:
		model.load_weights(weights)

	return model


def sc_model_cnn(args, saved_model=None, prune=False):
  """
  Builds a standard convolutional model.
  This is roughly the network labeled as 'cnn-trad-fpool3' in the 'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper: http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

  This is a Keras and TFv2 implementation from the tensorflow speech_commands example: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/models.py

  COMMENTS:
  As per authors, architecture is infeasible for small-footprint KWS.

  """

  if saved_model is not None:
      model = models.load_model(saved_model)

  else:
    model = Sequential([
      layers.Conv2D(64, (8,20), activation='relu', kernel_regularizer=l2(1e-5), padding='same'),
      layers.Dropout(0.5),
      layers.MaxPool2D((2,2)),
      layers.Conv2D(64, (4,10), activation='relu', kernel_regularizer=l2(1e-5)),
      layers.Dropout(0.5),
      layers.GlobalAveragePooling2D(data_format='channels_last'),
      layers.Dense(128, kernel_regularizer=l2(1e-5)),
      layers.Dense(args['num_classes'], kernel_regularizer=l2(1e-5)),
    ])

  return model



def sc_model_low_latency_cnn(args, saved_model=None, prune=False):
  """
  Builds a convolutional model with low compute requirements. 
  This is roughly the network labeled as 'cnn-one-fstride4' in the 'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper: http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

  This is a Keras and TFv2 implementation from the tensorflow speech_commands example: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/models.py

  """
  
  if saved_model is not None:
      model = models.load_model(saved_model)
  
  else:
    model = Sequential([
      layers.Conv2D(186, (8,32), activation='relu', kernel_regularizer=l2(1e-5), padding='same'),
      layers.Dropout(0.5),
      layers.GlobalAveragePooling2D(data_format='channels_last'),
      layers.Dense(128, kernel_regularizer=l2(1e-5)),
      layers.Dense(128, kernel_regularizer=l2(1e-5)),
      layers.Dropout(0.5),
      layers.Dense(args['num_classes'], kernel_regularizer=l2(1e-5)),
    ])

  return model



def sc_model_low_latency_svdf(args, weights=None, is_training=True):
  """
  Builds a SVDF model with low compute requirements. 
  This is based in the topology presented in the 'Compressing Deep Neural Networks using a Rank-Constrained Topology' paper:
  https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43813.pdf

  This is a Keras and TFv2 implementation from the tensorflow speech_commands example: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/models.py

  """

  model = Sequential([
    layers.InputLayer(),
    layers.Conv1D(186, (8,xx), activation='relu', kernel_regularizer=l2(1e-5)),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(128, kernel_regularizer=l2(1e-5)),
    layers.Dense(128, kernel_regularizer=l2(1e-5)),
    layers.Dropout(0.5),
    layers.Dense(args['num_classes'], kernel_regularizer=l2(1e-5)),
  ])

  return model

