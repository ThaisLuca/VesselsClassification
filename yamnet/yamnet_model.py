
import keras
from keras.models import Sequential, Model
from keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import features as features_lib
import keras.metrics
import params

try:
  from tensorflow.compat.v1 import ConfigProto
  from tensorflow.compat.v1 import InteractiveSession

  config = ConfigProto()
  config.gpu_options.allow_growth = True
  session = InteractiveSession(config=config)
except Exception as e:
  print(e)
  print("Not possible to set gpu allow growth")

def _batch_norm(name):
  def _bn_layer(layer_input):
    return layers.BatchNormalization(
      name=name,
      center=params.BATCHNORM_CENTER,
      scale=params.BATCHNORM_SCALE,
      epsilon=params.BATCHNORM_EPSILON)(layer_input)
  return _bn_layer

def get_model():
	waveform = layers.Input(batch_shape=(None, None))
	# Store the intermediate spectrogram features to use in visualization.
	spectrogram = features_lib.waveform_to_log_mel_spectrogram(tf.squeeze(waveform, axis=0), params)
	patches = features_lib.spectrogram_to_patches(spectrogram, params)
	net = layers.Reshape((params.PATCH_FRAMES, params.PATCH_BANDS, 1), input_shape=(params.PATCH_FRAMES, params.PATCH_BANDS))(patches)

	#First Convolutional Layer
	#_conv,          [3, 3], 2,   32)
	name = 'layer{}'.format(1)
	output = layers.Conv2D(name='{}/conv'.format(name),
	                       filters=32,
	                       kernel_size=[3,3],
	                       strides=2,
	                       padding=params.CONV_PADDING,
	                       use_bias=False,
	                       activation=None)(net)
	output = _batch_norm(name='{}/conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/relu'.format(name))(output)

	#Second Layer # (layer_function, kernel, stride, num_filters)
	# _separable_conv, [3, 3], 1,   64)
	name = 'layer{}'.format(2)
	output = layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name),
	                                kernel_size=[3,3],
	                                strides=1,
	                                depth_multiplier=1,
	                                padding=params.CONV_PADDING,
	                                use_bias=False,
	                                activation=None)(output)
	output = _batch_norm(name='{}/depthwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/depthwise_conv/relu'.format(name))(output)
	output = layers.Conv2D(name='{}/pointwise_conv'.format(name),
	                       filters=64,
	                       kernel_size=(1, 1),
	                       strides=1,
	                       padding=params.CONV_PADDING,
	                       use_bias=False,
	                       activation=None)(output)
	output = _batch_norm(name='{}/pointwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/pointwise_conv/relu'.format(name))(output)

	#Third # (layer_function, kernel, stride, num_filters)
	#_separable_conv, [3, 3], 2,  128)
	name = 'layer{}'.format(3)
	output = layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name),
	                                kernel_size=[3,3],
	                                strides=2,
	                                depth_multiplier=1,
	                                padding=params.CONV_PADDING,
	                                use_bias=False,
	                                activation=None)(output)
	output = _batch_norm(name='{}/depthwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/depthwise_conv/relu'.format(name))(output)
	output = layers.Conv2D(name='{}/pointwise_conv'.format(name),
	                       filters=128,
	                       kernel_size=(1, 1),
	                       strides=1,
	                       padding=params.CONV_PADDING,
	                       use_bias=False,
	                       activation=None)(output)
	output = _batch_norm(name='{}/pointwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/pointwise_conv/relu'.format(name))(output)

	#Fourth # (layer_function, kernel, stride, num_filters)
	#_separable_conv, [3, 3], 1,  128)
	name = 'layer{}'.format(4)
	output = layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name),
	                                kernel_size=[3,3],
	                                strides=1,
	                                depth_multiplier=1,
	                                padding=params.CONV_PADDING,
	                                use_bias=False,
	                                activation=None)(output)
	output = _batch_norm(name='{}/depthwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/depthwise_conv/relu'.format(name))(output)
	output = layers.Conv2D(name='{}/pointwise_conv'.format(name),
	                       filters=128,
	                       kernel_size=(1, 1),
	                       strides=1,
	                       padding=params.CONV_PADDING,
	                       use_bias=False,
	                       activation=None)(output)
	output = _batch_norm(name='{}/pointwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/pointwise_conv/relu'.format(name))(output)

	#Fifth # (layer_function, kernel, stride, num_filters)
	#_separable_conv, [3, 3], 2,  256)
	name = 'layer{}'.format(5)
	output = layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name),
	                                kernel_size=[3,3],
	                                strides=2,
	                                depth_multiplier=1,
	                                padding=params.CONV_PADDING,
	                                use_bias=False,
	                                activation=None)(output)
	output = _batch_norm(name='{}/depthwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/depthwise_conv/relu'.format(name))(output)
	output = layers.Conv2D(name='{}/pointwise_conv'.format(name),
	                       filters=256,
	                       kernel_size=(1, 1),
	                       strides=1,
	                       padding=params.CONV_PADDING,
	                       use_bias=False,
	                       activation=None)(output)
	output = _batch_norm(name='{}/pointwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/pointwise_conv/relu'.format(name))(output)

	#Sixth # (layer_function, kernel, stride, num_filters)
	#_separable_conv, [3, 3], 1,  256)
	name = 'layer{}'.format(6)
	output = layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name),
	                                kernel_size=[3,3],
	                                strides=1,
	                                depth_multiplier=1,
	                                padding=params.CONV_PADDING,
	                                use_bias=False,
	                                activation=None)(output)
	output = _batch_norm(name='{}/depthwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/depthwise_conv/relu'.format(name))(output)
	output = layers.Conv2D(name='{}/pointwise_conv'.format(name),
	                       filters=256,
	                       kernel_size=(1, 1),
	                       strides=1,
	                       padding=params.CONV_PADDING,
	                       use_bias=False,
	                       activation=None)(output)
	output = _batch_norm(name='{}/pointwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/pointwise_conv/relu'.format(name))(output)

	#Seventh # (layer_function, kernel, stride, num_filters)
	#_separable_conv, [3, 3], 2,  512)
	name = 'layer{}'.format(7)
	output = layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name),
	                            kernel_size=[3,3],
	                            strides=2,
	                            depth_multiplier=1,
	                            padding=params.CONV_PADDING,
	                            use_bias=False,
	                            activation=None)(output)
	output = _batch_norm(name='{}/depthwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/depthwise_conv/relu'.format(name))(output)
	output = layers.Conv2D(name='{}/pointwise_conv'.format(name),
	                   filters=512,
	                   kernel_size=(1, 1),
	                   strides=1,
	                   padding=params.CONV_PADDING,
	                   use_bias=False,
	                   activation=None)(output)
	output = _batch_norm(name='{}/pointwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/pointwise_conv/relu'.format(name))(output)

	#Eigth # (layer_function, kernel, stride, num_filters)
	#_separable_conv, [3, 3], 1,  512)
	name = 'layer{}'.format(8)
	output = layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name),
	                            kernel_size=[3,3],
	                            strides=1,
	                            depth_multiplier=1,
	                            padding=params.CONV_PADDING,
	                            use_bias=False,
	                            activation=None)(output)
	output = _batch_norm(name='{}/depthwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/depthwise_conv/relu'.format(name))(output)
	output = layers.Conv2D(name='{}/pointwise_conv'.format(name),
	                   filters=512,
	                   kernel_size=(1, 1),
	                   strides=1,
	                   padding=params.CONV_PADDING,
	                   use_bias=False,
	                   activation=None)(output)
	output = _batch_norm(name='{}/pointwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/pointwise_conv/relu'.format(name))(output)

	#Nineth # (layer_function, kernel, stride, num_filters)
	#_separable_conv, [3, 3], 1,  512)
	name = 'layer{}'.format(9)
	output = layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name),
	                                kernel_size=[3,3],
	                                strides=1,
	                                depth_multiplier=1,
	                                padding=params.CONV_PADDING,
	                                use_bias=False,
	                                activation=None)(output)
	output = _batch_norm(name='{}/depthwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/depthwise_conv/relu'.format(name))(output)
	output = layers.Conv2D(name='{}/pointwise_conv'.format(name),
	                       filters=512,
	                       kernel_size=(1, 1),
	                       strides=1,
	                       padding=params.CONV_PADDING,
	                       use_bias=False,
	                       activation=None)(output)
	output = _batch_norm(name='{}/pointwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/pointwise_conv/relu'.format(name))(output)

	#Tenth  # (layer_function, kernel, stride, num_filters)
	#_separable_conv, [3, 3], 1,  512)
	name = 'layer{}'.format(10)
	output = layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name),
	                                kernel_size=[3,3],
	                                strides=1,
	                                depth_multiplier=1,
	                                padding=params.CONV_PADDING,
	                                use_bias=False,
	                                activation=None)(output)
	output = _batch_norm(name='{}/depthwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/depthwise_conv/relu'.format(name))(output)
	output = layers.Conv2D(name='{}/pointwise_conv'.format(name),
	                       filters=512,
	                       kernel_size=(1, 1),
	                       strides=1,
	                       padding=params.CONV_PADDING,
	                       use_bias=False,
	                       activation=None)(output)
	output = _batch_norm(name='{}/pointwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/pointwise_conv/relu'.format(name))(output)

	#Eleventh # (layer_function, kernel, stride, num_filters)
	#_separable_conv, [3, 3], 1,  512)
	name = 'layer{}'.format(11)
	output = layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name),
	                            kernel_size=[3,3],
	                            strides=1,
	                            depth_multiplier=1,
	                            padding=params.CONV_PADDING,
	                            use_bias=False,
	                            activation=None)(output)
	output = _batch_norm(name='{}/depthwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/depthwise_conv/relu'.format(name))(output)
	output = layers.Conv2D(name='{}/pointwise_conv'.format(name),
	                   filters=512,
	                   kernel_size=(1, 1),
	                   strides=1,
	                   padding=params.CONV_PADDING,
	                   use_bias=False,
	                   activation=None)(output)
	output = _batch_norm(name='{}/pointwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/pointwise_conv/relu'.format(name))(output)

	#Twelth # (layer_function, kernel, stride, num_filters)
	#_separable_conv, [3, 3], 1,  512)
	name = 'layer{}'.format(12)
	output = layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name),
	                            kernel_size=[3,3],
	                            strides=1,
	                            depth_multiplier=1,
	                            padding=params.CONV_PADDING,
	                            use_bias=False,
	                            activation=None)(output)
	output = _batch_norm(name='{}/depthwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/depthwise_conv/relu'.format(name))(output)
	output = layers.Conv2D(name='{}/pointwise_conv'.format(name),
	                   filters=512,
	                   kernel_size=(1, 1),
	                   strides=1,
	                   padding=params.CONV_PADDING,
	                   use_bias=False,
	                   activation=None)(output)
	output = _batch_norm(name='{}/pointwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/pointwise_conv/relu'.format(name))(output)

	#Thirteenth  # (layer_function, kernel, stride, num_filters)
	#_separable_conv, [3, 3], 2, 1024)
	name = 'layer{}'.format(13)
	output = layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name),
	                                kernel_size=[3,3],
	                                strides=2,
	                                depth_multiplier=1,
	                                padding=params.CONV_PADDING,
	                                use_bias=False,
	                                activation=None)(output)
	output = _batch_norm(name='{}/depthwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/depthwise_conv/relu'.format(name))(output)
	output = layers.Conv2D(name='{}/pointwise_conv'.format(name),
	                       filters=1024,
	                       kernel_size=(1, 1),
	                       strides=1,
	                       padding=params.CONV_PADDING,
	                       use_bias=False,
	                       activation=None)(output)
	output = _batch_norm(name='{}/pointwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/pointwise_conv/relu'.format(name))(output)

	#Fourteenth # (layer_function, kernel, stride, num_filters)
	#_separable_conv, [3, 3], 1, 1024)
	name = 'layer{}'.format(14)
	output = layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name),
	                                kernel_size=[3,3],
	                                strides=1,
	                                depth_multiplier=1,
	                                padding=params.CONV_PADDING,
	                                use_bias=False,
	                                activation=None)(output)
	output = _batch_norm(name='{}/depthwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/depthwise_conv/relu'.format(name))(output)
	output = layers.Conv2D(name='{}/pointwise_conv'.format(name),
	                       filters=1024,
	                       kernel_size=(1, 1),
	                       strides=1,
	                       padding=params.CONV_PADDING,
	                       use_bias=False,
	                       activation=None)(output)
	output = _batch_norm(name='{}/pointwise_conv/bn'.format(name))(output)
	output = layers.ReLU(name='{}/pointwise_conv/relu'.format(name))(output)

	output = layers.GlobalAveragePooling2D()(output)
	
	output = layers.Dense(100, activation='relu')(output)
	output = layers.Dense(4, activation='softmax')(output)
	#logits = layers.Dense(units=params.NUM_CLASSES, use_bias=True)(net)
	#predictions = layers.Activation(name=params.EXAMPLE_PREDICTIONS_LAYER_NAME, activation=params.CLASSIFIER_ACTIVATION)(logits)

	model = Model(inputs=waveform, outputs=output)
	return model

def fine_tuning():
	prev_model = get_model()
	