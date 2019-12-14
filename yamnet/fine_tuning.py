# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys

import numpy as np
import resampy
import soundfile as sf

import params
import yamnet as yamnet_model

import tensorflow as tf
from sklearn.ensemble import AdaBoostClassifier

import keras
from keras.models import Sequential, Model
from keras import layers
from keras import backend as K
from tensorflow.keras import optimizers

import utils as util
import plot as plt

import features as features_lib

import keras.metrics
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

try:
  from tensorflow.compat.v1 import ConfigProto
  from tensorflow.compat.v1 import InteractiveSession

  config = ConfigProto()
  config.gpu_options.allow_growth = True
  session = InteractiveSession(config=config)
except Exception as e:
  print(e)
  print("Not possible to set gpu allow growth")

classes = [0,1,2,3]
labels_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

def _batch_norm(name):
  def _bn_layer(layer_input):
    return layers.BatchNormalization(
      name=name,
      center=params.BATCHNORM_CENTER,
      scale=params.BATCHNORM_SCALE,
      epsilon=params.BATCHNORM_EPSILON)(layer_input)
  return _bn_layer


def _conv(name, kernel, stride, filters):
  def _conv_layer(layer_input):
    output = layers.Conv2D(name='{}/conv'.format(name),
                           filters=filters,
                           kernel_size=kernel,
                           strides=stride,
                           padding=params.CONV_PADDING,
                           use_bias=False,
                           activation=None)(layer_input)
    output = _batch_norm(name='{}/conv/bn'.format(name))(output)
    output = layers.ReLU(name='{}/relu'.format(name))(output)
    return output
  return _conv_layer


def _separable_conv(name, kernel, stride, filters):
  def _separable_conv_layer(layer_input):
    output = layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name),
                                    kernel_size=kernel,
                                    strides=stride,
                                    depth_multiplier=1,
                                    padding=params.CONV_PADDING,
                                    use_bias=False,
                                    activation=None)(layer_input)
    output = _batch_norm(name='{}/depthwise_conv/bn'.format(name))(output)
    output = layers.ReLU(name='{}/depthwise_conv/relu'.format(name))(output)
    output = layers.Conv2D(name='{}/pointwise_conv'.format(name),
                           filters=filters,
                           kernel_size=(1, 1),
                           strides=1,
                           padding=params.CONV_PADDING,
                           use_bias=False,
                           activation=None)(output)
    output = _batch_norm(name='{}/pointwise_conv/bn'.format(name))(output)
    output = layers.ReLU(name='{}/pointwise_conv/relu'.format(name))(output)
    return output
  return _separable_conv_layer


_YAMNET_LAYER_DEFS = [
    # (layer_function, kernel, stride, num_filters)
    (_conv,          [3, 3], 2,   32),
    (_separable_conv, [3, 3], 1,   64),
    (_separable_conv, [3, 3], 2,  128),
    (_separable_conv, [3, 3], 1,  128),
    (_separable_conv, [3, 3], 2,  256),
    (_separable_conv, [3, 3], 1,  256),
    (_separable_conv, [3, 3], 2,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 2, 1024),
    (_separable_conv, [3, 3], 1, 1024)
]

def get_model():
	waveform = layers.Input(batch_shape=(None, None))
	# Store the intermediate spectrogram features to use in visualization.
	spectrogram = features_lib.waveform_to_log_mel_spectrogram(tf.squeeze(waveform, axis=0), params)
	patches = features_lib.spectrogram_to_patches(spectrogram, params)
	net = layers.Reshape((params.PATCH_FRAMES, params.PATCH_BANDS, 1),input_shape=(params.PATCH_FRAMES, params.PATCH_BANDS))(patches)
	for (i, (layer_fun, kernel, stride, filters)) in enumerate(_YAMNET_LAYER_DEFS):
		net = layer_fun('layer{}'.format(i + 1), kernel, stride, filters)(net)
	net = layers.GlobalAveragePooling2D()(net)
	logits = layers.Dense(units=params.NUM_CLASSES, use_bias=True)(net)
	predictions = layers.Activation(name=params.EXAMPLE_PREDICTIONS_LAYER_NAME, activation=params.CLASSIFIER_ACTIVATION)(logits)
	frames_model = Model(name='yamnet_frames', inputs=waveform, outputs=[predictions, spectrogram])
	return frames_model
	

def main():
	accuracy_train_scores = []
	accuracy_validation_scores = []
	precision_train_scores = []
	precision_validation_scores = []
	recall_train_scores = []
	recall_validation_scores = []

	accuracy_test_scores = []
	precision_test_scores = []
	recall_test_scores = []

	train_error = []
	validation_error = []
	test_error = []

	best_loss = 0
	best_model = None

	epochs=1000

	f_X_train = 0
	f_y_train = 1
	f_X_val = 2
	f_y_val = 3

	all_files = util.get_files_path()[4:]

	# Build network
	yamnet = yamnet_model.yamnet_frames_model(params)
	print(yamnet.summary())
	yamnet.load_weights('yamnet.h5')

	sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	yamnet.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

	waveforms = {}
	labels = []
	for file in all_files:
		 # Decode the WAV file.
		wav_data, sr = sf.read(file, dtype=np.int16)
		assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
		waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
		label = file.split('\\')[-2][-1]
		label = labels_dict[label]

		# Convert to mono and the sample rate expected by YAMNet.
		if len(waveform.shape) > 1:
			waveform = np.mean(waveform, axis=1)
		if sr != params.SAMPLE_RATE:
			waveform = resampy.resample(waveform, sr, params.SAMPLE_RATE)

		avg = len(waveform) / float(5)
		last = 0.0
		waveforms[label] = []

		while last < len(waveform):
			waveforms[label].append(waveform[int(last):int(last + avg)])
			labels.append(label)
			last += avg

	folds, X_test, Y_test = util.build_folds_test(waveforms, labels, classes)

	for fold in folds:
		X = np.array(fold[f_X_train])
		Y = np.array(fold[f_y_train])
		Y = to_categorical(Y)

		X_V = np.array(fold[f_X_val])
		Y_V = np.array(fold[f_y_val])
		Y_V = to_categorical(Y_V)

		history = yamnet.fit(X, Y, epochs=epochs, batch_size=32, validation_data=(X_V, Y_V)) #, callbacks=[callback])

		# Save train and validation accuracy
		accuracy_train_scores.append(history.history['accuracy'])
		accuracy_validation_scores.append(history.history['val_accuracy'])

		# Save train and validation precision
		precision_train_scores.append(history.history['precision_' + str(count)])
		precision_validation_scores.append(history.history['val_precision_' + str(count)])

		# Save train and validation recall
		recall_train_scores.append(history.history['recall_' + str(count)])
		recall_validation_scores.append(history.history['val_recall_' + str(count)])

		# Save train and validation error
		train_error.append(history.history['loss'])
		validation_error.append(history.history['val_loss'])

		# Evaluate on test set
		score = yamnet.evaluate(X_T, Y_T)

		if count == 1 or count == 2:
			best_loss = history.history['loss'][-1]
			losses = history.history['loss']
			val_losses = history.history['val_loss']

		if best_loss > history.history['val_loss'][-1]:
			best_loss = history.history['val_loss'][-1]
			losses = history.history['loss']
			val_losses = history.history['val_loss']

		print("Fold %d:" % count)
		#print("Training accuracy: %.2f%%" % (history.history['accuracy'][-1]*100))
		#print("Testing accuracy: %.2f%%" % (history.history['val_accuracy'][-1]*100))
		count += 1
		
	plt.plot(accuracy_train_scores, accuracy_validation_scores, epochs, "Treinamento", "Validação", "Acurácia")
	plt.plot(precision_train_scores, precision_validation_scores, epochs, "Treinamento", "Validação", "Precisão")
	plt.plot(recall_train_scores, recall_validation_scores, epochs, "Treinamento", "Validação", "Recall")
	plt.plot_loss(losses, val_losses, epochs)

	util.save_to_file(accuracy_train_scores, accuracy_validation_scores, precision_train_scores, precision_validation_scores, recall_train_scores, recall_validation_scores, accuracy_test_scores, precision_test_scores, recall_test_scores, train_error, validation_error, test_error)
	return
main()