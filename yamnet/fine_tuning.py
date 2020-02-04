# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys

import params
import yamnet as yamnet_model

import numpy as np
import resampy
import soundfile as sf

import tensorflow as tf
from sklearn.ensemble import AdaBoostClassifier

from tensorflow.keras import Model, layers
from tensorflow.keras import optimizers
from keras.models import load_model

import utils as util
import plot as plt

import features as features_lib

import keras.metrics
from keras.utils import to_categorical

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

def get_model():
	
	# Build network
	yamnet = yamnet_model.yamnet_frames_model(params)
	
	yamnet.load_weights('yamnet.h5', by_name=True)
	sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	yamnet.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

	return yamnet

def main():
	
	EPOCHS=1000

	f_X_train = 0
	f_y_train = 1
	f_X_val = 2
	f_y_val = 3

	# General log variables
	accuracy_train_scores, accuracy_validation_scores, accuracy_test_scores = [], [], []
	precision_train_scores, precision_validation_scores, precision_test_scores = [], [], []
	recall_train_scores, recall_validation_scores, recall_test_scores = [], [], []
	train_error, validation_error, test_error = [], [], []

	# Log variables for each class
	accuracy_train_per_class, accuracy_validation_per_class, accuracy_test_per_class = {}, {}, {}
	precision_train_per_class, precision_validation_per_class, precision_test_per_class = {}, {}, {}
	recall_train_per_class, recall_validation_per_class, recall_test_per_class = {}, {}, {}
	f1_score_train_per_class, f1_score_validation_per_class, f1_score_test_per_class = {}, {}, {}

	# Initialize dictionaries for each metric 
	accuracy_train_per_class, accuracy_validation_per_class, accuracy_test_per_class = util.initialize_metrics_per_class(classes, accuracy_train_per_class, accuracy_validation_per_class, accuracy_test_per_class)
	precision_train_per_class, precision_validation_per_class, precision_test_per_class = util.initialize_metrics_per_class(classes, precision_train_per_class, precision_validation_per_class, precision_test_per_class)
	recall_train_per_class, recall_validation_per_class, recall_test_per_class = util.initialize_metrics_per_class(classes, recall_train_per_class, recall_validation_per_class, recall_test_per_class)
	f1_score_train_per_class, f1_score_validation_per_class, f1_score_test_per_class = util.initialize_metrics_per_class(classes, f1_score_train_per_class, f1_score_validation_per_class, f1_score_test_per_class)

	all_files = util.get_files_path()[4:]

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
	X_T = np.array(X_test)
	Y_T = np.array(Y_test)
	Y_T = to_categorical(Y_T)

	for fold in folds:
		X = np.array(fold[f_X_train])
		Y = np.array(fold[f_y_train])
		Y = to_categorical(Y)

		X_V = np.array(fold[f_X_val])
		Y_V = np.array(fold[f_y_val])
		Y_V = to_categorical(Y_V)

		#TODO: ver essa conversão antes de colocar a rede pra treinar
		print(len(X), len(Y), len(X_V), len(Y_V), len(X_T), len(Y_T))
		model = get_model()
		print(model.summary())
		history, _ = model.fit(X, Y, epochs=EPOCHS, batch_size=32, validation_data=(X_V, Y_V)) #, callbacks=[callback])

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
		score = model.evaluate(X_T, Y_T)

		print("Fold %d:" % count)
		#print("Training accuracy: %.2f%%" % (history.history['accuracy'][-1]*100))
		#print("Testing accuracy: %.2f%%" % (history.history['val_accuracy'][-1]*100))
		count += 1
		
	plt.plot(accuracy_train_scores, accuracy_validation_scores, epochs, "Treinamento", "Validação", "Acurácia")
	plt.plot(precision_train_scores, precision_validation_scores, epochs, "Treinamento", "Validação", "Precisão")
	plt.plot(recall_train_scores, recall_validation_scores, epochs, "Treinamento", "Validação", "Recall")

	#TODO: fix loss plot
	#plt.plot_loss(losses, val_losses, epochs)

	util.save_to_file(accuracy_train_scores, accuracy_validation_scores, precision_train_scores, precision_validation_scores, recall_train_scores, recall_validation_scores, accuracy_test_scores, precision_test_scores, recall_test_scores, train_error, validation_error, test_error)
	return
main()