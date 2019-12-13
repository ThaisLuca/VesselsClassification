
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys

import numpy as np
import resampy
import soundfile as sf

import params
import yamnet as yamnet_model

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras import backend as K

import utils as util
import plot as plt

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

# for I in range(len(audio)/tamanho_entrada):
#	inputs.append(audio[I*tamanho_entrada:I*(tamanho_entrada+1)])

def get_model():
	# Build MLP
	visible1 = Input(shape=(1024,))
	hidden1 = Dense(100, activation='relu')(visible1)
	#hidden2 = Dense(1024, activation='relu')(hidden1)
	#hidden3 = Dense(1024, activation='relu')(hidden2)
	#output = Dense(4, activation='softmax')(hidden3)
	output = Dense(4, activation='softmax')(hidden1)
	model = Model(inputs=visible1, outputs=output)
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

	return model

def main():
	accuracy_train_scores = []
	accuracy_validation_scores = []
	precision_train_scores = []
	precision_validation_scores = []

	accuracy_test_scores = []
	precision_test_scores = []
	real_test_error = []

	train_error = []
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
	yamnet.load_weights('yamnet.h5')

	get_feature_layer_output = K.function([yamnet.layers[0].input], [yamnet.layers[-3].output])

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

	count = 1
	for fold in folds:
		X = []
		X_V = []
		Y = []
		Y_V = []
		for x, y in zip(fold[f_X_train], fold[f_y_train]):
			a = get_feature_layer_output([np.reshape(x, [1, -1])])[0]
			for i in a:
				X.append(i)
				Y.append(y)

		for x, y in zip(fold[f_X_val], fold[f_y_val]):
			v = get_feature_layer_output([np.reshape(x, [1, -1])])[0]
			for i in v:
				X_V.append(i)
				Y_V.append(y)

		model = get_model()

		X = np.array(X)
		Y = np.array(Y)

		Y = to_categorical(Y)

		X_V = np.array(X_V)
		Y_V = np.array(Y_V)

		Y_V = to_categorical(Y_V)

		callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
		history = model.fit(X, Y, epochs=epochs, batch_size=32, validation_data=(X_V, Y_V)) #, callbacks=[callback])
		score = loaded_model.evaluate(X, Y, verbose=0)
		
		accuracy_train_scores.append(history.history['accuracy'])
		accuracy_validation_scores.append(history.history['val_accuracy'])

		precision_train_scores.append(history.history['precision_' + str(count)])
		precision_validation_scores.append(history.history['val_precision_' + str(count)])

		train_error.append(history.history['loss'])
		test_error.append(history.history['val_loss'])

		if count == 1:
			best_loss = history.history['loss'][-1]
			losses = history.history['loss']
			val_losses = history.history['val_loss']
			model.save_weights("model.h5")

		if best_loss > history.history['val_loss'][-1]:
			best_loss = history.history['val_loss'][-1]
			losses = history.history['loss']
			val_losses = history.history['val_loss']
			model.save_weights("model.h5")


		print("Fold %d:" % count)
		#print("Training accuracy: %.2f%%" % (history.history['accuracy'][-1]*100))
		#print("Testing accuracy: %.2f%%" % (history.history['val_accuracy'][-1]*100))
		count += 1
		
	plt.plot(accuracy_train_scores, accuracy_validation_scores, epochs, "Treinamento", "Validação", "Acurácia")
	plt.plot(precision_train_scores, precision_validation_scores, epochs, "Treinamento", "Validação", "Precisão")
	#plt.plot(train_error, test_error, epochs, "Treinamento", "Validação", "Erro")
	plt.plot_loss(losses, val_losses, epochs)

	util.save_to_file(accuracy_train_scores, accuracy_validation_scores, precision_train_scores, precision_validation_scores, train_error, test_error)

	loaded_model = get_model()
	best_model = loaded_model.load_weights("model.h5")
	score = loaded_model.evaluate(X, Y, verbose=0)
	print(loaded_model.metrics_names)
	print(score)

	return
main()