
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys

import numpy as np
import resampy
import soundfile as sf

import params
import yamnet as yamnet_model

from sklearn.ensemble import AdaBoostClassifier

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras import backend as K
from keras import optimizers

import utils as util
import plot as plt

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
boost = False

# for I in range(len(audio)/tamanho_entrada):
#	inputs.append(audio[I*tamanho_entrada:I*(tamanho_entrada+1)])

def get_model():
	# Build MLP
	visible1 = Input(shape=(1024,))
	hidden1 = Dense(100, activation='relu')(visible1)
	output = Dense(4, activation='softmax')(hidden1)
	model = Model(inputs=visible1, outputs=output)
	#adam = optimizers.Adam(learning_rate=0.001)
	sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

	return model

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
	yamnet.load_weights('yamnet.h5')

	get_feature_layer_output = K.function([yamnet.layers[0].input], [yamnet.layers[-3].output])

	#model = get_model()
	#ann_estimator = KerasClassifier(build_fn=model, epochs=epochs, batch_size=10, verbose=0)
	#boosted_ann = AdaBoostClassifier(base_estimator=ann_estimator)

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
	
	X_T = []
	Y_T = []
	for x, y in zip(X_test, Y_test):
			a = get_feature_layer_output([np.reshape(x, [1, -1])])[0]
			for i in a:
				X_T.append(i)
				Y_T.append(y)

	X_T = np.array(X_T)
	Y_T = np.array(Y_T)

	Y_T = to_categorical(Y_T)

	count = 2
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

		if(boost):
			# Using AdaBoost
			boosted_ann.fit(X, Y)
			print(boosted_ann.score(X_V, Y_V))
		else:
			# Train and Validation
			#callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
			history = model.fit(X, Y, epochs=epochs, batch_size=32, validation_data=(X_V, Y_V)) #, callbacks=[callback])
		
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

		if(boost):
			# Evaluate on test set using boost
			score = boosted_ann.predict(X_T, Y_T)
			print("Score")
			print(score)
		else:
			score = model.evaluate(X_T, Y_T)

		# Save error, accuracy and precision
		test_error.append(score[0])
		accuracy_test_scores.append(score[1])
		precision_test_scores.append(score[2])
		recall_test_scores.append(score[3])

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
		
	#plt.plot(accuracy_train_scores, accuracy_validation_scores, epochs, "Treinamento", "Validação", "Acurácia")
	#plt.plot(precision_train_scores, precision_validation_scores, epochs, "Treinamento", "Validação", "Precisão")
	#plt.plot(recall_train_scores, recall_validation_scores, epochs, "Treinamento", "Validação", "Recall")
	#plt.plot_loss(losses, val_losses, epochs)

	util.save_to_file(accuracy_train_scores, accuracy_validation_scores, precision_train_scores, precision_validation_scores, recall_train_scores, recall_validation_scores, accuracy_test_scores, precision_test_scores, recall_test_scores, train_error, validation_error, test_error)
	return
main()