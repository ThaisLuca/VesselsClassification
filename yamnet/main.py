
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys

import numpy as np
import resampy
import soundfile as sf

import params
import yamnet as yamnet_model

import os
from os import listdir
from os.path import isfile, join

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras import backend as K

from matplotlib import pyplot as plt

import keras.metrics
from keras.utils import to_categorical
from keras import regularizers

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

def get_files_path():
	path = os.getcwd() + '\dataset_acoustic_lane_4_classes'
	directories = [x[0] for x in os.walk(path)]

	files = []
	for dire in directories:
		all_files = listdir(dire)
		for file in all_files:
			whole_path = join(dire, file)
			files.append(whole_path)
	return files

def pre_processing(filenames):
	inputs = []
	labels = []
	for file in filenames:
		label = file.split('\\')[-2][-1]
		wav_data, sr = sf.read(file, dtype=np.int16)
		for I in range(int(round(len(wav_data)/params.PATCH_FRAMES))):
			inputs.append(wav_data[I*params.PATCH_FRAMES:I*(params.PATCH_FRAMES+1)])
			labels.append(label)
	return inputs, labels

def build_folds_test(waveforms, labels):
	X_test = []
	y_test = []
	for c in classes:
		X_test.append(waveforms[c][-1])
		y_test.append(c)

	fold_1 = []
	fold_2 = []
	fold_3 = []
	fold_4 = []
	fold_5 = []

	#Fold 1
	X_val = []
	y_val = []
	X_train = []
	y_train = []
	for c in classes:
		X_val.append(waveforms[c][-2])
		y_val.append(c)
		X_train.append(waveforms[c][0])
		X_train.append(waveforms[c][1])
		X_train.append(waveforms[c][2])
		for i in range(0, 3):
			y_train.append(c)

	fold_1.append(X_train)
	fold_1.append(y_train)
	fold_1.append(X_val)
	fold_1.append(y_val)


	#Fold 2
	X_val = []
	y_val = []
	X_train = []
	y_train = []
	for c in classes:
		X_val.append(waveforms[c][0])
		y_val.append(c)
		X_train.append(waveforms[c][1])
		X_train.append(waveforms[c][3])
		X_train.append(waveforms[c][2])
		for i in range(0, 3):
			y_train.append(c)

	fold_2.append(X_train)
	fold_2.append(y_train)
	fold_2.append(X_val)
	fold_2.append(y_val)

	#Fold 3
	X_val = []
	y_val = []
	X_train = []
	y_train = []
	for c in classes:
		X_val.append(waveforms[c][1])
		y_val.append(c)
		X_train.append(waveforms[c][3])
		X_train.append(waveforms[c][0])
		X_train.append(waveforms[c][2])
		for i in range(0, 3):
			y_train.append(c)

	fold_3.append(X_train)
	fold_3.append(y_train)
	fold_3.append(X_val)
	fold_3.append(y_val)

	#Fold 4
	X_val = []
	y_val = []
	X_train = []
	y_train = []
	for c in classes:
		X_val.append(waveforms[c][2])
		y_val.append(c)
		X_train.append(waveforms[c][0])
		X_train.append(waveforms[c][1])
		X_train.append(waveforms[c][3])
		for i in range(0, 3):
			y_train.append(c)

	fold_4.append(X_train)
	fold_4.append(y_train)
	fold_4.append(X_val)
	fold_4.append(y_val)

	#Fold 5
	X_val = []
	y_val = []
	X_train = []
	y_train = []
	for c in classes:
		X_val.append(waveforms[c][3])
		y_val.append(c)
		X_train.append(waveforms[c][0])
		X_train.append(waveforms[c][2])
		X_train.append(waveforms[c][1])
		for i in range(0, 3):
			y_train.append(c)

	fold_5.append(X_train)
	fold_5.append(y_train)
	fold_5.append(X_val)
	fold_5.append(y_val)

	return [fold_1, fold_2, fold_3, fold_4, fold_5], X_test, y_test

# for I in range(len(audio)/tamanho_entrada):
#	inputs.append(audio[I*tamanho_entrada:I*(tamanho_entrada+1)])

def plot_loss(train_loss, val_loss, epochs):
	plt.figure()
	plt.title('Performance da Validação Cruzada')
	mininums = [min(train_loss), min(val_loss)]
	maxinums = [max(train_loss), max(val_loss)]
	plt.ylim(min(mininums), max(maxinums))
	plt.xlim(0, epochs-1)
	plt.xlabel("Épocas")
	plt.ylabel("Erro")
	plt.yscale('log')
	plt.grid()

	plt.plot(
	    train_loss,
	    '-',
	    color="b",
	    label="Treinamento"
	)
	plt.plot(
	    val_loss,
	    '-',
	    color="r",
	    label="Validação"
	)

	plt.legend(loc="lower right")
	plt.show()


def plot(train_scores, test_scores, epochs, training_label, validation_label, ylabel):
	

	plt.figure()
	plt.title('Performance da Validação Cruzada')
	plt.ylim(0.2, 1.01)
	plt.xlim(0, epochs-1)
	plt.xlabel("Épocas")
	plt.ylabel(ylabel)
	plt.grid()

	# Calculate mean and distribution of training history
	train_scores_mean = np.mean(train_scores, axis=0)
	train_scores_std = np.std(train_scores, axis=0)
	test_scores_mean = np.mean(test_scores, axis=0)
	test_scores_std = np.std(test_scores, axis=0)

	# Plot the average scores
	plt.plot(
	    train_scores_mean,
	    '-',
	    color="b",
	    label=training_label
	)
	plt.plot(
	    test_scores_mean,
	    '-',
	    color="r",
	    label=validation_label
	)

	# Plot a shaded area to represent the score distribution
	epochs = list(range(epochs))
	plt.fill_between(
	    epochs,
	    train_scores_mean - train_scores_std,
	    train_scores_mean + train_scores_std,
	    alpha=0.1,
	    color="b"
	)
	plt.fill_between(
	    epochs,
	    test_scores_mean - test_scores_std,
	    test_scores_mean + test_scores_std,
	    alpha=0.1,
	    color="r"
	)

	plt.legend(loc="lower right")
	plt.show()


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


def save_to_file(accuracy_train_scores, accuracy_test_scores, precision_train_scores, precision_test_scores, train_error, test_error):
	with open("logs/log.txt", "w") as f:

		accuracy_train = [accuracy_train_scores[0][-1], accuracy_train_scores[1][-1], accuracy_train_scores[2][-1], accuracy_train_scores[3][-1], accuracy_train_scores[4][-1]]
		accuracy_test = [accuracy_test_scores[0][-1], accuracy_test_scores[1][-1], accuracy_test_scores[2][-1], accuracy_test_scores[3][-1], accuracy_test_scores[4][-1]]

		precision_train = [precision_train_scores[0][-1], precision_train_scores[1][-1], precision_train_scores[2][-1], precision_train_scores[3][-1], precision_train_scores[4][-1]]
		precision_test = [precision_test_scores[0][-1], precision_test_scores[1][-1], precision_test_scores[2][-1], precision_test_scores[3][-1], precision_test_scores[4][-1]]

		t_error =  [train_error[0][-1], train_error[1][-1], train_error[2][-1], train_error[3][-1], train_error[4][-1]]
		v_error =  [test_error[0][-1], test_error[1][-1], test_error[2][-1], test_error[3][-1], test_error[4][-1]]

		f.write("Accuracy: \n")
		f.write("   Mean during training: " + str(np.mean(accuracy_train)) + "\n")
		f.write("   Mean during validation: " + str(np.mean(accuracy_test)) + "\n")
		f.write("   Standart desviation during training: " + str(np.std(accuracy_train)) + "\n")
		f.write("   Standart desviation during validation: " + str(np.std(accuracy_test)) + "\n")

		f.write("Precision: \n")
		f.write("   Mean during training: " + str(np.mean(precision_train)) + "\n")
		f.write("   Mean during validation: " + str(np.mean(precision_test)) + "\n")
		f.write("   Standart desviation during training: " + str(np.std(precision_train)) + "\n")
		f.write("   Standart desviation during validation: " + str(np.std(precision_test)) + "\n")

		f.write("Error: \n")
		f.write("   Mean during training: " + str(np.mean(t_error)) + "\n")
		f.write("   Mean during tests: " + str(np.mean(v_error)) + "\n")
		f.write("   Standart desviation during training: " + str(np.std(t_error)) + "\n")
		f.write("   Standart desviation during tests: " + str(np.std(v_error)) + "\n")

	f.close()

def main():
	accuracy_train_scores = []
	accuracy_test_scores = []
	precision_train_scores = []
	precision_test_scores = []
	train_error = []
	test_error = []
	best_loss = 0
	best_model = None

	epochs=100

	f_X_train = 0
	f_y_train = 1
	f_X_val = 2
	f_y_val = 3

	all_files = get_files_path()[4:]

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

	folds, X_test, Y_test = build_folds_test(waveforms, labels)

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

		history = model.fit(X, Y, epochs=epochs, batch_size=32, validation_data=(X_V, Y_V))
		
		accuracy_train_scores.append(history.history['accuracy'])
		accuracy_test_scores.append(history.history['val_accuracy'])

		precision_train_scores.append(history.history['precision_' + str(count)])
		precision_test_scores.append(history.history['val_precision_' + str(count)])

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
		
	#plot(accuracy_train_scores, accuracy_test_scores, epochs, "Treinamento", "Validação", "Acurácia")
	#plot(precision_train_scores, precision_test_scores, epochs, "Treinamento", "Validação", "Precisão")
	#plot(train_error, test_error, epochs, "Treinamento", "Validação", "Erro")
	#plot_loss(losses, val_losses, epochs)

	#save_to_file(accuracy_train_scores, accuracy_test_scores, precision_train_scores, precision_test_scores, train_error, test_error)

	loaded_model = get_model()
	best_model = loaded_model.load_weights("model.h5")
	score = loaded_model.evaluate(X, Y, verbose=0)
	print(loaded_model.metrics_names)
	print(score)

	return
main()