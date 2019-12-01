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
		X_train.append(waveforms[c][3])
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

def get_model():
	# Build MLP
	visible1 = Input(shape=(1024,))
	hidden1 = Dense(1024, activation='relu')(visible1)
	hidden2 = Dense(512, activation='relu')(hidden1)
	output = Dense(4, activation='softmax')(hidden2)
	model = Model(inputs=visible1, outputs=output)
	model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy', 'mse', keras.metrics.Precision()])

	return model

def plot(train_scores, test_scores, epochs):
	param_range = np.arange(1, epochs, 1)

	# Calculate mean and standard deviation for training set scores
	train_mean = np.mean(train_scores, axis=1)
	train_std = np.std(train_scores, axis=1)

	# Calculate mean and standard deviation for test set scores
	test_mean = np.mean(test_scores, axis=1)
	test_std = np.std(test_scores, axis=1)

	# Plot mean accuracy scores for training and test sets
	plt.plot(param_range, train_mean, label="Training score", color="black")
	plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

	# Plot accurancy bands for training and test sets
	plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
	plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

	# Create plot
	plt.title("Validation Curve With Yamnet")
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy Score")
	plt.tight_layout()
	plt.legend(loc="best")
	plt.show()

def main():
	all_accuracy = []
	all_loss = []

	epochs=10

	f_X_train = 0
	f_y_train = 1
	f_X_val = 2
	f_y_val = 3

	all_files = get_files_path()[4:]

	# Build network
	yamnet = yamnet_model.yamnet_frames_model(params)
	yamnet.load_weights('yamnet.h5')
	yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

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

	folds, X_test, y_test = build_folds_test(waveforms, labels)

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

		print("FOLD %d:" % count)
		print(X.shape, Y.shape)
		count += 1

		history_train = model.fit(X, Y, epochs=epochs, batch_size=32, validation_data=(X_V, Y_V))
		
		plot(history_train.history['accuracy'], history_test[1], epochs)
		
	return