
import os
from os import listdir
from os.path import isfile, join
import sys

import numpy as np

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


def build_folds_test(waveforms, labels, classes):
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

def save_to_file(accuracy_train_scores, accuracy_test_scores, precision_train_scores, precision_test_scores, train_error, test_error):
	with open("logs/log.txt", "w") as f:

		accuracy_train = [accuracy_train_scores[0][-1], accuracy_train_scores[1][-1], accuracy_train_scores[2][-1], accuracy_train_scores[3][-1], accuracy_train_scores[4][-1]]
		accuracy_test = [accuracy_test_scores[0][-1], accuracy_test_scores[1][-1], accuracy_test_scores[2][-1], accuracy_test_scores[3][-1], accuracy_test_scores[4][-1]]

		precision_train = [precision_train_scores[0][-1], precision_train_scores[1][-1], precision_train_scores[2][-1], precision_train_scores[3][-1], precision_train_scores[4][-1]]
		precision_test = [precision_test_scores[0][-1], precision_test_scores[1][-1], precision_test_scores[2][-1], precision_test_scores[3][-1], precision_test_scores[4][-1]]

		t_error =  [train_error[0][-1], train_error[1][-1], train_error[2][-1], train_error[3][-1], train_error[4][-1]]
		v_error =  [test_error[0][-1], test_error[1][-1], test_error[2][-1], test_error[3][-1], test_error[4][-1]]

		f.write("Accuracy: \n")
		f.write("	Fold 1: \n")
		f.write("		Training: " + str(accuracy_train_scores[0][-1]) + "\n")
		f.write("		Validation: " + str(accuracy_test_scores[0][-1]) + "\n\n")

		f.write("	Fold 2: " + str(accuracy_train_scores[1][-1]) + "\n")
		f.write("		Training: " + str(accuracy_train_scores[1][-1]) + "\n")
		f.write("		Validation: " + str(accuracy_test_scores[1][-1]) + "\n\n")

		f.write("	Fold 3: " + str(accuracy_train_scores[2][-1]) + "\n")
		f.write("		Training: " + str(accuracy_train_scores[2][-1]) + "\n")
		f.write("		Validation: " + str(accuracy_test_scores[2][-1]) + "\n\n")

		f.write("	Fold 4: " + str(accuracy_train_scores[3][-1]) + "\n")
		f.write("		Training: " + str(accuracy_train_scores[3][-1]) + "\n")
		f.write("		Validation: " + str(accuracy_test_scores[3][-1]) + "\n\n")

		f.write("	Fold 5: " + str(accuracy_train_scores[4][-1]) + "\n")
		f.write("		Training: " + str(accuracy_train_scores[4][-1]) + "\n")
		f.write("		Validation: " + str(accuracy_test_scores[4][-1]) + "\n\n")

		f.write("   Mean during training: " + str(np.mean(accuracy_train)) + "\n")
		f.write("   Mean during validation: " + str(np.mean(accuracy_test)) + "\n")
		f.write("   Standart desviation during training: " + str(np.std(accuracy_train)) + "\n")
		f.write("   Standart desviation during validation: " + str(np.std(accuracy_test)) + "\n")

		f.write("Precision: \n")
		f.write("	Fold 1: " + str(precision_train_scores[0][-1]) + "\n")
		f.write("	Fold 2: " + str(precision_train_scores[1][-1]) + "\n")
		f.write("	Fold 3: " + str(precision_train_scores[2][-1]) + "\n")
		f.write("	Fold 4: " + str(precision_train_scores[3][-1]) + "\n")
		f.write("	Fold 5: " + str(precision_train_scores[4][-1]) + "\n")
		f.write("\n")

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