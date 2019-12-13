
from matplotlib import pyplot as plt
import numpy as np

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