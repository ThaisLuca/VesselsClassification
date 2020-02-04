
from __future__ import print_function

from random import shuffle

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tf_slim as slim

import vggish_input
import vggish_params
import vggish_slim

import resampy
import soundfile as sf
import utils as util

from keras.utils import to_categorical
from collections import Counter

try:
  from tensorflow.compat.v1 import ConfigProto
  from tensorflow.compat.v1 import InteractiveSession

  config = ConfigProto()
  config.gpu_options.allow_growth = True
  session = InteractiveSession(config=config)
except Exception as e:
  print(e)
  print("Not possible to set gpu allow growth")

flags = tf.app.flags

classes = [0,1,2,3]
labels_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

flags.DEFINE_integer(
    'num_batches', 30,
    'Number of batches of examples to feed into the model. Each batch is of '
    'variable size and contains shuffled examples of each class of audio.')

flags.DEFINE_boolean(
    'train_vggish', True,
    'If True, allow VGGish parameters to change during training, thus '
    'fine-tuning VGGish. If False, VGGish parameters are fixed, thus using '
    'VGGish as a fixed feature extractor.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

FLAGS = flags.FLAGS

_NUM_CLASSES = 4

def main(argv):

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

  # Build network
  #TODO: add network

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
    if sr != vggish_params.SAMPLE_RATE:
      waveform = resampy.resample(waveform, sr, vggish_params.SAMPLE_RATE)

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
      a = vggish_input.wavfile_to_examples(x)
      print(a.shape)
      for i in a:
        X_T.append(i)
        Y_T.append(y)

  X_T = np.array(X_T)
  Y_T = np.array(Y_T)

  print("Test")
  print(Counter(Y_T))

  Y_T = to_categorical(Y_T)

  count = 1
  for fold in folds:

    print("Fold %d:\n" % count)

    X = []
    X_V = []
    Y = []
    Y_V = []
    for x, y in zip(fold[f_X_train], fold[f_y_train]):
      a = vggish_input.wavfile_to_examples(x)
      for i in a:
        X.append(i)
        Y.append(y)

    for x, y in zip(fold[f_X_val], fold[f_y_val]):
      a = vggish_input.wavfile_to_examples(x)
      for i in a:
        X_V.append(i)
        Y_V.append(y)

    X = np.array(X)
    Y = np.array(Y)

    print("Training")
    print(Counter(Y))

    Y = to_categorical(Y)

    X_V = np.array(X_V)
    Y_V = np.array(Y_V)

    print("Validation")
    print(Counter(Y_test))

    Y_V = to_categorical(Y_V)

    with tf.Graph().as_default(), tf.Session() as sess:
      # Define VGGish.
      embeddings = vggish_slim.define_vggish_slim(FLAGS.train_vggish)

      # Define a shallow classification model and associated training ops on top
      # of VGGish.
      with tf.variable_scope('mymodel'):
        # Add a fully connected layer with 100 units.
        num_units = 100
        fc = slim.fully_connected(embeddings, num_units)

        # Add a classifier layer at the end, consisting of parallel logistic
        # classifiers, one per class. This allows for multi-class tasks.
        logits = slim.fully_connected(
            fc, _NUM_CLASSES, activation_fn=None, scope='logits')
        tf.sigmoid(logits, name='prediction')

        # Add training ops.
        with tf.variable_scope('train'):
          global_step = tf.Variable(
              0, name='global_step', trainable=False,
              collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                           tf.GraphKeys.GLOBAL_STEP])

          # Labels are assumed to be fed as a batch multi-hot vectors, with
          # a 1 in the position of each positive class label, and 0 elsewhere.
          labels = tf.placeholder(
              tf.float32, shape=(None, _NUM_CLASSES), name='labels')

          # Cross-entropy label loss.
          xent = tf.nn.sigmoid_cross_entropy_with_logits(
              logits=logits, labels=labels, name='xent')
          loss = tf.reduce_mean(xent, name='loss_op')
          tf.summary.scalar('loss', loss)

          # We use the same optimizer and hyperparameters as used to train VGGish.
          optimizer = tf.train.AdamOptimizer(
              learning_rate=vggish_params.LEARNING_RATE,
              epsilon=vggish_params.ADAM_EPSILON)
          optimizer.minimize(loss, global_step=global_step, name='train_op')

      # Initialize all variables in the model, and then load the pre-trained
      # VGGish checkpoint.
      sess.run(tf.global_variables_initializer())
      vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)

      # Locate all the tensors and ops we need for the training loop.
      features_tensor = sess.graph.get_tensor_by_name(
          vggish_params.INPUT_TENSOR_NAME)
      labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
      global_step_tensor = sess.graph.get_tensor_by_name('mymodel/train/global_step:0')
      loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/loss_op:0')
      train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')

      # The training loop.
      for _ in range(EPOCHS):
        (features, labels) = X, Y #[num_steps, loss, _]
        resut = sess.run(
            [global_step_tensor, loss_tensor, train_op],
            feed_dict={features_tensor: features, labels_tensor: labels})
        #print('Step %d: loss %g' % (num_steps, loss))
        print(result)

        #TODO: medidas de acurácia, precisão e recall

  return

if __name__ == '__main__':
  tf.app.run()