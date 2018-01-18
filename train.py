
import argparse
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
import os
import json
import network
from preprocessing.training import random_flip_image_and_annotation
from preprocessing.read_data import tf_record_parser, rescale_image_and_annotation_and_crop
import matplotlib.pyplot as plt
from preprocessing import training

# 0=background
# 1=aeroplane
# 2=bicycle
# 3=bird
# 4=boat
# 5=bottle
# 6=bus
# 7=car
# 8=cat
# 9=chair
# 10=cow
# 11=diningtable
# 12=dog
# 13=horse
# 14=motorbike
# 15=person
# 16=potted plant
# 17=sheep
# 18=sofa
# 19=train
# 20=tv/monitor
# 255=unknown

os.environ["CUDA_VISIBLE_DEVICES"]="0"
parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Training params')
envarg.add_argument("--batch_norm_epsilon", type=float, default=1e-5, help="batch norm epsilon argument for batch normalization")
envarg.add_argument('--batch_norm_decay', type=float, default=0.9997, help='batch norm decay argument for batch normalization.')
envarg.add_argument("--number_of_classes", type=int, default=21, help="Number of classes to be predicted.")
envarg.add_argument("--l2_regularizer", type=float, default=0.0001, help="l2 regularizer parameter.")
envarg.add_argument('--starting_learning_rate', type=float, default=0.007, help="initial learning rate.")
envarg.add_argument("--multi_grid", type=list, default=[1,2,4], help="Spatial Pyramid Pooling rates")
envarg.add_argument("--output_stride", type=int, default=16, help="Spatial Pyramid Pooling rates")

envarg.add_argument("--current_best_val_loss", type=int, default=99999, help="Best validation loss value.")
envarg.add_argument("--accumulated_validation_miou", type=int, default=0, help="Accumulated validation intersection over union.")

trainarg = parser.add_argument_group('Training')
trainarg.add_argument("--batch_size", type=int, default=16, help="Batch size for network train.")

args = parser.parse_args()

LOG_FOLDER = './tboard_logs'
TRAIN_DATASET_DIR="./dataset/"
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'

training_filenames = [os.path.join(TRAIN_DATASET_DIR,TRAIN_FILE)]
training_dataset = tf.data.TFRecordDataset(training_filenames)
training_dataset = training_dataset.map(tf_record_parser)  # Parse the record into tensors.
training_dataset = training_dataset.map(random_flip_image_and_annotation)  # Parse the record into tensors.
training_dataset = training_dataset.repeat()  # number of epochs
training_dataset = training_dataset.shuffle(buffer_size=1000)
training_dataset = training_dataset.batch(args.batch_size)

validation_filenames = [os.path.join(TRAIN_DATASET_DIR,VALIDATION_FILE)]
validation_dataset = tf.data.TFRecordDataset(validation_filenames)
validation_dataset = validation_dataset.map(tf_record_parser)  # Parse the record into tensors.
validation_dataset = validation_dataset.shuffle(buffer_size=100)
validation_dataset = validation_dataset.batch(args.batch_size)

# A feedable iterator is defined by a handle placeholder and its structure. We
# could use the `output_types` and `output_shapes` properties of either
# `training_dataset` or `validation_dataset` here, because they have
# identical structure.
handle = tf.placeholder(tf.string, shape=[])

iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
batch_images, batch_labels = iterator.get_next()

# You can use feedable iterators with a variety of different kinds of iterator
# (such as one-shot and initializable iterators).
training_iterator = training_dataset.make_initializable_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

class_labels = [v for v in range((args.number_of_classes+1))]
class_labels[-1] = 255

is_training = tf.placeholder(tf.bool, shape=[])

logits = network.densenet(batch_images, args, is_training=is_training, reuse=False)

valid_labels_batch_tensor, valid_logits_batch_tensor = training.get_valid_logits_and_labels(
    annotation_batch_tensor=batch_labels,
    logits_batch_tensor=logits,
    class_labels=class_labels)

# get the error and predictions from the network
cross_entropy, pred, probabilities = network.model_loss(logits, valid_logits_batch_tensor, valid_labels_batch_tensor)
tf.summary.image("prediction", tf.expand_dims(tf.cast(pred, tf.float32),3), 1)
tf.summary.image("input", batch_images, 1)
tf.summary.image("label", tf.expand_dims(tf.cast(batch_labels, tf.float32),3), 1)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = args.starting_learning_rate
end_learning_rate = 0.0
decay_steps = 30000
learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                          decay_steps, end_learning_rate,
                                          power=0.9)
tf.summary.scalar('learning_rate', learning_rate)

with tf.variable_scope("optimizer_vars"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_step = slim.learning.create_train_op(cross_entropy, optimizer, global_step=global_step)

# Define the accuracy metric: Mean Intersection Over Union
miou, update_op = slim.metrics.streaming_mean_iou(predictions=tf.argmax(valid_logits_batch_tensor, axis=1),
                                                   labels=tf.argmax(valid_labels_batch_tensor, axis=1),
                                                   num_classes=args.number_of_classes)
tf.summary.scalar('miou', miou)

# Put all summary ops into one op. Produces string when you run it.
process_str_id = str(os.getpid())
merged_summary_op = tf.summary.merge_all()
LOG_FOLDER = os.path.join(LOG_FOLDER, process_str_id)
# Create the tboard_log folder if doesn't exist yet
if not os.path.exists(LOG_FOLDER):
    print("Tensoboard folder:", LOG_FOLDER)
    os.makedirs(LOG_FOLDER)

saver = tf.train.Saver()

current_best_val_loss = np.inf

with tf.Session() as sess:
    # Create the summary writer -- to write all the tboard_log
    # into a specified file. This file can be later read
    # by tensorboard.
    train_writer = tf.summary.FileWriter(LOG_FOLDER + "/train", sess.graph)
    test_writer = tf.summary.FileWriter(LOG_FOLDER + '/val')

    # Create a saver.
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())


    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
    # and used to feed the `handle` placeholder.
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    sess.run(training_iterator.initializer)

    accumulated_validation_loss = []
    accumulated_validation_miou = []

    while True:
        accumulated_train_loss = []
        for i in range(1): # run this number of batches before validation
            lab, _, global_step_np, train_loss, pred_np, probabilities_np, summary_string = sess.run([batch_labels, train_step,
                                                                                global_step, cross_entropy, pred,
                                                                                probabilities, merged_summary_op],
                                                                                feed_dict={is_training:True,
                                                                                           handle: training_handle})
            accumulated_train_loss.append(train_loss)
            train_writer.add_summary(summary_string, global_step_np)


        # at the end of each train interval, run validation
        sess.run(validation_iterator.initializer)

        for i in range(20):
            val_loss, pred_np, probabilities_np, summary_string, _ = sess.run([cross_entropy, pred, probabilities, merged_summary_op, update_op],
                                                                feed_dict={handle: validation_handle,
                                                                           is_training:False})

            miou_np = sess.run(miou)
            accumulated_validation_miou.append(miou_np)
            accumulated_validation_loss.append(val_loss)

        # keep running average of the miou and validation loss
        mean_accumulated_val_loss = np.mean(accumulated_validation_loss)
        mean_accumulated_val_miou = np.mean(accumulated_validation_miou)

        if mean_accumulated_val_loss < current_best_val_loss:
            # Save the variables to disk.
            save_path = saver.save(sess, LOG_FOLDER + "/train" + "/model.ckpt")
            print("Model checkpoints written! Best average val loss:", mean_accumulated_val_loss)
            current_best_val_loss = mean_accumulated_val_loss

            # update metadata and save it
            args.current_best_val_loss = current_best_val_loss
            args.accumulated_validation_miou = mean_accumulated_val_miou

        print("Global step:", global_step_np, "Average train loss:",
              np.mean(accumulated_train_loss), "\tValidation average Loss:",
              mean_accumulated_val_loss, "\tAverage mIOU:", mean_accumulated_val_miou)

        test_writer.add_summary(summary_string, global_step_np)

    train_writer.close()
