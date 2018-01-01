
import argparse
from network import model_input, model_loss,  model_loss_with_class_balancing, densenet_segmentation, densenet_fcn_segmentation, deeplab_segmentation #,dense_segmentation_convnet_out_stride_8 #, model_arg_scope
import tensorflow as tf
from read_data import tf_record_parser, cutout
import numpy as np
slim = tf.contrib.slim
import os
import json
#from preprocessing import random_flip_image_and_annotation
# from preprocessing import distort_randomly_image_color
from shutil import copyfile
from deeplab import deeplab_v3

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

os.environ["CUDA_VISIBLE_DEVICES"]="3"
parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Training params')
envarg.add_argument("--batch_norm_epsilon", type=float, default=1e-5, help="batch norm epsilon argument for batch normalization")
envarg.add_argument('--batch_norm_decay', type=float, default=0.9997, help='batch norm decay argument for batch normalization.')
envarg.add_argument("--number_of_classes", type=int, default=21, help="Number of classes to be predicted.")
envarg.add_argument("--l2_regularizer", type=float, default=0.0001, help="l2 regularizer parameter.")
envarg.add_argument('--starting_learning_rate', type=float, default=0.007, help="initial learning rate.")
envarg.add_argument("--muiti_grid", type=list, default=[1,2,3], help="Spatial Pyramid Pooling rates")
envarg.add_argument("--output_stride", type=int, default=16, help="Spatial Pyramid Pooling rates")

envarg.add_argument("--current_best_val_loss", type=int, default=99999, help="Best validation loss value.")
envarg.add_argument("--accumulated_validation_miou", type=int, default=0, help="Accumulated validation intersection over union.")

dataarg = parser.add_argument_group('Read data')
dataarg.add_argument("--crop_size", type=float, default=513, help="Crop size for batch train.")

trainarg = parser.add_argument_group('Training')
trainarg.add_argument("--batch_size", type=int, default=16, help="Batch size for network train.")

args = parser.parse_args()

log_folder = './tboard_logs'

TRAIN_DATASET_DIR="./dataset/"
TRAIN_FILE = 'train_65x65__.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'

training_filenames = [os.path.join(TRAIN_DATASET_DIR,TRAIN_FILE)]
training_dataset = tf.data.TFRecordDataset(training_filenames)
training_dataset = training_dataset.map(tf_record_parser)  # Parse the record into tensors.
training_dataset = training_dataset.map(random_flip_image_and_annotation)
training_dataset = training_dataset.map(distort_randomly_image_color)
training_dataset = training_dataset.map(cutout)
training_dataset = training_dataset.repeat()  # number of epochs
training_dataset = training_dataset.shuffle(buffer_size=1000)
training_dataset = training_dataset.batch(args.batch_size)

validation_filenames = [os.path.join(TRAIN_DATASET_DIR,VALIDATION_FILE)]
validation_dataset = tf.data.TFRecordDataset(validation_filenames)
validation_dataset = validation_dataset.map(tf_record_parser)  # Parse the record into tensors.
validation_dataset = validation_dataset.shuffle(buffer_size=100)
validation_dataset = validation_dataset.batch(1)

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

# get the model placeholders
#is_training_placeholder = model_input()

class_labels = [v for v in range((args.number_of_classes+1))]
class_labels[-1] = 255



#logits = dense_segmentation_convnet(batch_images, args, is_training_placeholder)
logits_train = deeplab_segmentation(batch_images, args, True, reuse=False)
logits_val = deeplab_segmentation(batch_images, args, False, reuse=True)

# get the error and predictions from the network
cross_entropy, pred, probabilities = model_loss(logits_train, batch_labels, class_labels)
cross_entropy_val, pred_val, probabilities_val = model_loss(logits_val, batch_labels, class_labels)

# Example: decay from 0.01 to 0.0001 in 10000 steps using sqrt (i.e. power=1. linearly):
global_step = tf.Variable(0, trainable=False)

with tf.variable_scope("optimizer_vars"):
    optimizer = tf.train.AdamOptimizer(learning_rate=args.starting_learning_rate)

train_step = slim.learning.create_train_op(cross_entropy, optimizer, global_step=global_step)

# Define the accuracy metric: Mean Intersection Over Union
miou, update_op = slim.metrics.streaming_mean_iou(predictions=pred,
                                                   labels=batch_labels,
                                                   num_classes=args.number_of_classes)

tf.summary.scalar('miou', miou)

# Put all summary ops into one op. Produces string when you run it.
process_str_id = str(os.getpid())
merged_summary_op = tf.summary.merge_all()
log_folder = os.path.join(log_folder, process_str_id)
# Create the tboard_log folder if doesn't exist yet
if not os.path.exists(log_folder):
    print("Tensoboard folder:", log_folder)
    os.makedirs(log_folder)

saver = tf.train.Saver()

current_best_val_loss = np.inf

with tf.Session() as sess:
    # Create the summary writer -- to write all the tboard_log
    # into a specified file. This file can be later read
    # by tensorboard.
    train_writer = tf.summary.FileWriter(log_folder + "/train", sess.graph)
    test_writer = tf.summary.FileWriter(log_folder + '/val')

    with open(log_folder + "/train/" + 'README.txt', 'w+') as f:
        f.write("Deeplab v3 - output-stride=8, aspp(1,2,3)")

    # Create a saver.
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    accumulated_train_loss = 0
    accumulated_validation_loss = 0
    accumulated_validation_miou = args.accumulated_validation_miou

    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
    # and used to feed the `handle` placeholder.
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    sess.run(training_iterator.initializer)

    copyfile("./network.py", log_folder + "/train/network.py")
    #copyfile("./densenet/densenet_utils.py", log_folder + "/train/densenet_utils.py")
    #copyfile("./densenet/densenet.py", log_folder + "/train/densenet.py")

    while True:
        for i in range(100): # run this number of batches before validation
            _, global_step_np, train_loss, pred_np, probabilities_np, summary_string = sess.run([train_step,
                                                                                global_step, cross_entropy, pred,
                                                                                probabilities, merged_summary_op],
                                                                                feed_dict={handle: training_handle})

            accumulated_train_loss += train_loss
            train_writer.add_summary(summary_string, global_step_np)

        accumulated_train_loss/=100

        # at the end of each train interval, run validation
        sess.run(validation_iterator.initializer)

        for _ in range(4): # pass over the entire validation set
            try:
                val_loss, pred_np, probabilities_np, summary_string, _ = sess.run([cross_entropy_val, pred_val, probabilities_val, merged_summary_op, update_op],
                                                                    feed_dict={handle: validation_handle})

                miou_np = sess.run(miou)
                accumulated_validation_miou+=miou_np
                accumulated_validation_loss+=val_loss
            except tf.errors.OutOfRangeError:
                break

        accumulated_validation_miou/=4
        accumulated_validation_loss/=4

        if accumulated_validation_loss < current_best_val_loss:
            # Save the variables to disk.
            save_path = saver.save(sess, log_folder + "/train" + "/model.ckpt")
            print("Model checkpoints written! Best average val loss:", accumulated_validation_loss)
            current_best_val_loss = accumulated_validation_loss

            # update metadata and save it
            args.current_best_val_loss = current_best_val_loss
            args.accumulated_validation_miou = accumulated_validation_miou
            with open(log_folder + "/train/" + 'data.json', 'w') as fp:
                json.dump(args.__dict__, fp, sort_keys=True, indent=4)

        print("Global step:", global_step_np, "Average train loss:",
              accumulated_train_loss, "\tValidation average Loss:",
              accumulated_validation_loss, "\taverage mIOU:", accumulated_validation_miou)

        test_writer.add_summary(summary_string, global_step_np)

    train_writer.close()
