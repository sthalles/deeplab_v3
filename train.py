
import argparse
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
import os
import json
import network
from preprocessing.training import random_flip_image_and_annotation
from preprocessing.read_data import tf_record_parser
import matplotlib.pyplot as plt
import eval

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


parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Training params')
envarg.add_argument("--batch_norm_epsilon", type=float, default=1e-5, help="batch norm epsilon argument for batch normalization")
envarg.add_argument('--batch_norm_decay', type=float, default=0.9997, help='batch norm decay argument for batch normalization.')
envarg.add_argument("--number_of_classes", type=int, default=21, help="Number of classes to be predicted.")
envarg.add_argument("--l2_regularizer", type=float, default=0.0001, help="l2 regularizer parameter.")
envarg.add_argument('--starting_learning_rate', type=float, default=0.0001, help="initial learning rate.")
envarg.add_argument("--multi_grid", type=list, default=[1,2,4], help="Spatial Pyramid Pooling rates")
envarg.add_argument("--output_stride", type=int, default=16, help="Spatial Pyramid Pooling rates")
envarg.add_argument("--gpu_id", type=int, default=0, help="Id of the GPU to be used")

envarg.add_argument("--current_best_val_loss", type=int, default=99999, help="Best validation loss value.")
envarg.add_argument("--accumulated_validation_miou", type=int, default=0, help="Accumulated validation intersection over union.")

trainarg = parser.add_argument_group('Training')
trainarg.add_argument("--batch_size", type=int, default=16, help="Batch size for network train.")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)

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

logits = tf.cond(is_training, true_fn=lambda: network.densenet(batch_images, args, is_training=True, reuse=False),
                              false_fn=lambda: network.densenet(batch_images, args, is_training=False, reuse=True))

tf.summary.image("logits", logits[:, :, :, :3], 1)

# get the error and predictions from the network
cross_entropy, pred = network.model_loss(logits, batch_labels, class_labels)

tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.image("prediction", tf.expand_dims(tf.cast(pred, tf.float32),3), 1)
tf.summary.image("input", batch_images, 1)
tf.summary.image("label", tf.expand_dims(tf.cast(batch_labels, tf.float32),3), 1)

with tf.variable_scope("optimizer_vars"):
    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.starting_learning_rate)
    train_step = slim.learning.create_train_op(cross_entropy, optimizer, global_step=global_step)

# Put all summary ops into one op. Produces string when you run it.
process_str_id = str(os.getpid())
merged_summary_op = tf.summary.merge_all()
LOG_FOLDER = os.path.join(LOG_FOLDER, process_str_id)
# Create the tboard_log folder if doesn't exist yet
if not os.path.exists(LOG_FOLDER):
    print("Tensoboard folder:", LOG_FOLDER)
    os.makedirs(LOG_FOLDER)

#print(end_points)
variables_to_restore = slim.get_variables_to_restore(exclude=["resnet_v2_50/logits", "optimizer_vars",
                                                              "DeepLab_v3/ASPP_layer", "DeepLab_v3/logits"])

pred = tf.reshape(pred, [-1,])
gt = tf.reshape(batch_labels, [-1,])
indices = tf.squeeze(tf.where(tf.less_equal(gt, args.number_of_classes - 1)), 1) ## ignore all labels >= num_classes
gt = tf.cast(tf.gather(gt, indices), tf.int32)
pred = tf.gather(pred, indices)
miou, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=args.number_of_classes)
tf.summary.scalar('miou', miou)


# Add ops to restore all the variables.
restorer = tf.train.Saver(variables_to_restore)

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

    restorer.restore(sess, "./resnet/checkpoints/resnet_v2_50.ckpt")
    print("Model checkpoits restored!")

    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
    # and used to feed the `handle` placeholder.
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    sess.run(training_iterator.initializer)

    validation_running_loss = []

    train_steps_before_eval = 100
    validation_steps = 20
    while True:
        accumulated_train_loss = 0
        for i in range(train_steps_before_eval): # run this number of batches before validation
            _, global_step_np, train_loss, summary_string = sess.run([train_step,
                                                                                global_step, cross_entropy,
                                                                                merged_summary_op],
                                                                                feed_dict={is_training:True,
                                                                                           handle: training_handle})
            accumulated_train_loss += train_loss
            train_writer.add_summary(summary_string, global_step_np)

        accumulated_train_loss/=train_steps_before_eval

        # at the end of each train interval, run validation
        sess.run(validation_iterator.initializer)

        validation_average_loss = 0
        mean_IoU_list = []
        freq_IoU_list = []
        mean_acc_list = []
        pixel_acc_list = []

        for i in range(validation_steps):
            val_loss, summary_string, _= sess.run([cross_entropy, merged_summary_op, update_op],
                                                                feed_dict={handle: validation_handle,
                                                                           is_training:False})


            validation_average_loss+=val_loss


        validation_average_loss/=validation_steps

        # keep running average of the miou and validation loss
        validation_running_loss.append(validation_average_loss)

        validation_global_loss = np.mean(validation_running_loss)

        if validation_global_loss < current_best_val_loss:
            # Save the variables to disk.
            save_path = saver.save(sess, LOG_FOLDER + "/train" + "/model.ckpt")
            print("Model checkpoints written! Best average val loss:", validation_global_loss)
            current_best_val_loss = validation_global_loss

            # update metadata and save it
            args.current_best_val_loss = str(current_best_val_loss)

            with open(LOG_FOLDER + "/train/" + 'data.json', 'w') as fp:
                json.dump(args.__dict__, fp, sort_keys=True, indent=4)

        print("Global step:", global_step_np, "Average train loss:",
              accumulated_train_loss, "\tGlobal Validation Avg Loss:", validation_global_loss,
              "MIoU:", sess.run(miou))

        test_writer.add_summary(summary_string, global_step_np)

    train_writer.close()
