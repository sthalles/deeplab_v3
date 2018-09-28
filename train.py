
import argparse
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
import os
import json
import network
from preprocessing.read_data import download_resnet_checkpoint_if_necessary, tf_record_parser, \
    rescale_image_and_annotation_by_factor, scale_image_with_crop_padding, \
    random_flip_image_and_annotation, distort_randomly_image_color
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

parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Training params')
envarg.add_argument("--batch_norm_epsilon", type=float, default=1e-5, help="batch norm epsilon argument for batch normalization")
envarg.add_argument('--batch_norm_decay', type=float, default=0.9997, help='batch norm decay argument for batch normalization.')
envarg.add_argument("--number_of_classes", type=int, default=21, help="Number of classes to be predicted.")
envarg.add_argument("--l2_regularizer", type=float, default=0.0001, help="l2 regularizer parameter.")
envarg.add_argument('--starting_learning_rate', type=float, default=0.00001, help="initial learning rate.")
envarg.add_argument("--multi_grid", type=list, default=[1,2,4], help="Spatial Pyramid Pooling rates")
envarg.add_argument("--output_stride", type=int, default=16, help="Spatial Pyramid Pooling rates")
envarg.add_argument("--gpu_id", type=int, default=0, help="Id of the GPU to be used")
envarg.add_argument("--crop_size", type=int, default=513, help="Image Cropsize.")
envarg.add_argument("--resnet_model", default="resnet_v2_50", choices=["resnet_v2_50", "resnet_v2_101", "resnet_v2_152", "resnet_v2_200"], help="Resnet model to use as feature extractor. Choose one of: resnet_v2_50 or resnet_v2_101")

envarg.add_argument("--current_best_val_loss", type=int, default=99999, help="Best validation loss value.")
envarg.add_argument("--accumulated_validation_miou", type=int, default=0, help="Accumulated validation intersection over union.")

trainarg = parser.add_argument_group('Training')
trainarg.add_argument("--batch_size", type=int, default=8, help="Batch size for network train.")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)

LOG_FOLDER = './tboard_logs'
TRAIN_DATASET_DIR="./dataset/tfrecords"
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'

crop_size = args.crop_size
training_filenames = [os.path.join(TRAIN_DATASET_DIR,TRAIN_FILE)]
training_dataset = tf.data.TFRecordDataset(training_filenames)
training_dataset = training_dataset.map(tf_record_parser)
training_dataset = training_dataset.map(rescale_image_and_annotation_by_factor)
training_dataset = training_dataset.map(distort_randomly_image_color)
training_dataset = training_dataset.map(lambda image, annotation, image_shape: scale_image_with_crop_padding(image, annotation, image_shape, crop_size))
training_dataset = training_dataset.map(random_flip_image_and_annotation)  # Parse the record into tensors.
training_dataset = training_dataset.repeat()  # number of epochs
training_dataset = training_dataset.shuffle(buffer_size=500)
training_dataset = training_dataset.batch(args.batch_size)

validation_filenames = [os.path.join(TRAIN_DATASET_DIR,VALIDATION_FILE)]
validation_dataset = tf.data.TFRecordDataset(validation_filenames)
validation_dataset = validation_dataset.map(tf_record_parser)  # Parse the record into tensors.
validation_dataset = validation_dataset.map(lambda image, annotation, image_shape: scale_image_with_crop_padding(image, annotation, image_shape, crop_size))
validation_dataset = validation_dataset.shuffle(buffer_size=100)
validation_dataset = validation_dataset.batch(args.batch_size)

resnet_checkpoints_path = "./resnet/checkpoints/"
download_resnet_checkpoint_if_necessary(resnet_checkpoints_path, args.resnet_model)

# A feedable iterator is defined by a handle placeholder and its structure. We
# could use the `output_types` and `output_shapes` properties of either
# `training_dataset` or `validation_dataset` here, because they have
# identical structure.
handle = tf.placeholder(tf.string, shape=[])

iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
batch_images_tf, batch_labels_tf, _ = iterator.get_next()

# You can use feedable iterators with a variety of different kinds of iterator
# (such as one-shot and initializable iterators).
training_iterator = training_dataset.make_initializable_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

class_labels = [v for v in range((args.number_of_classes+1))]
class_labels[-1] = 255

is_training_tf = tf.placeholder(tf.bool, shape=[])

logits_tf = tf.cond(is_training_tf, true_fn= lambda: network.deeplab_v3(batch_images_tf, args, is_training=True, reuse=False),
                    false_fn=lambda: network.deeplab_v3(batch_images_tf, args, is_training=False, reuse=True))

# get valid logits and labels (factor the 255 padded mask out for cross entropy)
valid_labels_batch_tf, valid_logits_batch_tf = training.get_valid_logits_and_labels(
    annotation_batch_tensor=batch_labels_tf,
    logits_batch_tensor=logits_tf,
    class_labels=class_labels)

cross_entropies = tf.nn.softmax_cross_entropy_with_logits_v2(logits=valid_logits_batch_tf,
                                                             labels=valid_labels_batch_tf)
cross_entropy_tf = tf.reduce_mean(cross_entropies)
predictions_tf = tf.argmax(logits_tf, axis=3)

tf.summary.scalar('cross_entropy', cross_entropy_tf)
#tf.summary.image("prediction", tf.expand_dims(tf.cast(pred, tf.float32),3), 1)
#tf.summary.image("label", tf.expand_dims(tf.cast(batch_labels, tf.float32),3), 1)

with tf.variable_scope("optimizer_vars"):
    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.starting_learning_rate)
    train_step = slim.learning.create_train_op(cross_entropy_tf, optimizer, global_step=global_step)

# Put all summary ops into one op. Produces string when you run it.
process_str_id = str(os.getpid())
merged_summary_op = tf.summary.merge_all()
LOG_FOLDER = os.path.join(LOG_FOLDER, process_str_id)
# Create the tboard_log folder if doesn't exist yet
if not os.path.exists(LOG_FOLDER):
    print("Tensoboard folder:", LOG_FOLDER)
    os.makedirs(LOG_FOLDER)

#print(end_points)
variables_to_restore = slim.get_variables_to_restore(exclude=[args.resnet_model + "/logits", "optimizer_vars",
                                                              "DeepLab_v3/ASPP_layer", "DeepLab_v3/logits"])

miou, update_op = tf.contrib.metrics.streaming_mean_iou(tf.argmax(valid_logits_batch_tf, axis=1),
                                                        tf.argmax(valid_labels_batch_tf, axis=1),
                                                        num_classes=args.number_of_classes)
tf.summary.scalar('miou', miou)

# Add ops to restore all the variables.
restorer = tf.train.Saver(variables_to_restore)

saver = tf.train.Saver()

current_best_val_loss = np.inf

with tf.Session() as sess:
    # Create the summary writer -- to write all the tboard_log
    # into a specified file. This file can be later read by tensorboard.
    train_writer = tf.summary.FileWriter(LOG_FOLDER + "/train", sess.graph)
    test_writer = tf.summary.FileWriter(LOG_FOLDER + '/val')

    # Create a saver.
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # load resnet checkpoints
    try:
        restorer.restore(sess, "./resnet/checkpoints/" + args.resnet_model + ".ckpt")
        print("Model checkpoits for " + args.resnet_model + " restored!")
    except FileNotFoundError:
        print("ResNet checkpoints not found. Please download " + args.resnet_model + " model checkpoints from: https://github.com/tensorflow/models/tree/master/research/slim")

    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
    # and used to feed the `handle` placeholder.
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    sess.run(training_iterator.initializer)

    validation_running_loss = []

    train_steps_before_eval = 100
    validation_steps = 20
    while True:
        training_average_loss = 0
        for i in range(train_steps_before_eval): # run this number of batches before validation
            _, global_step_np, train_loss, summary_string = sess.run([train_step,
                                                                      global_step, cross_entropy_tf,
                                                                      merged_summary_op],
                                                                                feed_dict={is_training_tf:True,
                                                                                           handle: training_handle})
            training_average_loss += train_loss

            if i % 10 == 0:
                train_writer.add_summary(summary_string, global_step_np)

        training_average_loss/=train_steps_before_eval

        # at the end of each train interval, run validation
        sess.run(validation_iterator.initializer)

        validation_average_loss = 0
        validation_average_miou = 0
        for i in range(validation_steps):
            val_loss, summary_string, _= sess.run([cross_entropy_tf, merged_summary_op, update_op],
                                                  feed_dict={handle: validation_handle,
                                                             is_training_tf:False})


            validation_average_loss+=val_loss
            validation_average_miou+=sess.run(miou)

        validation_average_loss/=validation_steps
        validation_average_miou/=validation_steps

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
              training_average_loss, "\tGlobal Validation Avg Loss:", validation_global_loss,
              "MIoU:", validation_average_miou)

        test_writer.add_summary(summary_string, global_step_np)

    train_writer.close()
