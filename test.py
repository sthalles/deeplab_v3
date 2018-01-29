import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np
from matplotlib import pyplot as plt
import network
slim = tf.contrib.slim
import os
import itertools
import json
from preprocessing.read_data import tf_record_parser, scale_image_with_crop_padding
from preprocessing import training
from metrics import *

plt.interactive(False)

# best: 16645
model_name = "16645"

os.environ["CUDA_VISIBLE_DEVICES"]="0"
log_folder = './tboard_logs'

if not os.path.exists(os.path.join(log_folder, model_name, "test")):
    os.makedirs(os.path.join(log_folder, model_name, "test"))

with open(log_folder + '/' + model_name + '/train/data.json', 'r') as fp:
    args = json.load(fp)

class Dotdict(dict):
     """dot.notation access to dictionary attributes"""
     __getattr__ = dict.get
     __setattr__ = dict.__setitem__
     __delattr__ = dict.__delitem__

args = Dotdict(args)

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

class_labels = [v for v in range((args.number_of_classes+1))]
class_labels[-1] = 255

LOG_FOLDER = './tboard_logs'
TEST_DATASET_DIR="./dataset/"
TEST_FILE = 'test.tfrecords'

test_filenames = [os.path.join(TEST_DATASET_DIR,TEST_FILE)]
test_dataset = tf.data.TFRecordDataset(test_filenames)
test_dataset = test_dataset.map(tf_record_parser)  # Parse the record into tensors.
test_dataset = test_dataset.map(scale_image_with_crop_padding)
test_dataset = test_dataset.shuffle(buffer_size=100)
test_dataset = test_dataset.batch(args.batch_size)

iterator = test_dataset.make_one_shot_iterator()
batch_images, batch_labels, batch_shapes = iterator.get_next()

logits =  network.densenet(batch_images, args, is_training=False, reuse=False)

valid_labels_batch_tensor, valid_logits_batch_tensor = training.get_valid_logits_and_labels(
    annotation_batch_tensor=batch_labels,
    logits_batch_tensor=logits,
    class_labels=class_labels)

cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tensor,
                                                          labels=valid_labels_batch_tensor)

cross_entropy_mean = tf.reduce_mean(cross_entropies)
tf.summary.scalar('cross_entropy', cross_entropy_mean)

predictions = tf.argmax(logits, axis=3)
probabilities = tf.nn.softmax(logits)

merged_summary_op = tf.summary.merge_all()
saver = tf.train.Saver()

test_folder = os.path.join(log_folder, model_name, "test")
train_folder = os.path.join(log_folder, model_name, "train")

with tf.Session() as sess:

    # Create a saver.
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Restore variables from disk.
    saver.restore(sess, os.path.join(train_folder, "model.ckpt"))
    print("Model", model_name, "restored.")

    mean_IoU = []
    mean_pixel_acc = []
    mean_freq_weighted_IU = []
    mean_acc = []

    while True:
        try:
            images_np, pred_np, annotations_np, shapes_np, summary_string= sess.run([batch_images, predictions, batch_labels, batch_shapes, merged_summary_op])
            heights, widths = shapes_np

            # loop through the images in the batch and extract the valid areas from
            for i in range(pred_np.shape[0]):

                label_image = annotations_np[i]
                pred_image = pred_np[i]
                input_image = images_np[i]

                indices = np.where(label_image != 255)
                label_image = label_image[indices]
                pred_image = pred_image[indices]
                input_image = input_image[indices]

                if label_image.shape[0] == 263169:
                    label_image = np.reshape(label_image, (513,513))
                    pred_image = np.reshape(pred_image, (513,513))
                    input_image = np.reshape(input_image, (513,513,3))
                else:
                    label_image = np.reshape(label_image, (heights[i], widths[i]))
                    pred_image = np.reshape(pred_image, (heights[i], widths[i]))
                    input_image = np.reshape(input_image, (heights[i], widths[i], 3))

                pix_acc = pixel_accuracy(pred_image, label_image)
                m_acc = mean_accuracy(pred_image, label_image)
                IoU = mean_IU(pred_image, label_image)
                freq_weighted_IU = frequency_weighted_IU(pred_image, label_image)

                mean_pixel_acc.append(pix_acc)
                mean_acc.append(m_acc)
                mean_IoU.append(IoU)
                mean_freq_weighted_IU.append(freq_weighted_IU)

                #f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 8))

                #ax1.imshow(input_image.astype(np.uint8))
                #ax2.imshow(label_image)
                #ax3.imshow(pred_image)
                #plt.show()

        except tf.errors.OutOfRangeError:
            break

    print("Mean pixel accuracy:", np.mean(mean_pixel_acc))
    print("Mean accuraccy:", np.mean(mean_acc))
    print("Mean IoU:", np.mean(mean_IoU))
    print("Mean frequency weighted IU:", np.mean(mean_freq_weighted_IU))
