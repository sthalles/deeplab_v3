import tensorflow as tf
import numpy as np
import deeplab_v3, deeplab_utils
import network
import argparse

slim = tf.contrib.slim

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

class_labels = [v for v in range((args.number_of_classes+1))]
class_labels[-1] = 255

inputs = np.zeros((1,513,513,1), dtype=np.float32)
labels = np.zeros((1,513,513), dtype=np.int32)

multi_grid = [1,2,4]

# inputs has shape [batch, 513, 513, 3]
net = network.densenet(inputs, args, is_training=False, reuse=False)
cross_entropy, pred, probabilities = network.model_loss(net, labels, class_labels)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    res = sess.run(net)
    print(res.shape)