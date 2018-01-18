import tensorflow as tf
slim = tf.contrib.slim
import deeplab_v3, deeplab_utils


def densenet(inputs, args, is_training, reuse):

    # inputs has shape [batch, 513, 513, 3]
    inputs = tf.matmul(inputs, 255)

    with slim.arg_scope(deeplab_v3.deeplab_arg_scope()):
        net, end_points = deeplab_v3.deeplab_v3_50(inputs,
                                                  num_classes=args.number_of_classes,
                                                  multi_grid=args.multi_grid,
                                                  is_training=is_training,
                                                  output_stride=args.output_stride,
                                                  reuse=reuse)
        size = tf.shape(inputs)[1:3]
        # resize the output logits to match the labels dimensions
        net = tf.image.resize_nearest_neighbor(net, size)
        return net


def model_loss(logits, valid_logits, valid_labels):

    cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits,
                                                              labels=valid_labels)

    cross_entropy_mean = tf.reduce_mean(cross_entropies)

    pred = tf.argmax(logits, axis=3)

    probabilities = tf.nn.softmax(logits)
    return cross_entropy_mean, pred, probabilities

