import tensorflow as tf
import numpy as np
from deeplab import deeplab_v3, deeplab_utils_utils
slim = tf.contrib.slim

inputs = np.zeros((1,513,513,1), dtype=np.float32)

multi_grid = [1,2,4]

# inputs has shape [batch, 513, 513, 3]
with slim.arg_scope(deeplab_v3.deeplab_arg_scope()):
    net, end_points = deeplab_v3.deeplab_v3_50(inputs,
                                              21,
                                              multi_grid=multi_grid,
                                              is_training=False,
                                              output_stride=16,)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    res = sess.run(net)
    print(res.shape)