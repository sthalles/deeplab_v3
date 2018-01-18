"""Contains definitions for the DeepLab_v3 segmentation model.

Residual networks (DeepLab_v3) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer.

Typical use:

   from tensorflow.contrib.slim.nets import deeplab_v3


DeepLab_v3-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(deeplab_v3.deeplab_arg_scope()):
      net, end_points = deeplab_v3.deeplab_v3_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import deeplab_utils

slim = tf.contrib.slim
deeplab_arg_scope = deeplab_utils.deeplab_arg_scope


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
  """Bottleneck residual unit variant with BN before convolutions.

  This is the full preactivation residual unit variant proposed in [2]. See
  Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
  variant which has an extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
    if depth == depth_in:
      shortcut = deeplab_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                             normalizer_fn=None, activation_fn=None,
                             scope='shortcut')

    residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = deeplab_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                         rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           normalizer_fn=None, activation_fn=None,
                           scope='conv3')

    output = shortcut + residual

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


@slim.add_arg_scope
def atrous_spatial_pyramid_pooling(net, scope, depth=256):
    """
    ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
    (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
    :param net: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
    :param scope: scope name of the aspp layer
    :return: network layer with aspp applyed to it.
    """

    with tf.variable_scope(scope):
        feature_map_size = tf.shape(net)

        # apply global average pooling
        image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keep_dims=True)
        image_level_features = slim.conv2d(image_level_features, depth, [1, 1], scope="image_level_conv_1x1",
                                           activation_fn=None)
        image_level_features = tf.image.resize_bilinear(image_level_features, (feature_map_size[1], feature_map_size[2]))

        at_pool1x1 = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_0", activation_fn=None)

        at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1", rate=6, activation_fn=None)

        at_pool3x3_2 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_2", rate=12, activation_fn=None)

        at_pool3x3_3 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_3", rate=18, activation_fn=None)

        net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3,
                        name="concat")
        net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)
        return net


def deeplab_v3(inputs,
               blocks,
               num_classes=None,
               multi_grid=None,
               is_training=True,
               output_stride=None,
               include_root_block=True,
               reuse=None,
               scope=None):
  """Generator for Deeplab_v3 models.

  This function generates a family of Deeplab_v3 models. See the deeplab_v3_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce DeepLab_v3 of various depths.

  For dense prediction tasks we advise that one uses inputs with
  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
  this case the feature maps at the ResNet output will have spatial shape
  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
  and corners exactly aligned with the input image corners, which greatly
  facilitates alignment of the features to the image. Using as input [225, 225]
  images results in [8, 8] feature maps at the output of the last Deeplab_v3 block.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a deeplab_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks.
      If 0 or None, we return the features before the logit layer.
    multi_grid: list of integers containing the atrous convolution rates for the last
      resnet block.
    is_training: whether batch_norm layers are in train mode.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it. If excluded, `inputs` should be the
      results of an activation-less convolution.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.


  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      The original input tensor is reduced by a
      factor of output_stride compared to the respective height_in and width_in.
      If num_classes is 0 or None, then net is the output of the last ResNet block.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with tf.variable_scope(scope, 'deeplab_v3', [inputs], reuse=reuse) as sc:

    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         deeplab_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          # We do not include batch normalization or activation functions in
          # conv1 because the first ResNet unit will perform these.
          with slim.arg_scope([slim.conv2d],
                              activation_fn=None, normalizer_fn=None):
            net = deeplab_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
          net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

        net = deeplab_utils.stack_blocks_dense(net, blocks, multi_grid, output_stride)

        # apply ASPP on top of the last ResNet block
        net = atrous_spatial_pyramid_pooling(net, "aspp_layer", depth=256)

        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        if num_classes is not None:
          net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='logits')

          end_points[sc.name + '/logits'] = net
          end_points['predictions'] = slim.softmax(net, scope='predictions')
        return net, end_points
deeplab_v3.default_image_size = 224


def deeplab_v3_block(scope, base_depth, num_units, stride, factor=1):
  """Helper function for creating a deeplab_v3 bottleneck block.

  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.

  Returns:
    A deeplab_v3 bottleneck block.
  """
  return deeplab_utils.Block(scope, bottleneck, [{
      'depth': base_depth * factor,
      'depth_bottleneck': base_depth,
      'stride': 1
  }] * (num_units - 1) + [{
      'depth': base_depth * factor,
      'depth_bottleneck': base_depth,
      'stride': stride
  }])
deeplab_v3.default_image_size = 224


def deeplab_v3_50(inputs,
                  num_classes=None,
                  multi_grid=[1,2,4],
                  is_training=True,
                  output_stride=None,
                  reuse=None,
                  scope='deeplab_v3_50'):
  """Deeplab_v3 model based ResNet-50 model of [1]. See deeplab_v3() for arg and return description."""
  blocks = [
      deeplab_v3_block('block1', base_depth=64, num_units=3, stride=2),
      deeplab_v3_block('block2', base_depth=128, num_units=4, stride=2),
      deeplab_v3_block('block3', base_depth=256, num_units=6, stride=2),
      deeplab_v3_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return deeplab_v3(inputs, blocks, num_classes, multi_grid=multi_grid, is_training=is_training,
                    output_stride=output_stride, include_root_block=True,
                    reuse=reuse, scope=scope)
deeplab_v3_50.default_image_size = deeplab_v3.default_image_size


def deeplab_v3_101(inputs,
                   num_classes=None,
                   multi_grid=[1,2,4],
                   is_training=True,
                   output_stride=None,
                   reuse=None,
                   scope='deeplab_v3_101'):
  """Deeplab_v3 model based on ResNet-101 model of [1]. See deeplab_v3() for arg and return description."""
  blocks = [
      deeplab_v3_block('block1', base_depth=64, num_units=3, stride=2),
      deeplab_v3_block('block2', base_depth=128, num_units=4, stride=2),
      deeplab_v3_block('block3', base_depth=256, num_units=23, stride=2),
      deeplab_v3_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return deeplab_v3(inputs, blocks, num_classes, multi_grid=multi_grid, is_training=is_training,
                    output_stride=output_stride, include_root_block=True,
                    reuse=reuse, scope=scope)
deeplab_v3_101.default_image_size = deeplab_v3.default_image_size
