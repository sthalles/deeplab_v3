import tensorflow as tf

def tf_record_parser(record):
    keys_to_features = {
        "image_raw": tf.FixedLenFeature((), tf.string, default_value=""),
        'annotation_raw': tf.FixedLenFeature([], tf.string),
        "height": tf.FixedLenFeature((), tf.int64),
        "width": tf.FixedLenFeature((), tf.int64)
    }

    features = tf.parse_single_example(record, keys_to_features)

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    annotation = tf.decode_raw(features['annotation_raw'], tf.uint8)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    # reshape input and annotation images
    image = tf.reshape(image, (height, width, 3), name="image_reshape")
    annotation = tf.reshape(annotation, (height, width, 1), name="annotation_reshape")
    annotation = tf.to_int32(annotation)

    image, annotation = rescale_image_and_annotation_and_crop(image, annotation)

    image_croped = tf.image.resize_image_with_crop_or_pad(image, 513, 513)

    # Shift all the classes by one -- to be able to differentiate
    # between zeros representing padded values and zeros representing
    # a particular semantic class.
    annotation_shifted_classes = annotation + 1

    cropped_padded_annotation = tf.image.resize_image_with_crop_or_pad(annotation_shifted_classes, 513, 513)

    mask_out_number = 255
    annotation_additional_mask_out = tf.to_int32(tf.equal(cropped_padded_annotation, 0)) * (mask_out_number + 1)
    cropped_padded_annotation = cropped_padded_annotation + annotation_additional_mask_out - 1

    return tf.squeeze(image_croped), tf.squeeze(tf.cast(cropped_padded_annotation, tf.uint8))

def rescale_image_and_annotation_and_crop(image, annotation, nin_scale=0.5, max_scale=2):
    # We apply data augmentation by randomly scaling theinput images(from 0.5 to 2.0)
    # and randomly left - right flipping during training.
    input_shape = tf.shape(image)[0:2]
    input_shape_float = tf.to_float(input_shape)

    scale = tf.random_uniform(shape=[1],
                              minval=0.5,
                              maxval=2)

    scaled_input_shape = tf.to_int32(tf.round(input_shape_float * scale))

    image = tf.image.resize_images(image, scaled_input_shape,
                                   method=tf.image.ResizeMethod.BILINEAR)

    # use nearest neighbour for annotations resizing in order to keep proper values
    annotation = tf.image.resize_images(annotation, scaled_input_shape,
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image, annotation
