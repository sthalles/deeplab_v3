import numpy as np
from scipy.misc import imread
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

    image = tf.reshape(image, (height,width,3), name="image_reshape")
    annotation = tf.reshape(annotation, (height,width), name="annotation_reshape")
    return tf.to_float(image), annotation


def read_image_and_annotation(train_images_dir, train_annotations_dir, image_name):
    # read the input and annotation images
    image = imread(train_images_dir + image_name.strip() + ".tiff")
    annotation = imread(train_annotations_dir + image_name.strip() + ".tif")
    return image.astype(np.float32), annotation

def cutout(image, label):
    cutout_shape = [24,24]
    original_image_shape = tf.shape(image)
    def random_cutout(image):
        center_x = np.random.randint(0, image.shape[0])
        center_y = np.random.randint(0, image.shape[0])

        # check boundaries conditions
        from_x = center_x-cutout_shape[0]//2 if center_x-cutout_shape[0]//2 > 0 else 0
        to_x = image.shape[0] if (center_x+cutout_shape[0]//2) > image.shape[0] else center_x+cutout_shape[0]//2

        from_y = center_y-cutout_shape[0]//2 if center_y-cutout_shape[0]//2 > 0 else 0
        to_y = image.shape[1] if (center_y+cutout_shape[0]//2) > image.shape[1] else center_y+cutout_shape[1]//2

        image[from_x:to_x,from_y:to_y] = 0
        return image

    return tf.reshape(tf.py_func(random_cutout, [image], (image.dtype)), original_image_shape), label

def random_crop(image_np, annotation_np, crop_size=128):
    """
    image_np: rgb image shape (H,W,3)
    annotation_np: 1D image shape (H,W,1)
    crop_size: integer
    """
    image_h = image_np.shape[0]
    image_w = image_np.shape[1]

    random_x = np.random.randint(0, image_w-crop_size+1) # Return random integers from low (inclusive) to high (exclusive).
    random_y = np.random.randint(0, image_h-crop_size+1) # Return random integers from low (inclusive) to high (exclusive).

    offset_x = random_x + crop_size
    offset_y = random_y + crop_size

    return image_np[random_x:offset_x, random_y:offset_y,:], annotation_np[random_x:offset_x, random_y:offset_y]


def next_batch(train_images_dir, train_annotations_dir, image_filenames_list, batch_size=5, crop_size=128, random_cropping=True, cutout=True):

    for image_name in image_filenames_list:

        image_np, annotation_np = read_image_and_annotation(train_images_dir, train_annotations_dir, image_name)

        batch_images = None
        batch_labels = None
        image_h = image_np.shape[0]
        image_w = image_np.shape[1]
        cutout_shape = [24,24]
        if random_cropping:
            for batch_i in range(batch_size):
                while True:
                    random_image_patch, random_annotation_patch = random_crop(image_np, annotation_np, crop_size)

                    # count the # of zeros in the image patch, because the dataset has some images with zeros (invalid areas)
                    # we crop patches that have less than 10% of white pixels in it
                    n_of_zeros = np.sum(np.all(random_image_patch == [255,255,255], axis=2))

                    #print("# of zeros:", n_of_zeros, "from image:", image_name)
                    if n_of_zeros < 0.01 * (crop_size * crop_size):
                        break

                if cutout:
                    center_x = np.random.randint(0, random_image_patch.shape[0])
                    center_y = np.random.randint(0, random_image_patch.shape[0])

                    # check boundaries conditions
                    from_x = center_x-cutout_shape[0]//2 if center_x-cutout_shape[0]//2 > 0 else 0
                    to_x = random_image_patch.shape[0] if (center_x+cutout_shape[0]//2) > random_image_patch.shape[0] else center_x+cutout_shape[0]//2

                    from_y = center_y-cutout_shape[0]//2 if center_y-cutout_shape[0]//2 > 0 else 0
                    to_y = random_image_patch.shape[1] if (center_y+cutout_shape[0]//2) > random_image_patch.shape[1] else center_y+cutout_shape[1]//2

                    random_image_patch[from_x:to_x,from_y:to_y] = 0

                random_image_patch = np.expand_dims(random_image_patch, axis=0)
                random_annotation_patch = np.expand_dims(random_annotation_patch, axis=0)

                if batch_images is None:
                    batch_images = random_image_patch
                    batch_labels = random_annotation_patch
                else:
                    #print(batch_images.shape, cropped_image.shape)
                    batch_images = np.concatenate((batch_images, random_image_patch), axis=0)
                    batch_labels = np.concatenate((batch_labels, random_annotation_patch), axis=0)
        else:

            if image_w % crop_size  != 0:
                raise Exception("Error, for serial cropping, the crop size:", crop_size, "must be a factor of the image size:", image_w)

            # perform serial crops on the input image/annotation and build a batch of serial patches
            for i in range(0, image_w, crop_size):
                for j in range(0, image_h, crop_size):

                    offset_x = i + crop_size
                    offset_y = j + crop_size

                    image_patch = image_np[i:offset_x, j:offset_y,:]
                    image_patch = np.expand_dims(image_patch, axis=0)

                    annotation_patch = annotation_np[i:offset_x, j:offset_y]
                    annotation_patch = np.expand_dims(annotation_patch, axis=0)

                    if batch_images is None:
                        batch_images = image_patch
                        batch_labels = annotation_patch
                    else:
                        batch_images = np.concatenate((batch_images, image_patch), axis=0)
                        batch_labels = np.concatenate((batch_labels, annotation_patch), axis=0)

        yield batch_images, batch_labels, image_np, annotation_np

def get_labels_from_annotation(annotation_tensor, class_labels):
    """Returns tensor of size (width, height, num_classes) derived from annotation tensor.
    The function returns tensor that is of a size (width, height, num_classes) which
    is derived from annotation tensor with sizes (width, height) where value at
    each position represents a class. The functions requires a list with class
    values like [0, 1, 2 ,3] -- they are used to derive labels. Derived values will
    be ordered in the same way as the class numbers were provided in the list. Last
    value in the aforementioned list represents a value that indicate that the pixel
    should be masked out. So, the size of num_classes := len(class_labels) - 1.

    Parameters
    ----------
    annotation_tensor : Tensor of size (width, height)
        Tensor with class labels for each element
    class_labels : list of ints
        List that contains the numbers that represent classes. Last
        value in the list should represent the number that was used
        for masking out.

    Returns
    -------
    labels_2d_stacked : Tensor of size (width, height, num_classes).
        Tensor with labels for each pixel.
    """

    # Last value in the classes list should show
    # which number was used in the annotation to mask out
    # the ambigious regions or regions that should not be
    # used for train.
    # TODO: probably replace class_labels list with some custom object
    valid_entries_class_labels = class_labels[:-1]

    # Stack the binary masks for each class
    labels_2d = list(map(lambda x: tf.equal(annotation_tensor, x),
                    valid_entries_class_labels))

    # Perform the merging of all of the binary masks into one matrix
    labels_2d_stacked = tf.stack(labels_2d, axis=2)

    # Convert tf.bool to tf.float
    # Later on in the labels and logits will be used
    # in tf.softmax_cross_entropy_with_logits() function
    # where they have to be of the float type.
    labels_2d_stacked_float = tf.to_float(labels_2d_stacked)

    return labels_2d_stacked_float

def get_labels_from_annotation_batch(annotation_batch_tensor, class_labels):
    """Returns tensor of size (batch_size, width, height, num_classes) derived
    from annotation batch tensor. The function returns tensor that is of a size
    (batch_size, width, height, num_classes) which is derived from annotation tensor
    with sizes (batch_size, width, height) where value at each position represents a class.
    The functions requires a list with class values like [0, 1, 2 ,3] -- they are
    used to derive labels. Derived values will be ordered in the same way as
    the class numbers were provided in the list. Last value in the aforementioned
    list represents a value that indicate that the pixel should be masked out.
    So, the size of num_classes len(class_labels) - 1.

    Parameters
    ----------
    annotation_batch_tensor : Tensor of size (batch_size, width, height)
        Tensor with class labels for each element
    class_labels : list of ints
        List that contains the numbers that represent classes. Last
        value in the list should represent the number that was used
        for masking out.

    Returns
    -------
    batch_labels : Tensor of size (batch_size, width, height, num_classes).
        Tensor with labels for each batch.
    """

    batch_labels = tf.map_fn(fn=lambda x: get_labels_from_annotation(annotation_tensor=x, class_labels=class_labels),
                             elems=annotation_batch_tensor,
                             dtype=tf.float32)

    return batch_labels

def reconstruct_image(annotations_batch, crop_size=250, channels=1):
    output_image = np.zeros((1500,1500,channels), dtype=np.float32)
    g_counter = 0

    for i in range(0, 1500, crop_size):
        for j in range(0, 1500, crop_size):
            # print(i, j)
            patch = annotations_batch[g_counter]
            patch = np.expand_dims(patch, axis=2)
            offset_x = i + crop_size
            offset_y = j + crop_size

            #print(i,offset_x, ", ", j, offset_y)
            output_image[i:offset_x, j:offset_y,:] = patch
            g_counter += 1

    return output_image
