import tensorflow as tf
##from tensorflow.keras.datasets import fashion_mnist

def _parse_function(image, label, size):
    #image = tf.image.resize(image, [size, size])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
    return image, label


def input_fn(is_training, images, labels, params):
    #Input function for the Fashion MNIST dataset
    num_samples = len(images)
    assert num_samples == len(labels), "Number of images and labels must be equal."

    parse_fn = lambda x, y: _parse_function(x, y, params.image_size)
    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((images, labels))
                   .shuffle(num_samples)
                   .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
                   .batch(params.batch_size)
                   .prefetch(1))
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((images, labels))
                   .map(parse_fn)
                   .batch(params.batch_size)
                   .prefetch(1))

    iterator = iter(dataset)
    images, labels = next(iterator)

    inputs = {'images': images, 'labels': labels}
    return inputs