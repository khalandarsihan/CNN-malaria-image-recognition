# Print sample info
import tensorflow as tf
def print_sample_info(dataset, num_samples=5):
    """
    Prints the shape, size, dimensions, and labels of samples from the dataset.

    :param dataset: tf.data.Dataset object
    :param num_samples: number of samples to display
    """
    # Shuffle the dataset
    shuffled_dataset = dataset.shuffle(buffer_size=1000)  # Adjust buffer_size based on your dataset size

    for i, (image, label) in enumerate(shuffled_dataset.take(num_samples)):
        print(f"Sample {i+1}:")
        print(f"Image shape: {image.shape}")
        print(f"Image size: {tf.size(image).numpy()}")
        print(f"Image dimensions: {image.ndim}")
        print(f"Label: {label.numpy()}\n")


# Check if there are any missing labels or corrupt images
def check_missing_data(dataset):
    missing_count = 0
    for image, label in dataset:
        if image is None or label is None:
            missing_count += 1
    return missing_count

# Define image preprocessing functions
import tensorflow as tf
def preprocess_image(image, label):
    image = tf.image.resize(image, [128, 128])  # Resize images
    image = image / 255.0  # Rescale images
    return image, label

import tensorflow as tf
def preprocess_dataset(dataset, target_size=(128, 128), batch_size=64, buffer_size=1000):
    """
    Preprocesses the given dataset by resizing images, normalizing pixel values, and shuffling.

    :param dataset: tf.data.Dataset object
    :param target_size: target size to resize the images
    :param batch_size: batch size for processing the dataset
    :param buffer_size: buffer size for prefetching
    :return: preprocessed tf.data.Dataset object
    """
    def preprocess_image(image, label):
        """
        Resizes and normalizes the image.

        :param image: TensorFlow image tensor
        :param label: corresponding label tensor
        :return: tuple of (preprocessed image tensor, label tensor)
        """
        # Ensure the image has 3 dimensions (height, width, channels)
        if image.shape.rank == 2:
            image = tf.expand_dims(image, axis=-1)  # Add channel dimension
            image = tf.image.grayscale_to_rgb(image)  # Convert grayscale to RGB
        elif image.shape.rank == 3 and image.shape[-1] == 1:  # Check if the image is grayscale
            image = tf.image.grayscale_to_rgb(image)  # Convert grayscale to RGB

        image = tf.image.resize(image, target_size)  # Resize images
        image = image / 255.0  # Normalize images
        return image, label

    # Shuffle, map the preprocessing function, batch the dataset, and prefetch
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset