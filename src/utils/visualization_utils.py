import tensorflow as tf
import matplotlib.pyplot as plt

def visualize_sample_images_with_labels(dataset, title):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    shuffled_dataset = dataset.shuffle(buffer_size=1000).take(9)
    for ax, (image, label) in zip(axes.flatten(), dataset):
        ax.imshow(image.numpy().astype("uint8"))
        ax.set_title("Parasitized" if label==0 else "Uninfected")
        ax.axis("off")
    plt.suptitle(f'{title} - Sample Images with Labels')
    plt.tight_layout()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
def visualize_class_distribution(train_dataset, test_dataset, val_dataset, title):
    """
    Visualizes the class distribution for the given datasets.

    :param train_dataset: tf.data.Dataset object for training data
    :param test_dataset: tf.data.Dataset object for test data
    :param title: title for the plot
    """
    def extract_labels(dataset):
        return [label.numpy() for _, label in dataset]

    train_labels = extract_labels(train_dataset)
    test_labels = extract_labels(test_dataset)
    val_labels = extract_labels(val_dataset)

    def class_dist(labels):
        return pd.Series(labels).value_counts()

    train_dist = class_dist(train_labels)
    test_dist = class_dist(test_labels)
    val_dist = class_dist(val_labels)

    df = pd.DataFrame({
        'Train_dist': train_dist,
        'Test_dist': test_dist,
        'Val_dist': val_dist,
    }).fillna(0)  # Handle cases where a class might be missing in train or test

    df.plot(kind='bar', figsize=(10, 6))
    plt.title(f'{title} - Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def visualize_image_mean_and_std(dataset, title, target_size=(128, 128), batch_size=64):
    """
    Visualizes the mean and standard deviation images for the given dataset.

    :param dataset: tf.data.Dataset object
    :param title: title for the plot
    :param target_size: target size to resize the images
    :param batch_size: batch size for processing the dataset
    """
    # Map the resizing directly within the function and batch the dataset
    dataset = dataset.map(lambda x, y: (tf.image.resize(x, target_size), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    images = []
    for batch in dataset:
        batch_images = batch[0].numpy()
        images.append(batch_images)

    images = np.concatenate(images, axis=0)

    mean_image = np.mean(images, axis=0)
    std_image = np.std(images, axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(mean_image.astype(np.uint8), cmap='gray')
    axes[0].set_title('Mean Image')
    axes[0].axis('off')
    
    axes[1].imshow(std_image.astype(np.uint8), cmap='gray')
    axes[1].set_title('Standard Deviation Image')
    axes[1].axis('off')
    
    fig.suptitle(f'{title} - Image Mean and Standard Deviation')
    plt.tight_layout()
    plt.show()