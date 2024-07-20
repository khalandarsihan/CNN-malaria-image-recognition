# src/utils/model_utils.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.utils import dataset_utils as du
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, classification_report
from IPython.display import display, Markdown
import pandas as pd


def extract_labels_predictions(model, dataset):
    """
    Returns the actual images and labels and predicted classes/labels from the dataset using the given model.
    
    Args:
    - model: The trained TensorFlow model to use for predictions.
    - dataset: A preprocessed tf.data.Dataset containing images and labels.
    
    Returns:
    - actual_labels: A numpy array of actual labels from the dataset.
    - predicted_labels: A numpy array of predicted labels from the model.
    """
    # Extract images and labels from the dataset using the helper function from du
    actual_images, actual_labels = du.extract_images_labels(dataset)

    # Generate predictions on the images
    predictions = model.predict(actual_images)
    predicted_labels = (predictions > 0.5).astype(int).flatten()

    return actual_images, actual_labels, predicted_labels

def eval_visualize_performance(test_labels, predicted_classes):
    """
    Evaluates the model's performance and visualizes key metrics.
    
    Args:
    - actual_labels: Actual labels of the test dataset.
    - predicted_classes: Predicted classes/labels from the model.
    - model: The trained TensorFlow model.
    """
    # Debug prints
    print(f"\nActual Labels of first ten samples: {test_labels[:10]}")
    print(f"Prediction for corresponding samples: {predicted_classes[:10]}\n")

    # Compute the confusion matrix
    cm = confusion_matrix(test_labels, predicted_classes)

    # Visualize the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Parasitized', 'Uninfected'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    # Calculate accuracy, precision, recall, and F1 score using scikit-learn
    accuracy = accuracy_score(test_labels, predicted_classes)
    precision = precision_score(test_labels, predicted_classes)
    recall = recall_score(test_labels, predicted_classes)
    f1 = f1_score(test_labels, predicted_classes)

    print(f'\nAccuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Generate a classification report
    class_report = classification_report(test_labels, predicted_classes, target_names=['Parasitized', 'Uninfected'])

    # Display the classification report using IPython.display
    display(Markdown("### Classification Report"))
    display(Markdown(f"```\n{class_report}\n```"))

def show_actuals_predictions(images, actual_labels, predicted_labels, title):
    """
    Visualizes sample images with their actual and predicted labels.
    
    Args:
    - images: A numpy array of images.
    - actual_labels: A numpy array of actual labels.
    - predicted_classes: A numpy array of predicted classes/labels.
    - title: The title for the visualization.
    """
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    indices = np.random.choice(len(images), size=16, replace=False)
    for ax, idx in zip(axes.flatten(), indices):
        image = images[idx]
        ax.imshow(image)
        actual_label = "Parasitized" if actual_labels[idx] == 0 else "Uninfected"
        predicted_label = "Parasitized" if predicted_labels[idx] == 0 else "Uninfected"
        ax.set_title(f"Actual: {actual_label}\nPredicted: {predicted_label}")
        ax.axis("off")
    plt.suptitle(f'{title} - Sample Images with Labels')
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss from the model training history.
    
    Args:
    - history: A History object returned by the fit method of a Keras model, containing training and validation metrics.
    """
    # Convert the history dictionary to a DataFrame
    df = pd.DataFrame(data=history.history)
    display(df)

    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot training and validation accuracy on the primary y-axis
    ax1.plot(df['binary_accuracy'], label='Train Accuracy', color='blue')
    ax1.plot(df['val_binary_accuracy'], label='Validation Accuracy', color='green')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='upper left', bbox_to_anchor=(0., 1))

    # Create a second y-axis to plot training and validation loss
    ax2 = ax1.twinx()
    ax2.plot(df['loss'], label='Train Loss', color='magenta')
    ax2.plot(df['val_loss'], label='Validation Loss', color='red')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper left', bbox_to_anchor=(0., 0.89))

    # Add grid and title
    ax1.grid(True, 'both')
    plt.title('Training and Validation Metrics')

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Show the plot
    plt.show()
