The project focuses on training models, performing exploratory data analysis, and providing utility functions for visualization and data preprocessing.

## Repository Structure

- model_training.ipynb: Jupyter notebook for training various models.
- cnn-model_training.ipynb: Jupyter notebook specifically for training Convolutional Neural Network (CNN) models.
- exploratory_data_analysis.ipynb: Jupyter notebook for performing exploratory data analysis on the dataset.
- visualization_utils.py: Python script containing functions for visualizing data.
- dataset_utils.py: Python script with functions for preprocessing the dataset.

## Setup

To run the notebooks and scripts in this repository, follow these steps:

1. Clone this repository:
    git clone https://github.com/your-username/mtech-ai-dse-project.git
    cd mtech-ai-dse-project

2. Create a virtual environment and activate it:
    python3 -m venv venv
    source venv/bin/activate

3. Install the required dependencies:
    pip install -r requirements.txt

4. Ensure you have Docker installed and set up the environment using the provided Docker command:
    sudo docker run -it --gpus all -v /nfsl/tensorflow:/tf/notebooks -p 8888:8888 tensorflow-jupyter

## Usage

### Model Training

- Open model_training.ipynb in Jupyter Notebook and follow the steps to train various models.
- Open cnn-model_training.ipynb for training CNN models.

### Exploratory Data Analysis

- Open exploratory_data_analysis.ipynb to perform exploratory data analysis on the dataset. This notebook includes data visualization and statistical analysis.

### Utilities

- visualization_utils.py: Contains functions to visualize sample images, image statistics, and class distributions.
- dataset_utils.py: Provides functions for preprocessing the dataset, including resizing images, normalization, and checking for missing or corrupt data.
- model_utils.py`: Python script with utility functions for model training and evaluation like plotting training history, confusion matrix, image mismatches etc.

## Code Details

### model_training.ipynb
This notebook includes the following steps:
- Loading and preprocessing the dataset.
- Defining and compiling the model.
- Training the model with the training data.
- Evaluating the model performance on the validation data.
- Saving the trained model.

### cnn-model_training.ipynb
This notebook focuses on training Convolutional Neural Network (CNN) models:
- Loading and preprocessing the dataset specific to CNN requirements.
- Defining a CNN architecture.
- Compiling the CNN model.
- Training the CNN model with the training data.
- Evaluating the CNN model performance on the validation data.
- Saving the trained CNN model.

### exploratory_data_analysis.ipynb
This notebook performs exploratory data analysis (EDA) including:
- Loading the dataset.
- Visualizing sample images.
- Analyzing the distribution of classes.
- Generating descriptive statistics.
- Creating various plots to understand the data characteristics.

### visualization_utils.py
This script contains functions to assist with data visualization:
- visualize_sample_images: Displays a grid of sample images with labels.
- compute_image_statistics: Computes and prints the mean and standard deviation of the images.
- plot_image_heatmap: Plots a heatmap of the first image using seaborn.
- visualize_class_distribution: Plots the distribution of classes in the dataset.

### dataset_utils.py
This script provides utility functions for dataset preprocessing:
- resize_and_normalize_images: Resizes images and normalizes pixel values.
- check_missing_data: Checks for any missing data in the dataset.
- validate_images: Identifies and handles corrupt images in the dataset.

### model_utils.py
- extract_labels_predictions: Extracts actual images, labels, and predicted labels from a tf.data.Dataset using a trained model.
- eval_visualize_performance: Evaluates the model's performance and visualizes key metrics including confusion matrix, accuracy, precision, recall, and F1 score.
- show_actuals_predictions: Visualizes sample images with their actual and predicted labels.
- plot_training_history: Plots training and validation accuracy and loss from the model training history.

### Optimization of CNN Model Training
The cnn-model_training.ipynb notebook has been optimized to use the new utility functions. Please check the updated notebook for the improved workflow and ensure that the changes work as expected.



## Contributing

If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

This structure clearly separates the setup, usage, and code details, making it easier for others to understand and use your project. Adjust any specific details as necessary.
