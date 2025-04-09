 # Content-based Image Retrieval System

This project implements a content-based image retrieval system using Streamlit for the user interface. It uses image descriptors and various distance functions to compare an uploaded image with a pre-calculated dataset of images.

## Project Structure

- `app.py`: Main file for the Streamlit user interface.
- `data_processing.py`: Script for processing image datasets and calculating signatures.
- `descriptor.py`: Implementation of image descriptors (GLCM, BIT, Haralick, combined).
- `distances.py`: Functions for calculating distances between descriptors.
- `images`: Folder containing the dataset images.

## Prerequisites

- Python 3.x
- Streamlit
- NumPy
- OpenCV
- scikit-image
- scikit-learn
- Pillow
- mahotas
- scipy

## Dependencies to Install

```bash
pip install numpy
pip install opencv-python
pip install scikit-learn
pip install scikit-image
pip install streamlit
pip install Pillow
pip install mahotas
pip install scipy
```

## Usage

### Step 1: Image Preprocessing

Before using the Streamlit application, it's necessary to preprocess the images and calculate their signatures. To do this, run the data_processing.py script:

```bash
python data_processing.py
```

This script generates the files `signatures_glcm.npy`, `signatures_bitdesc.npy`, `signatures_haralick_feat.npy`, and `signatures_bit_glcm_haralick.npy` that contain the image signatures.

### Step 2: Launch the Streamlit Application

After preprocessing the images, launch the Streamlit application:

```bash
streamlit run app.py
```

### Step 3: Using the Interface

The application offers three main modes:

#### Basic CBIR
- Upload an image using the interface
- Select a descriptor (GLCM, BIT, HARALICK, or BIT_GLCM_HARALICK) and a distance function (Manhattan, Euclidean, Chebyshev, Canberra) from the sidebar
- Specify the number of similar images to display
- The application will display the most similar images from the dataset

#### Advanced CBIR
- Upload an image using the interface
- Select a descriptor, distance function, and number of images
- Choose a classification algorithm (LDA, KNN, SVM, Random Forest, AdaBoost, Decision Tree, Naive Bayes)
- Toggle GridSearchCV for model optimization
- Select a data transformation method (No Transform, Rescale, Normalization, Standardization)
- The application will display performance metrics and similar images from the predicted class

#### Multimodal Search
- Enter a search term in the text input
- Specify the number of images to display
- The application will display images from folders containing the search term

A Settings mode is also available to toggle the night mode interface.

## File Details

### app.py

This file contains the code for the user interface. It allows for uploading an image, selecting parameters, and displaying similar images from the dataset. It includes three main functions:
- `cbir_basic()`: Basic image retrieval functionality
- `cbir_advanced()`: Advanced retrieval with machine learning models
- `multimodal()`: Text-based folder searching
- `settings()`: UI customization options

### data_processing.py

This script processes images from the images folder, calculates their descriptors, and saves the signatures to .npy files. It assigns class labels based on folder names.

### descriptor.py

This file implements various feature extraction methods:
- `glcm()`: Gray-Level Co-occurrence Matrix features
- `bitdesc()`: Bio-inspired Taxonomy features
- `haralick_feat()`: Haralick texture features
- `bit_glcm_haralick()`: Combined features from all descriptors

### distances.py

Contains implementations of distance metrics:
- `manhattan()`: Sum of absolute differences
- `euclidean()`: Square root of sum of squared differences
- `chebyshev()`: Maximum absolute difference
- `canberra()`: Weighted version of Manhattan distance

## Features

- Multiple feature descriptors for image analysis
- Various distance metrics for comparing images
- Machine learning models for image classification
- GridSearchCV for model parameter tuning
- Data transformation options
- Night mode UI toggle
- Text-based multimodal search
- Performance metrics visualization
