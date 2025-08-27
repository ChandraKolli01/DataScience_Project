Pneumonia Detection Using Deep Learning 
===========================================

Overview
--------
This project aims to classify chest X-ray images into two categories: Normal and Pneumonia. 
It uses deep learning techniques including a custom Convolutional Neural Network (CNN) and transfer learning with MobileNetV2 to build models capable of detecting pneumonia from grayscale X-ray scans.


Dataset
---------
Dataset Link: https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/e9a18c27-67da-40a2-8f48-1c12f03de39d 

The dataset used is the publicly available Chest X-ray Images (Pneumonia) dataset. 
It contains images organized into the following categories:

- Normal: X-ray images of healthy patients
- Pneumonia: X-ray images of patients diagnosed with pneumonia

Dataset is split into:
- Training set
- Validation set
- Test set


Features
--------
- Data preprocessing and augmentation using ImageDataGenerator
- Balanced dataset sampling to reduce class imbalance
- Model training and evaluation for:
  * Custom CNN architecture
  * Transfer Learning with MobileNetV2 (adapted for grayscale images)
- Visualization of training history and dataset distribution
- Performance evaluation on validation and test sets


Requirements
------------
The following Python libraries are required:

- Python 3.x
- TensorFlow / Keras
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn


Installation and Setup
----------------------
1. Clone this repository or upload the notebook to Google Colab.
2. Mount Google Drive to access the dataset:
   from google.colab import drive
   drive.mount('/content/drive')
3. Update dataset paths in the notebook if necessary, using the dataset link provided above.


How to Run
----------
1. Open the notebook (23070982_ProjectV1.ipynb) in Google Colab.
2. Run all cells sequentially to:
   - Load and preprocess the dataset
   - Train the CNN and MobileNetV2 models
   - Evaluate results on validation and test sets
3. Visualize results such as accuracy, loss curves, and confusion matrices.


Model Architectures
-------------------
1. Custom CNN
   - Input: 150x150 grayscale images
   - Layers: Convolution → Batch Normalization → MaxPooling → Dense → Dropout → Output
   - Output: Binary classification (Normal vs Pneumonia)

2. MobileNetV2 (Transfer Learning)
   - Pretrained on ImageNet
   - Adapted to handle grayscale input
   - Custom classification head with dense and dropout layers


Results
-------
The models are evaluated using:
- Training and validation accuracy
- Training and validation loss
- Confusion matrix on test set
- Precision, Recall, and F1-score

