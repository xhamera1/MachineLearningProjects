
## Lab 11: Image Analysis with Convolutional Neural Networks (CNNs)

### Overview

This lab focused on **image analysis using Convolutional Neural Networks (CNNs)**. The exercises involved working with the **`tf_flowers` dataset**, loaded via Tensorflow Datasets. Key aspects included data preprocessing with the Dataset API, building a CNN from scratch, and implementing **transfer learning** using a pre-trained Xception model.

Key activities included:

* Loading the `tf_flowers` dataset and preparing it using the `tf.data.Dataset` API, including image resizing and batching.
* Designing, building, and training a **simple CNN model** for image classification, including normalization and appropriate convolutional/dense layers.
* Evaluating the simple CNN's accuracy on training, validation, and test sets.
* Implementing **transfer learning** by utilizing the pre-trained **Xception model**, adapting its input preprocessing and adding custom classification layers.
* Training the transfer learning model in stages (freezing base layers then fine-tuning) and evaluating its performance.
