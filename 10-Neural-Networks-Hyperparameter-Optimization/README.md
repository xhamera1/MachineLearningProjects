## Lab 10: Hyperparameter Tuning for Neural Networks

### Overview

This lab concentrated on **optimizing neural network performance** by tuning hyperparameters for the **California Housing dataset**. The core tasks involved investigating the effects of different hyperparameters (like learning rate, layer count, neuron count, and optimizer) and using automated tools for this search.

Key activities included:

* Observing how changes in hyperparameters impact learning.
* Comparing different optimization algorithms (e.g., Adam, SGD).
* Automating hyperparameter searches using:
    * `RandomizedSearchCV` from **scikit-learn**, with Keras models wrapped by **scikeras**.
    * **Keras Tuner** (e.g., `RandomSearch`).
* Utilizing callbacks like `EarlyStopping` for efficiency and `TensorBoard` for visualizing the tuning process.

The lab provided practical experience in systematically finding optimal hyperparameter settings for neural networks to improve model accuracy and efficiency.
