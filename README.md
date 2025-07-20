# Rock Paper Scissors Image Classification with CNNs ✊📄✂️

This repository showcases a deep learning project focused on building and training Convolutional Neural Networks (CNNs) from scratch to classify hand gestures as **Rock**, **Paper**, or **Scissors**. The project progresses through different model architectures and training strategies, demonstrating best practices in data preprocessing, model development, and rigorous performance evaluation.

---

## 🎯 Project Goals

The core objectives of this project were to:

* **Prepare Image Data**: Implement robust data loading and augmentation strategies for a given image dataset.
* **Design Custom CNNs**: Develop and compare distinct CNN architectures (a "Simple CNN" and a "Deeper CNN") for efficient image feature extraction and classification.
* **Optimize Training**: Set up efficient training loops, progressively integrating advanced techniques like **learning rate scheduling** and **early stopping** to enhance model performance and stability.
* **Evaluate Performance**: Thoroughly assess model accuracy, generalization capabilities, and internal consistency through a suite of comprehensive tests on unseen images.


## 📊 Dataset

This project utilizes the **Rock Paper Scissors Dataset** from Kaggle, specifically designed for image classification tasks.

* **Dataset Source**: [Rock Paper Scissors Dataset on Kaggle](https://www.kaggle.com/datasets/sanikamal/rock-paper-scissors-dataset)

### Dataset Overview

The dataset structure is well-organized and pre-split, simplifying the initial setup:

* **Total Images**: 2,892
* **Training Images**: 2,520 (87.1% of total)
    * Rock: 840 images
    * Paper: 840 images
    * Scissors: 840 images
    * *(This indicates a perfectly balanced training set, preventing class bias.)*
* **Test Images**: 372 (12.9% of total)
* **Validation Images**: The original dataset has a small validation set. For this project, the `train` folder was further split to create a more robust **validation set of 504 images (20% of 2,520)**, with the remaining 2,016 images used for actual training.

### Image Characteristics

Sample images reveal consistent characteristics conducive to a robust model:

* **Controlled Environment**: Images feature clear hand gestures against a white background with minimal shadows.
* **Diversity**: The dataset includes diverse representations, such as multiple skin tones.
* **Dimensions**: All images are 300x300 pixels, offering a good balance between detail preservation and computational manageability.


## 💻 Technical Stack & Implementation
The project leverages a robust set of Python libraries and a specialized development environment to facilitate efficient deep learning model development and training.

### Development Environment
* Google Colab Pro: Utilized for its free access to GPU acceleration (specifically NVIDIA T4 GPUs) and extended runtime, crucial for training deep learning models efficiently.

  * This environment allowed for faster experimentation and reduced computational overhead compared to local CPU-only setups.

### Core Libraries

* PyTorch: The primary deep learning framework. Its flexibility and dynamic computation graph were essential for building, training, and evaluating custom CNN architectures.

  * ```torch.nn```: Used extensively for defining neural network layers (e.g., ```Conv2d```, ```ReLU```, ```MaxPool2d```, ```BatchNorm2d```, ```BatchNorm1d```, ```Dropout2d```, ```Linear```).

  * ```torch.optim```: Employed for implementing optimization algorithms, specifically ```AdamW``` (Adam with weight decay) for efficient parameter updates.

  * ```torchvision.transforms```: Critical for image preprocessing, data augmentation (e.g., resizing, normalization, random flips, rotations), and converting images to PyTorch tensors.

  * ```torchvision.datasets```: Used for convenient loading of image datasets, especially ```ImageFolder```.

  * ```torch.utils.data.DataLoader```: For efficient batching and loading of data during training and evaluation.

* NumPy: Fundamental for numerical operations, especially for handling arrays and mathematical computations.

* Matplotlib: Used for visualizing training metrics (loss and accuracy curves) and sample image data to gain insights into model performance and data characteristics.

* tqdm: Provided progress bars for training loops, offering real-time feedback on the training process.

* PIL (Pillow): Utilized by ```torchvision.transforms``` for image manipulation and loading.

* torchsummary: A helpful utility for printing a summary of PyTorch models, showing layer types, output shapes, and parameter counts, aiding in debugging and understanding model architecture.

* scikit-learn: Used for evaluating model performance metrics beyond simple accuracy, such as precision, recall, and F1-score, providing a more comprehensive assessment.

### Key Implementation Techniques
* Custom CNN Architectures: Two distinct CNN models ("Simple CNN" and "Deeper CNN") were designed from scratch, exploring different depths and complexities to identify the optimal architecture for the task.

* Data Augmentation: Extensive data augmentation techniques were applied to the training dataset to increase its diversity, reduce overfitting, and improve the model's generalization capabilities.

* Learning Rate Scheduling: ```ReduceLROnPlateau``` was implemented, automatically decreasing the learning rate when the validation loss stopped improving. This helps the model converge more effectively.

* Early Stopping: A custom early stopping mechanism was implemented to halt training when the validation loss did not improve for a specified number of epochs, preventing overfitting and saving computational resources.

* Loss Function: ```CrossEntropyLoss``` was used as the criterion for training, suitable for multi-class classification problems.

* Optimizer: The ```AdamW``` optimizer was chosen for its adaptive learning rates and decoupled weight decay, which generally leads to better generalization performance in deep learning.

