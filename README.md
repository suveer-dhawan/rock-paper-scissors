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

## ⚙️ Data Preprocessing & Augmentation
To enhance model robustness and prevent overfitting, a strategic data preprocessing pipeline was implemented. This ensured that the models were trained on diverse and appropriately transformed image data.

1. Custom Train/Validation Split
The original ```train folder``` (containing 2,520 images) was meticulously split to create distinct datasets for training and validation:

   * Actual Training Set: 2,016 images (80% of the original training data).

   * Validation Set: 504 images (20% of the original training data).

   This custom split was crucial because the original Kaggle dataset, while providing ```train``` and ```test splits```, had only a very small validation subset. By creating a larger, dedicated validation set, we could more reliably monitor model performance during training and detect overfitting early. The split was carefully executed to maintain the original class proportions (equal numbers of 'rock', 'paper', 'scissors' images) across both the training and validation sets, preventing any unintended class imbalance in the training process.

2. Image Transformations
Different sets of transformations were applied to the training, validation, and test datasets to suit their respective purposes:

   * Training Data Transformations: To maximize data diversity and improve generalization, a comprehensive suite of data augmentation techniques was applied:
   
      * ```RandomResizedCrop(224)```: Randomly crops and resizes the image to 224x224 pixels. This simulates variations in scale and position of the hand gestures.
   
      * ```ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)```: Randomly changes the brightness, contrast, saturation, and hue of the images. This helps the model become invariant to lighting conditions.
   
      * ```RandomHorizontalFlip()```: Randomly flips the image horizontally. This is a common and effective augmentation for many image tasks as hand gestures can appear mirrored.
   
      * ```RandomRotation(15)```: Randomly rotates the image by up to 15 degrees. This accounts for slight variations in hand orientation.
   
      * ```RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))```: Applies random affine transformations, including translations (shifts) and scaling. This further enhances robustness to position and size variations.
   
      * ```GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))```: Applies a Gaussian blur, simulating slight out-of-focus conditions or different image qualities.
   
      * ```ToTensor()```: Converts the image from a PIL Image or NumPy array to a PyTorch ```Tensor```.
   
      * ```Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])```: Normalizes the pixel values using the ImageNet means and standard deviations. This is a common practice when using pre-trained models or similar architectures, as it helps standardize the input distribution, which can speed up convergence and improve performance.
    
   * Validation & Test Data Transformations: For evaluation datasets, consistency is key, so only deterministic transformations were applied to accurately assess performance on unseen data:
   
      * ```Resize(224)```: Resizes the image to 224x224 pixels.
   
      * ```CenterCrop(224)```: Crops the center of the image to 224x224 pixels, ensuring consistent input dimensions.
   
      * ```ToTensor()```: Converts the image to a PyTorch ```Tensor```.
   
      * ```Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])```: Normalizes pixel values using ImageNet statistics, matching the training data normalization.
 
3. Custom Dataset Class (```RockPaperScissorsDataset```)
A custom PyTorch ```Dataset``` class, ```RockPaperScissorsDataset```, was implemented to efficiently handle the loading of images and their corresponding labels. This class:

   * Inherits from ```torch.utils.data.Dataset```.

   * Reads image file paths from specified directories.

   * Maps string labels to integer IDs: ```paper: 0```, ```rock: 1```, ```scissors: 2```. This numerical representation is required for training with ```nn.CrossEntropyLoss```.

   * Applies the defined image transformations to each image upon retrieval.

4. DataLoaders
```torch.utils.data.DataLoader``` instances were configured for all three datasets (training, validation, and test) to manage data loading in mini-batches:

   * ```batch_size=32```: A standard batch size that offers a good balance between training stability and memory usage.

   * ```num_workers=2```: Specifies the number of subprocesses to use for data loading. This allows for parallel data loading, preventing bottlenecks during GPU training.

   * ```pin_memory=True```: This optimizes data transfer to the GPU by loading data into pinned (page-locked) memory, which is directly accessible by the GPU.

   * Shuffling: The training data ```DataLoader``` was set to ```shuffle=True``` to randomize the order of samples in each epoch, which is crucial for preventing the model from learning the order of the data and improving generalization. Validation and test DataLoaders had ```shuffle=False``` to ensure consistent evaluation.
  

## 🏛️ Model Architectures
We developed and evaluated two distinct Convolutional Neural Network (CNN) models. The second, a "Deeper CNN," was further refined with advanced training techniques to achieve optimal performance.

1. Simple CNN (SimpleCNN)
   Our initial venture was a foundational CNN designed for straightforward image classification. It's built on a series of basic convolutional blocks:
   
   * Convolutional Blocks: The model consists of four sequential blocks. Each block typically includes:
   
      * ```nn.Conv2d```: A 2D convolutional layer with a ```3x3``` kernel and ```padding=1``` to preserve spatial dimensions initially.
      
      * ```nn.ReLU```: A Rectified Linear Unit activation function, introducing non-linearity.
      
      * ```nn.MaxPool2d```: A max-pooling layer with a ```2x2``` kernel and ```stride=2``` to progressively reduce the spatial dimensions of the feature maps.
   
   * Channel Progression: The number of output channels incrementally increases through the layers:
   
      * Starts with ```in_channels=3``` (for RGB images).
      
      * Progresses through ```16``` -> ```32``` -> ```64``` -> ```128``` ```out_channels```. This allows the network to learn increasingly complex features.
   
   * Feature Map Reduction: Each MaxPool2d operation effectively halves the spatial dimensions of the feature maps. For example, an input of ```224x224``` pixels would be reduced to ```112x112```, then ```56x56```, ```28x28```, and finally 14x14.
   
   * Fully Connected Layers: After the convolutional layers, the flattened features are fed into two dense (fully connected) layers for final classification. The last layer has 3 output neurons, corresponding to the three classes: Rock, Paper, and Scissors.
  
2. Deeper CNN (```DeeperCNN```)
   Building on the ```SimpleCNN```, the ```DeeperCNN``` is an enhanced and more robust architecture. It uses ```nn.Sequential``` for better modularity and integrates advanced regularization techniques to improve stability and prevent overfitting.
   
   * Similar Core Structure: It retains the four convolutional blocks, much like the ```SimpleCNN```.
   
   * Key Enhancements per Block: Each convolutional block in the ```DeeperCNN``` is significantly improved with:
   
      * ``nn.BatchNorm2d`` (Batch Normalization): Placed immediately after each ```Conv2d``` layer. Batch normalization standardizes the inputs to layers, leading to improved training stability, faster convergence, and often better performance by reducing internal covariate shift.
      
      * ```nn.Dropout2d(p=0.25)```: Applied after each ```MaxPool2d``` operation. This form of dropout randomly sets a fraction (```p=0.25```) of input channels to zero during training. It's a powerful regularization technique that prevents the network from relying too heavily on specific features, thus mitigating overfitting.
      
   * Enhanced Fully Connected Layers: The dense layers at the end of the network are also more sophisticated:
   
      * They now incorporate ```nn.BatchNorm1d``` (Batch Normalization for 1D inputs) and ```nn.ReLU``` activation. This helps manage the potentially larger and more complex feature vectors emerging from the deeper convolutional layers, ensuring stable and effective classification.

   These architectural choices were designed to progressively handle more intricate patterns within the image data, leading to the superior performance observed with the DeeperCNN model.
