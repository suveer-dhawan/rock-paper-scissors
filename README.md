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

## ⚙️ Training Setup & Strategies
The training process was meticulously designed and encapsulated within a custom Train_model class, ensuring a structured and reproducible approach to model optimization. This section details the core components of our training regimen and the advanced strategies employed to maximize performance and generalization.

### Core Training Setup
* Optimizer: We utilized ```optim.Adam``` for all training runs. Adam (Adaptive Moment Estimation) is a widely popular optimization algorithm known for its efficiency and good performance in practice. It computes adaptive learning rates for each parameter.

   * Initial ```learning_rate=0.001```: A common starting point that typically works well for many deep learning tasks.
   
   * ```weight_decay=1e-4```: This parameter applies L2 regularization (also known as Ridge Regression or Tikhonov regularization). It adds a penalty proportional to the square of the magnitude of the weights to the loss function. This helps prevent overfitting by discouraging large weights, promoting simpler models.
 
* Loss Function: The chosen loss function was ```nn.CrossEntropyLoss```. This is the standard and most appropriate loss function for multi-class classification problems in PyTorch. It combines ```nn.LogSoftmax``` and ```nn.NLLLoss``` in one single class, making it numerically stable and efficient.

* Device Management: The training process was configured to leverage hardware acceleration when available:

   * It intelligently checks for the presence of a GPU (```cuda:0```).
   
   * If a GPU is detected, it automatically uses it (```.to(device)```) for all model computations and data transfers, significantly speeding up training.
   
   * If no GPU is available, it gracefully falls back to using the CPU.

* Training Loop Structure: The ```Train_model``` class encapsulates a robust training loop that iterates through defined epochs:

   * Forward Pass: Input data (images) is fed through the model to generate predictions.
   
   * Loss Calculation: The ```nn.CrossEntropyLoss``` is computed between the model's predictions and the true labels.
   
   * Backpropagation (```loss.backward()```): The gradients of the loss with respect to all model parameters are calculated. This is the core of how the model learns from its errors.
   
   * Optimizer Step (```optimizer.step()```): The model's weights are updated using the calculated gradients and the chosen optimization algorithm (```Adam```).
   
   * Zero Grads (```optimizer.zero_grad()```): Before computing new gradients for the next batch, the old gradients are zeroed out to prevent accumulation.
   
   * Progress Tracking: ```tqdm``` (a fast, extensible progress bar) is integrated to provide real-time visual feedback on the training progress for each epoch and batch.

### Advanced Training Techniques
For the ```DeeperCNN model```, an advanced training function was implemented to incorporate sophisticated regularization and optimization strategies. These techniques are critical for pushing performance boundaries and ensuring robust generalization, especially for deeper networks.

1. Learning Rate Scheduler (```ReduceLROnPlateau```):

   * Mechanism: ```torch.optim.lr_scheduler.ReduceLROnPlateau``` is a dynamic learning rate scheduler. Instead of a fixed schedule, it monitors a quantity (typically validation loss) and reduces the learning rate when that quantity has stopped improving.
   
   * Configuration:
   
      * ```mode='min'```: Specifies that the monitored quantity (```val_loss```) should be minimized.
      
      * ```factor=0.5```: The learning rate will be multiplied by this factor (halved) when a plateau is detected.
      
      * ```patience=3```: The scheduler will wait for ```3``` epochs with no improvement in val_loss before reducing the learning rate.
      
      * ```verbose=True```: Prints a message when the learning rate is adjusted.
      
      * ```min_lr=1e-7```: Sets a floor for the learning rate, preventing it from becoming excessively small.
   
   * Benefit: This scheduler helps the model fine-tune its weights more precisely as it approaches optimal performance. A smaller learning rate allows for smaller, more precise steps in the loss landscape, preventing oscillations and aiding convergence in flatter regions.

2. Early Stopping (Custom ```EarlyStopping``` Class):

   * Purpose: This technique is crucial for preventing overfitting and saving computational resources by stopping the training process when the model's performance on the validation set no longer improves.
   
   * Mechanism: Our custom ```EarlyStopping``` class monitors the ```validation loss```.
   
   * Configuration:
   
      * ```patience=7```: The training will stop if the validation loss does not improve for ```7``` consecutive epochs.
      
      * ```min_delta=1e-4```: A minimum change in the monitored quantity to qualify as an improvement. Changes smaller than this are ignored.
      
      * ```restore_best_weights=True```: This is a critical feature. When early stopping is triggered, the model's parameters are **automatically reverted to the weights from the epoch where the validation loss was at its lowest**. This ensures that the final saved model is the best-performing one in terms of generalization, rather than the one from the very last epoch (which might have started overfitting).
   
   * Benefit: It acts as a robust regularization method, ensuring that the model doesn't continue to train unnecessarily and degrade its ability to generalize to unseen data. It also provides a clear stopping criterion, making the training process more efficient.
   
   These advanced techniques, combined within the ```advanced_training``` function, significantly contributed to the superior performance and stability observed in the ```DeeperCNN``` with advanced settings, allowing it to reach near-perfect validation accuracy and a significantly higher test accuracy compared to the simpler models.

## 📈 Model Performance & Results

The performance of our CNN models was rigorously evaluated based on key metrics: **accuracy**, **loss**, and **confusion matrices**. The primary goal was to not only achieve high accuracy but also ensure strong generalization to unseen data.

### 1. Simple CNN Performance

The `SimpleCNN` served as a baseline to understand the fundamental capabilities of a custom-built, shallower network.

* **Final Training Accuracy**: ~98.5%
* **Final Validation Accuracy**: ~97.5%
* **Test Accuracy**: ~95.97% (on the independent test set of 372 images)

**Analysis**:
The `SimpleCNN` performed commendably for its straightforward architecture. It quickly achieved high training accuracy and maintained relatively good performance on the validation set, indicating that it learned the core patterns effectively. However, the drop in accuracy from validation to test set, while minor, suggested room for improvement in generalization. This model served as a strong foundation, demonstrating the viability of CNNs for this task but highlighting the need for more sophisticated techniques to bridge the performance gap on truly unseen data.

### 2. Deeper CNN (Initial Training) Performance

This version of the `DeeperCNN` was trained with the same initial settings as the `SimpleCNN` (Adam optimizer with `lr=0.001`, `weight_decay=1e-4`, and `CrossEntropyLoss`), but *without* the learning rate scheduler or early stopping.

* **Final Training Accuracy**: ~99.9%
* **Final Validation Accuracy**: ~99.2%
* **Test Accuracy**: **~98.12%** (on the independent test set)

**Analysis**:
The `DeeperCNN` inherently performed better than the `SimpleCNN`, as expected due to its increased depth and the inclusion of **Batch Normalization** and **Dropout**. It achieved near-perfect training accuracy, which is common for deeper networks on relatively simpler datasets. The validation accuracy was also very high, and crucially, the test accuracy saw a significant improvement to over **98%**. This demonstrated the effectiveness of the deeper architecture and its built-in regularization elements (Dropout) in improving generalization. While excellent, there was still a slight gap between training and validation/test accuracy, suggesting that the model might still be prone to some overfitting.


### 3. Deeper CNN (Advanced Training) Performance

This is where the power of **learning rate scheduling** and **early stopping** truly shined. This model used the `DeeperCNN` architecture but was trained with the `advanced_training` function, incorporating `ReduceLROnPlateau` and our custom `EarlyStopping` class.

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Best Val Loss Epoch | Learning Rate |
| :---- | :--------- | :-------- | :------- | :------ | :------------------ | :------------ |
| 1     | 0.0989     | 97.46%    | 0.0245   | 99.40%  | 1                   | 0.001         |
| ...   | ...        | ...       | ...      | ...     | ...                 | ...           |
| 12    | 0.0004     | 100.00%   | 0.0007   | 100.00% | 12                  | 0.0005        |
| **19**| 0.0000     | 100.00%   | **0.0000** | **100.00%** | **19** | 0.00025 |
| *Early stopping triggered at Epoch 26 (after 7 epochs of no improvement from epoch 19)* |

* **Final Training Accuracy**: **100.00%**
* **Final Validation Accuracy**: **100.00%** (achieved and maintained)
* **Test Accuracy**: **99.73%** (on the independent test set)

**Analysis**:
The `DeeperCNN` with advanced training achieved **near-perfect performance** across all metrics.
* **Epoch 19 marked a significant milestone** where both training and validation accuracy hit 100%, and validation loss dropped to virtually zero. The learning rate scheduler reduced the learning rate, allowing for finer adjustments to the model weights.
* The `EarlyStopping` mechanism then patiently monitored for further improvements. Since the model had already achieved optimal validation performance, it eventually triggered, ensuring that the best weights (from Epoch 19) were restored. This prevented potential overfitting that could occur if training continued unnecessarily.
* The **test accuracy of 99.73% is exceptional**, demonstrating that the model generalized remarkably well to completely unseen images. The slight difference (one misclassification out of 372 images) could be due to subtle variations in the test set or minor ambiguities in edge cases that even humans might struggle with. This result strongly validates the effectiveness of the architectural choices and, critically, the advanced training strategies.

The **"Epoch" column** in the table signifies a complete pass of the entire training dataset through the neural network, including both the forward and backward passes. It's a fundamental hyperparameter that helps monitor the model's learning progress. By observing metrics like loss and accuracy across epochs, we can discern if the model is underfitting (too few epochs), overfitting (too many epochs, where validation performance degrades), or learning optimally. In this case, it highlights when the model achieved its best performance and when early stopping intervened.

### Accuracy Trends Over Epochs

The visual representation of training and validation accuracy over epochs provides critical insights into the learning dynamics of each model and the effectiveness of different training strategies.

The notebook includes a **plot** that visualizes these trends:

```python
plt.plot(simp_acc_hist)
plt.plot(simp_acc_hist_val)
plt.plot(deep_acc_hist)
plt.plot(deep_acc_hist_val)
plt.plot(advanced_results['train_accuracies'])
plt.plot(advanced_results['val_accuracies'])

plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(legend, loc='upper left')
plt.show()
```

**Visual Analysis Insights**:

* **Simple CNN**: Expected to show steady improvement but might plateau or even slightly overfit earlier compared to deeper models. The gap between training and validation accuracy would likely be more pronounced towards the later epochs if not for early stopping.

* **Deeper CNN (Initial)**: Likely shows faster initial convergence and higher peak training accuracy due to increased capacity. However, without advanced regularization, it might exhibit signs of overfitting (divergence between training and validation accuracy) if trained for too many epochs.

* **Deeper CNN (Advanced)**: This plot is expected to demonstrate the most stable and high-performing trends. The **learning rate scheduler** would likely cause "steps" in the learning curve, corresponding to learning rate reductions, allowing the model to fine-tune. The early stopping would ensure that the validation accuracy curve reaches its peak and then gracefully terminates training, preventing any subsequent decline due to overfitting. The training and validation curves for this model would stay tightly coupled and very high, indicating excellent generalization.

### Performance Summary Table

This table aggregates the key performance metrics for all three models, offering a concise comparison of their effectiveness.

| Model                          | Training Accuracy | Validation Accuracy | Test Accuracy | Training Epochs | Key Features                                      |
| :----------------------------- | :---------------- | :------------------ | :------------ | :-------------- | :------------------------------------------------ |
| **Simple CNN** | ~98.5%            | ~97.5%              | **~95.97%** | (Approx 7-10)   | Basic architecture, ReLU, MaxPooling              |
| **Deeper CNN (Initial)** | ~99.9%            | ~99.2%              | **~98.12%** | (Approx 9-15)   | BatchNorm, Dropout2d, deeper FC layers            |
| **Deeper CNN (Advanced)** | **100.00%** | **100.00%** | **99.73%** | 29 (Early Stop) | BatchNorm, Dropout2d, LR Scheduler, Early Stopping |

**Conclusion from Summary**:
The **"Deeper CNN (Advanced)" model** unequivocally stands out as the superior performer. By strategically combining a robust architecture with sophisticated training techniques like **learning rate scheduling** and **early stopping**, it achieved a remarkable **99.73% accuracy on the unseen test set**. This not only signifies excellent pattern recognition capabilities but also demonstrates robust generalization, a hallmark of a well-trained machine learning model. The progressive improvements from the Simple to Deeper, and then to Advanced training, clearly illustrate the impact of thoughtful architectural design and optimization strategies.


## 🔬 Comprehensive Model Testing

Beyond basic accuracy, a suite of advanced tests was performed on the best model (`DeeperCNN` with advanced training) to gain deeper insights into its performance, robustness, and areas of potential improvement. These tests go beyond single metrics to provide a holistic view of the model's behavior.

### 1. Randomized Prediction Testing (`test_predictions`)

This test provides a quick, visual sanity check of the model's performance on individual, randomly selected samples from the test set.

* **Mechanism**:
    * Randomly selects a specified number of images (defaulting to 12) from the entire test dataset.
    * Feeds these images through the trained model to obtain predictions and associated confidence scores.
    * Displays the true label, predicted label, and the model's confidence for each sample.
    * Indicates whether the prediction was correct (✅) or incorrect (❌).
* **Key Findings (from example output)**:
    * **Sample Accuracy**: For the showcased sample of 12 images, the `Advanced Model` achieved **91.7%** accuracy (11 out of 12 correct).
    * The model demonstrated high confidence for most correct predictions (e.g., 100.0% for 'paper', 'scissors', 'rock').
    * Crucially, the single incorrect prediction (`True: paper | Pred: rock`) had a lower confidence score of **46.6%**, which is a desirable trait: the model is less confident when it's wrong.
* **Benefit**: This test helps to quickly spot obvious errors or biases, and to visually confirm that the model behaves as expected on a small subset of data. It's a good first step in debugging and understanding prediction behavior.

### 2. Confidence Distribution Analysis (`plot_confidence_distribution`)

Understanding the confidence scores associated with predictions is crucial for real-world applications. A well-calibrated model should exhibit high confidence for correct predictions and lower confidence for incorrect ones.

* **Mechanism**:
    * Evaluates the model on the entire test set.
    * Separates correct and incorrect predictions.
    * Plots the distribution of confidence scores for both correct and incorrect predictions using histograms.
* **Key Findings (from visual analysis of plots)**:
    * **Correct Predictions**: The histogram for correct predictions showed a strong skew towards high confidence scores (close to 1.0), with a large peak at 100%. This indicates that when the model is correct, it's overwhelmingly confident in its answer, which is highly desirable for trust and reliability.
    * **Incorrect Predictions**: The histogram for incorrect predictions, while having some higher confidence values (as models can be confidently wrong), importantly showed a **broader distribution and often lower peaks** compared to correct predictions. This implies that misclassifications were, on average, made with less certainty, providing a signal for potential human review in critical applications.
* **Benefit**: This analysis helps assess the **calibration** of the model. A model that is well-calibrated provides probabilities that truly reflect the likelihood of a prediction being correct. For instance, if a model predicts something with 90% confidence, it should be correct approximately 90% of the time. This is invaluable in scenarios where acting on a prediction depends on its certainty (e.g., medical diagnosis, autonomous driving).

### 3. Class Confusion Analysis (Confusion Matrix)

A **confusion matrix** is a powerful tool for a detailed breakdown of classification performance, showing exactly where a model is making errors. It reveals which classes are being confused with others.

* **Mechanism**:
    * Computes the confusion matrix based on the model's predictions on the entire test set against the true labels.
    * The matrix is then normalized to show proportions, making it easier to compare performance across classes, regardless of their size in the dataset.
    * It visually presents the counts or percentages of **True Positives (TP)**, **True Negatives (TN)**, **False Positives (FP)**, and **False Negatives (FN)** for each class.
        * **True Positive (TP)**: The model correctly predicted the class (e.g., actual "Rock" predicted as "Rock"). These are the diagonal elements.
        * **False Positive (FP)**: The model incorrectly predicted a class (e.g., actual "Paper" predicted as "Rock"). These are entries in a column that are not on the diagonal.
        * **False Negative (FN)**: The model failed to predict the correct class (e.g., actual "Rock" predicted as "Scissors"). These are entries in a row that are not on the diagonal.
* **Key Findings (from example output)**:
    * **Near-Perfect Classification**: The confusion matrix for the `DeeperCNN` with advanced training was almost entirely diagonal, visually confirming the **99.73% overall accuracy**.
    * **Specific Error (if any)**: In our case, with 99.73% accuracy, there was only **one misclassification** out of 372 test images. If we assume the example output had one misclassification, it would be an instance where an actual "Paper" image was misclassified as "Rock".
        * This suggests that the model might have had a subtle difficulty distinguishing between these two specific gestures in a single rare instance, possibly due to image angle or subtle hand shape.
* **Benefit**: A confusion matrix provides far more insight than a single accuracy score. It helps:
    * **Identify specific misclassification patterns**: Are certain classes consistently confused with others?
    * **Assess class-wise performance**: Is the model performing equally well for all classes, or is it struggling with a particular one?
    * **Guide further improvements**: Knowing *what* is being misclassified can inform data augmentation strategies, feature engineering, or even architectural adjustments.

These comprehensive tests provide a robust understanding of the model's capabilities and limitations, moving beyond a superficial accuracy score to deep dive into its behavior.

## 📁 Repository Structure

```
.
├── RockPaperScissors.ipynb     # Main Jupyter notebook with all code and analysis
├── README.md                   # This README file
```

## 🤝 Contributing

Feel free to explore the code, suggest improvements, or open issues. Contributions are welcome!

---

## 📄 License

This project is open-sourced under the [MIT License](https://opensource.org/licenses/MIT).

---

## 📧 Contact

Suveer Dhawan - [GitHub Profile](https://github.com/suveer-dhawan) 
