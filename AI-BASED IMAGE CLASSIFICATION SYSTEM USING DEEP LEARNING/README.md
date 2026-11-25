# AI-Based Image Classification System Using Deep Learning

## 1. Project Overview

This project develops an AI-based image classification system using Deep Learning techniques. It uses a custom Convolutional Neural Network (CNN) and Transfer Learning with MobileNetV2 to classify images into different categories.

The project uses the CIFAR-10 dataset and compares the performance of the custom CNN model with a pre-trained MobileNetV2 model.

## 2. Project Objective

The main objective is to develop and evaluate deep learning models for image classification.

The specific objectives are:

- Load and explore an image classification dataset.
- Preprocess images and normalize pixel values.
- Build and train a custom CNN model.
- Implement Transfer Learning using MobileNetV2.
- Evaluate and compare both models.
- Predict labels for unseen test images.
- Visualize correctly classified and misclassified images.
- Analyze model performance, limitations, and possible improvements.

## 3. Dataset Description

**Dataset:** CIFAR-10  
**Dataset source:** https://www.cs.toronto.edu/~kriz/cifar.html

CIFAR-10 contains 60,000 colour images. Each image has dimensions of 32 × 32 pixels and belongs to one of 10 classes.

### Dataset distribution

| Dataset split | Number of images |
|---|---:|
| Training images | 50,000 |
| Testing images | 10,000 |
| Total images | 60,000 |

**Image dimensions:** 32 × 32 × 3  
**Number of classes:** 10

### Image classes

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## 4. Methodology

The project follows these stages:

1. **Data loading:** Load the CIFAR-10 dataset using TensorFlow/Keras.
2. **Data exploration:** Inspect dataset dimensions, labels, class distribution, and sample images.
3. **Preprocessing:** Normalize pixel values between 0 and 1 for the custom CNN.
4. **Data splitting:** Divide the original training data into training and validation sets while retaining the official test set.
5. **CNN development:** Build, compile, and train a custom CNN model.
6. **Transfer learning:** Load MobileNetV2 with ImageNet pre-trained weights, resize CIFAR-10 images to 96 × 96 pixels, apply MobileNetV2 preprocessing, and add a custom classification head.
7. **Evaluation:** Evaluate both models using accuracy, loss, confusion matrix, and classification report.
8. **Prediction and visualization:** Predict unseen test images and visualize correct and incorrect predictions.
9. **Comparison:** Compare the CNN and MobileNetV2 results and discuss their performance.

## 5. Model Architecture

### 5.1 Custom CNN

The custom Convolutional Neural Network contains the following main layers:

- Input Layer
- Convolutional Layers
- Max Pooling Layers
- Dropout Layers
- Flatten Layer
- Dense Layer
- Softmax Output Layer

The model is compiled using:

- **Optimizer:** Adam
- **Loss function:** Sparse Categorical Crossentropy
- **Evaluation metric:** Accuracy

### 5.2 MobileNetV2 – Transfer Learning

MobileNetV2 is used as a pre-trained model with ImageNet weights.

Implementation steps:

- Load MobileNetV2 with ImageNet pre-trained weights.
- Resize CIFAR-10 images to 96 × 96 pixels.
- Apply MobileNetV2 preprocessing.
- Freeze the pre-trained base model layers.
- Add a custom classification head for the 10 CIFAR-10 classes.
- Train and evaluate the model.

## 6. Project Tasks

### Task 1 – Data Preparation

- Load and explore the CIFAR-10 dataset.
- Inspect dataset dimensions and class labels.
- Visualize sample images.
- Check image dimensions.
- Normalize pixel values between 0 and 1.
- Split the original training data into training and validation sets.
- Retain the official test dataset.
- Analyze class distribution.

### Task 2 – CNN Model Development

- Develop a custom CNN model.
- Compile the model using Adam optimizer and Sparse Categorical Crossentropy.
- Train the model.
- Evaluate using accuracy, loss, confusion matrix, and classification report.

### Task 3 – Transfer Learning Using MobileNetV2

- Load MobileNetV2 with ImageNet pre-trained weights.
- Resize images to 96 × 96 pixels.
- Apply MobileNetV2 preprocessing.
- Freeze the pre-trained base model.
- Add a custom classification head.
- Train and evaluate the model.
- Compare its performance with the custom CNN.

### Task 4 – Prediction and Visualization

- Predict labels for unseen test images.
- Display actual and predicted labels.
- Visualize correctly classified images.
- Visualize misclassified images.
- Plot training and validation accuracy.
- Plot training and validation loss.

### Task 5 – Model Analysis and Documentation

- Compare CNN and MobileNetV2 performance.
- Compare accuracy and loss.
- Discuss advantages and limitations of both models.
- Describe implementation challenges.
- Identify real-world applications.
- Suggest future improvements.

## 7. Results and Analysis

Both models should be evaluated using the results produced by the completed notebook.

### Model performance comparison

| Evaluation metric | Custom CNN | MobileNetV2 |
|---|---:|---:|
| Test accuracy | Enter actual result | Enter actual result |
| Test loss | Enter actual result | Enter actual result |

**Important:** Replace the placeholders with the actual values printed by your notebook. Do not enter estimated results.

### Evaluation and visual analysis

Include the following outputs from the notebook when available:

- Training and validation accuracy graph.
- Training and validation loss graph.
- Confusion matrix.
- Classification report.
- Correctly classified image examples.
- Misclassified image examples.

### Analysis

Use the actual evaluation results to explain:

- Which model achieved higher test accuracy.
- Which model had lower test loss.
- How the confusion matrices compare.
- Which classes were classified correctly or incorrectly.
- Any observed difference between the custom CNN and MobileNetV2.

Do not claim that one model performed better until the actual results have been checked.

## 8. Real-World Applications

Image classification techniques can be applied in:

- Medical image analysis.
- Face and object recognition.
- Self-driving vehicles.
- Industrial quality inspection.
- Security surveillance.
- Retail product classification.
- Agricultural image analysis.

## 9. Advantages and Limitations

### Advantages of Custom CNN

- Learns image features directly from the dataset.
- Architecture can be customized.
- Suitable for image classification tasks.

### Advantages of MobileNetV2

- Uses pre-trained ImageNet weights.
- Reduces training requirements through transfer learning.
- Designed for efficient image processing.
- Can be adapted to new image classification tasks.

### Limitations

- Model performance depends on dataset quality.
- Training deep learning models can require significant computing resources.
- Small image dimensions may limit feature details.
- Visually similar classes may be misclassified.
- Transfer learning may require additional fine-tuning.

## 10. Future Improvements

- Apply data augmentation.
- Increase training epochs where appropriate.
- Fine-tune selected MobileNetV2 layers.
- Experiment with other pre-trained models.
- Use learning-rate scheduling.
- Improve model generalization.
- Deploy the trained model as a web application.

## 11. Project Folder Structure

The following is a suggested structure. Keep only the folders and files that are actually present in your repository.

```text
MODULE--5-END-ASSIGNMENT-DEEP-LEARNING-APPLICATIONS/
│
├── AI BASED IMAGE CLASSIFICATION SYSTEM USING DEEP- LEARNING.ipynb
├── README.md
├── requirements.txt
├── models/
│   └── mobilenetv2_cifar10.keras
└── outputs/
    ├── accuracy_graph.png
    ├── loss_graph.png
    ├── confusion_matrix.png
    └── prediction_results.png
```

Create the `models` and `outputs` folders when the corresponding files have been generated. CIFAR-10 can be loaded directly through TensorFlow/Keras, so a separate dataset folder is optional.

## 12. How to Run the Project

1. Open Google Colab or Jupyter Notebook.
2. Upload or open `AI BASED IMAGE CLASSIFICATION SYSTEM USING DEEP- LEARNING.ipynb`.
3. Select a suitable runtime. A GPU runtime can help with model training.
4. Install the required libraries if necessary.
5. Run the notebook cells in order, from Task 1 to Task 5.
6. Wait for dataset loading and model training to finish.
7. Review the evaluation results and visualizations.
8. Save the trained model and output graphs.
9. Update the Results and Analysis section with the actual test metrics.

## 13. Conclusion

This project demonstrates an image classification workflow using Deep Learning. A custom CNN model and a pre-trained MobileNetV2 model are implemented using the CIFAR-10 dataset.

The project covers image data preparation, preprocessing, CNN development, transfer learning, model evaluation, prediction, visualization, and performance comparison.

The final conclusion about which model performs better should be based on the actual test accuracy, test loss, confusion matrix, and classification report generated by the notebook.

## 14. GitHub Repository

**Repository name:** `MODULE--5-END-ASSIGNMENT-DEEP-LEARNING-APPLICATIONS`

**GitHub owner:** `priyanithi876-ship-it`

**Repository link:**  
https://github.com/priyanithi876-ship-it/MODULE--5-END-ASSIGNMENT-DEEP-LEARNING-APPLICATIONS

## 15. Project Status

The README documents the five project tasks and the expected workflow. Run the complete notebook, verify that each task and its outputs are present, and replace all result placeholders with actual values before final submission.
