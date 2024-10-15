# LeafGuard: Plant Disease Detection from Leaf Images

LeafGuard is a deep learning-based application that detects plant diseases from leaf images. By leveraging convolutional neural networks (CNNs) and image classification techniques, this project aims to help farmers monitor crop health and take timely action to prevent yield loss.

## Overview

Plant diseases significantly affect crop yield and quality, leading to losses for farmers and the agriculture industry. LeafGuard offers an efficient solution to detect diseases in plants using deep learning models. By analyzing leaf images and identifying visual patterns associated with different plant diseases, this project can help predict diseases early, enabling farmers to implement necessary precautions.

## Project Objectives

- Use deep learning (CNN) models to classify plant leaf images into different disease categories.
- Apply image preprocessing and enhancement techniques to improve the model’s performance.
- Address challenges such as class imbalance and optimize model performance through hyperparameter tuning.
- Evaluate the model’s accuracy, sensitivity, and specificity.
- Provide visual insights into the model’s predictions.

## Dataset

The **PlantVillage Dataset** is used in this project. It includes images of healthy and diseased plant leaves across multiple crop species. The dataset is pre-labeled with the type of disease or healthy condition.

### Dataset Link:
[PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

## Technologies Used

- Python
- TensorFlow / PyTorch
- Keras
- OpenCV (for image preprocessing)
- Matplotlib / Seaborn (for visualization)

## Features

- **Image Preprocessing**: Resize, normalize, and augment images to improve the robustness of the model.
- **Convolutional Neural Networks (CNN)**: Designed to extract patterns and features from leaf images, enabling accurate disease classification.
- **Transfer Learning**: Integrate pre-trained models like VGG16, ResNet, or Inception for better performance.
- **Class Imbalance Handling**: Apply techniques such as oversampling, undersampling, and class weights to manage imbalance in the dataset.
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, sensitivity, specificity, and confusion matrix are used to evaluate the model’s performance.

## Project Structure

```
├── data/                 # Directory for the dataset
├── models/               # Pre-trained and saved models
├── scripts/              # Python scripts for training, evaluation, and preprocessing
├── notebooks/            # Jupyter notebooks for exploration and experimentation
├── images/               # Sample images for visualization
├── results/              # Evaluation metrics and graphs
├── requirements.txt      # Required dependencies
└── README.md             # Project documentation
```

## Learning Outcomes

- Learn image preprocessing and augmentation techniques to enhance model performance.
- Design and implement CNN architectures for image classification.
- Experience the use of transfer learning for improved model accuracy.
- Learn how to handle imbalanced datasets in a deep learning project.
- Understand the model optimization process using hyperparameter tuning.
- Evaluate and visualize the model’s predictions and decision-making process.

## Real-World Applications

- **Farmers**: Early detection of plant diseases for better crop management.
- **Agricultural Companies**: Use disease detection for yield prediction and improved management strategies.
- **Government**: Provide timely advice to farmers to prevent crop loss and manage yield more effectively.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.