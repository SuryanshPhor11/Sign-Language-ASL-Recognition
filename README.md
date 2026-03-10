# Sign Language Recognition (ASL Alphabet)

A machine learning-based application for recognizing American Sign Language (ASL) alphabet letters using Convolutional Neural Networks (CNN) and computer vision techniques.

## 🎯 Project Overview

This project implements a deep learning model to recognize and classify ASL alphabet letters (A-Z) plus special gestures (del, nothing, space) in real-time. It combines computer vision, machine learning, and data processing to achieve accurate sign language recognition.

**Total Classes**: 29 (26 letters + 3 special gestures)
**Training Accuracy**: ~98%
**Validation Accuracy**: ~95%

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.x** - Primary programming language
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Computer vision library for image processing
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Data visualization
- **Scikit-learn** - Machine learning utilities (metrics, preprocessing)

### Database
- **SQLite/SQL** - Data storage and management for dataset metadata

### Additional Libraries
- **Jupyter Notebook** - Interactive development and documentation
- **PIL/Pillow** - Image processing

## 📋 Features

✅ Real-time ASL gesture recognition
✅ Support for 29 ASL classes (A-Z + del, nothing, space)
✅ CNN-based deep learning model
✅ Data augmentation for improved robustness
✅ Batch normalization and dropout for regularization
✅ L2 regularization to prevent overfitting
✅ Early stopping and learning rate reduction callbacks
✅ Comprehensive evaluation metrics (confusion matrix, classification report)

## 📊 Dataset

**Source**: ASL Alphabet Dataset
- **Training samples**: 69,600 images
- **Validation samples**: 13,920 images (20% split)
- **Testing samples**: 17,400 images
- **Image size**: 64×64 pixels
- **Format**: RGB images

### Data Structure
```
Dataset/
├── asl_alphabet_train/
│   └── asl_alphabet_train/
│       ├── A/
│       ├── B/
│       └── ... (29 folders total)
└── asl_alphabet_test/
    └── test/
```

## 🏗️ Model Architecture

```
Sequential Model:
├── Conv2D(64 filters, 3×3) + BatchNormalization
├── Conv2D(64 filters, 3×3) + BatchNormalization
├── MaxPooling2D(2×2)
├── Dropout(0.3)
├── Conv2D(128 filters, 3×3) + BatchNormalization
├── Conv2D(128 filters, 3×3) + BatchNormalization
├── MaxPooling2D(2×2)
├── Dropout(0.3)
├── Conv2D(256 filters, 3×3) + BatchNormalization
├── Conv2D(256 filters, 3×3) + BatchNormalization
├── MaxPooling2D(2×2)
├── Dropout(0.4)
├── Flatten()
├── Dense(512, L2 regularization) + BatchNormalization
├── Dropout(0.5)
└── Dense(29, softmax activation)

Total Parameters: 9,555,037
Trainable Parameters: 9,552,221
```

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | 98.86% |
| Validation Accuracy | 95.89% |
| Test Accuracy | ~93-95% |
| Total Epochs Trained | 25 |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip or conda

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/SuryanshPhor11/SignLanguageRecognition.git
cd SignLanguageRecognition
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
- Download from [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- Extract to the appropriate directory structure

## 💻 Usage

### Training the Model

```python
# Open SignLanguageRecognition.ipynb in Jupyter Notebook
jupyter notebook SignLanguageRecognition.ipynb
```

Run cells sequentially to:
1. Load and preprocess data
2. Create data generators with augmentation
3. Build and compile the model
4. Train the model
5. Evaluate performance

### Making Predictions

```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load trained model
model = load_model("SLR_final.h5")

# Load and preprocess image
image = cv2.imread('path/to/image.jpg')
image = cv2.resize(image, (64, 64))
image = image / 255.0
image = np.expand_dims(image, axis=0)

# Make prediction
prediction = model.predict(image)
predicted_class = np.argmax(prediction)
```

## 📁 Project Structure

```
SignLanguageRecognition/
├── SignLanguageRecognition.ipynb    # Main Jupyter notebook
├── SLR_final.h5                      # Trained model weights
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── Dataset/
│   ├── asl_alphabet_train/
│   └── asl_alphabet_test/
└── output/
    ├── training_history.png
    └── confusion_matrix.png
```

## 🔍 Key Implementation Details

### Data Augmentation
- Rotation: ±15 degrees
- Width/Height shift: 10%
- Zoom: 10%
- Brightness adjustment: 0.8-1.2

### Regularization Techniques
- **Batch Normalization**: Stabilizes training and allows higher learning rates
- **Dropout**: 0.3-0.5 to prevent overfitting
- **L2 Regularization**: 0.001 penalty on dense layer weights
- **Early Stopping**: Monitors validation loss with patience of 6 epochs
- **Learning Rate Reduction**: Reduces LR by 50% after 3 epochs without improvement

### Training Configuration
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 25 (with early stopping)
- **Validation Split**: 20%

## 📊 Results & Evaluation

### Confusion Matrix
The model shows excellent performance across all 29 classes with minimal confusion between similar gestures.

### Class Distribution
All classes are well-balanced in the dataset with 2,400 training samples per class.

## 🎓 Model Classes

```
Letters: A-Z (26 classes)
Special Gestures:
  - 'del': Delete gesture
  - 'nothing': No gesture detected
  - 'space': Space gesture
```

## 🔧 Customization

### Modify Model Architecture
Edit the `layers` in the Sequential model creation cell to experiment with different architectures.

### Adjust Data Augmentation
Modify parameters in `ImageDataGenerator`:
```python
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,  # Change rotation
    width_shift_range=0.1,  # Change width shift
    height_shift_range=0.1,  # Change height shift
    zoom_range=0.1,  # Change zoom
)
```

### Fine-tune Hyperparameters
```python
BATCH_SIZE = 32  # Change batch size
IMG_SIZE = 64    # Change image size
epochs = 25      # Adjust number of epochs
```

## 📚 Dependencies

See `requirements.txt` for complete list:
```
tensorflow>=2.10.0
opencv-python>=4.7.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
Pillow>=9.4.0
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🎯 Future Enhancements

- [ ] Real-time webcam integration for live recognition
- [ ] Support for dynamic gestures and sign sequences
- [ ] Mobile app deployment (TensorFlow Lite)
- [ ] API development for external applications
- [ ] Improved accuracy with transfer learning (MobileNet, EfficientNet)
- [ ] Support for continuous sign language sentences
- [ ] Multi-hand detection and recognition
- [ ] Database integration for gesture logging

## 📧 Contact & Support

**Author**: Suryansh Phor
**GitHub**: [@SuryanshPhor11](https://github.com/SuryanshPhor11)

For issues, questions, or suggestions, please open an issue on GitHub.

## 🙏 Acknowledgments

- Kaggle for the ASL Alphabet Dataset
- TensorFlow and Keras teams
- OpenCV community
- Python data science community

## 📖 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- [Deep Learning Best Practices](https://arxiv.org/abs/1409.1556)

---

**Status**: Active Development ✅
**Last Updated**: 2026-03-10
**Python Version**: 3.7+
