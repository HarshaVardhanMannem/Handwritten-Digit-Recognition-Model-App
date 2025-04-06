# Handwritten Digit Recognition

A complete neural network-based solution for handwritten digit recognition using TensorFlow and Streamlit. The project includes both the training pipeline and an interactive web application for real-time digit recognition.

## 🌟 Features

- **GPU-Accelerated CNN Model**: Convolutional Neural Network optimized for GPU training
- **Interactive Web UI**: Draw or upload digits for instant recognition
- **High Accuracy**: ~99.3% test accuracy on the MNIST dataset
- **Real-time Visualization**: See confidence scores and probability distributions
- **Data Augmentation**: Improved model generalization through augmentation techniques

## 📋 Project Structure

```
mnist-digit-recognition/
├── models/                   # Directory for saved models
│   └── mnist_cnn_best.h5     # Best model checkpoint during training
├── mnist_cnn_model.h5        # Final trained model
├── mnist_cnn_model.tflite    # TensorFlow Lite model for mobile/embedded
├── train_model.py            # CNN training script
├── app.py                    # Streamlit web application
└── README.md                 # Project documentation
```

## 🧠 Neural Network Architecture

The CNN architecture consists of:

1. **Three Convolutional Blocks**:
   - Each block contains two Conv2D layers with increasing filter sizes (32→64→128)
   - Batch normalization for stable training
   - MaxPooling for spatial dimension reduction
   - Dropout for regularization

2. **Dense Layers**:
   - 256 neurons with ReLU activation and L2 regularization
   - Output layer with 10 neurons (one per digit) and softmax activation
   
3. **Training Configuration**:
   - Adam optimizer with learning rate scheduling
   - Early stopping to prevent overfitting
   - Data augmentation with rotation, shifting, zooming

## 💻 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mnist-digit-recognition.git
   cd mnist-digit-recognition
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Requirements:
   ```
   tensorflow>=2.8.0
   numpy>=1.20.0
   matplotlib>=3.5.0
   scikit-learn>=1.0.0
   streamlit>=1.18.0
   pillow>=9.0.0
   streamlit-drawable-canvas>=0.9.0
   ```

## 🚀 Usage

### Training the Model

Run the training script to train the CNN model on the MNIST dataset:

```bash
python train_model.py
```

This script will:
- Download the MNIST dataset
- Preprocess the data
- Train the CNN model with GPU acceleration (if available)
- Evaluate model performance
- Save the trained model as `mnist_cnn_model.h5` and a TFLite version

### Running the Web Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

The application provides two main functionalities:
1. **Draw Digit**: Draw a digit on the canvas and predict
2. **Upload Image**: Upload an image containing a handwritten digit

## 📊 Performance

The model achieves excellent performance on the MNIST dataset:
- Training accuracy: ~99.5%
- Validation accuracy: ~99.4%
- Test accuracy: ~99.3%

## 🖥️ GPU Acceleration

The training script includes optimizations for GPU acceleration:
- Automatic GPU detection and configuration
- Memory growth settings to optimize GPU memory usage
- Mixed precision training for compatible GPUs
- TensorFlow data pipeline optimizations with prefetching

## 🌐 Web Application Details

The Streamlit web application offers a user-friendly interface with:

- **Drawing Canvas**: Draw digits directly in the browser
- **Image Upload**: Test with existing images of handwritten digits
- **Preprocessing Visualization**: See how images are processed before prediction
- **Confidence Scores**: Visualize model's confidence for each digit class
- **Top Predictions**: View top 3 most likely digits and their probabilities

## 📷 Screenshots

[Screenshots of the application would be placed here]

## 🔍 Future Improvements

- Implement batch prediction for multiple digits
- Add model interpretability tools (like Grad-CAM)
- Support for handwritten character recognition beyond digits
- Mobile deployment using TensorFlow Lite

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

HARSHA VARDHAN MANNEM
