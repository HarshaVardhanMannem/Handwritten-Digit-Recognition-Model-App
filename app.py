import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import io
import base64
from datetime import datetime
import os

# Load the trained neural network model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_cnn_model.h5')

def preprocess_image(image, size=(28, 28)):
    """Preprocess the image for neural network prediction."""
    # Convert to grayscale
    image = image.convert('L')
    # Resize
    image = image.resize(size, Image.LANCZOS)
    # Invert colors (MNIST has white digits on black background)
    image = ImageOps.invert(image)
    # Convert to numpy array
    img_array = np.array(image)
    # Normalize
    img_array = img_array / 255.0
    # Reshape for CNN input (batch_size, height, width, channels)
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

def main():
    st.set_page_config(page_title="Neural Network Digit Recognition", layout="wide")
    
    st.title("Handwritten Digit Recognition - Neural Network")
    st.write("Draw a digit or upload an image of a handwritten digit (0-9)")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Upload Image", "About Model"])
    
    # Initialize model once
    model = load_model()
    
    
    with tab1:
        st.header("Upload an Image")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image of a handwritten digit", type=["jpg", "jpeg", "png"])
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if uploaded_file is not None:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=200)
                
                predict_uploaded = st.button("Predict Uploaded Image")
                
                if predict_uploaded:
                    # Preprocess the uploaded image
                    processed_img = preprocess_image(image)
                    
                    # Make prediction
                    with st.spinner('Predicting...'):
                        pred_probs = model.predict(processed_img)[0]
                        prediction = np.argmax(pred_probs)
                        confidence = float(pred_probs[prediction])
                    
                    # Display preprocessed image
                    st.subheader("Preprocessed Image")
                    preprocessed_display = processed_img.reshape(28, 28)
                    st.image(preprocessed_display, width=150)
        
        with col2:
            if uploaded_file is not None and predict_uploaded:
                # Display results
                st.subheader(f"Prediction: {prediction}")
                st.write(f"Confidence: {confidence:.4f}")
                
                # Show probability distribution
                st.subheader("Probability Distribution")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(range(10), pred_probs)
                ax.set_xticks(range(10))
                ax.set_xlabel('Digit')
                ax.set_ylabel('Probability')
                st.pyplot(fig)
                
                # Display top 3 predictions
                top3_indices = np.argsort(pred_probs)[-3:][::-1]
                
                st.subheader("Top 3 Predictions:")
                for i, idx in enumerate(top3_indices):
                    st.write(f"{i+1}. Digit {idx}: {pred_probs[idx]:.4f}")
    
    with tab2:
        st.header("About the Neural Network Model")
        
        st.write("""
        ## CNN Architecture
        This application uses a Convolutional Neural Network (CNN) trained on the MNIST dataset of handwritten digits.
        
        The model architecture consists of:
        - 3 convolutional blocks with increasing filter sizes (32, 64, 128)
        - Each block has batch normalization, max pooling, and dropout layers
        - Dense layers with regularization to prevent overfitting
        - Training with data augmentation to improve generalization
        
        ## Performance
        - Training accuracy: ~99.5%
        - Test accuracy: ~99.3% 
        - This model significantly outperforms traditional machine learning approaches
        
        ## Data Preprocessing
        When you draw or upload a digit, the application:
        1. Converts it to grayscale
        2. Resizes to 28x28 pixels (MNIST format)
        3. Normalizes pixel values to [0-1]
        4. Reshapes for CNN input (1, 28, 28, 1)
        """)
        
        # Show model architecture diagram
        st.subheader("Model Architecture Visualization")
        
        # Simple visual representation of the model architecture
        st.code("""
        Model: Sequential
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        conv2d (Conv2D)              (None, 28, 28, 32)        320       
        batch_normalization          (None, 28, 28, 32)        128       
        conv2d_1 (Conv2D)            (None, 28, 28, 32)        9,248     
        batch_normalization_1        (None, 28, 28, 32)        128       
        max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0         
        dropout (Dropout)            (None, 14, 14, 32)        0         
        conv2d_2 (Conv2D)            (None, 14, 14, 64)        18,496    
        batch_normalization_2        (None, 14, 14, 64)        256       
        conv2d_3 (Conv2D)            (None, 14, 14, 64)        36,928    
        batch_normalization_3        (None, 14, 14, 64)        256       
        max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0         
        dropout_1 (Dropout)          (None, 7, 7, 64)          0         
        conv2d_4 (Conv2D)            (None, 7, 7, 128)         73,856    
        batch_normalization_4        (None, 7, 7, 128)         512       
        conv2d_5 (Conv2D)            (None, 7, 7, 128)         147,584   
        batch_normalization_5        (None, 7, 7, 128)         512       
        max_pooling2d_2 (MaxPooling2 (None, 3, 3, 128)         0         
        dropout_2 (Dropout)          (None, 3, 3, 128)         0         
        flatten (Flatten)            (None, 1152)              0         
        dense (Dense)                (None, 256)               295,168   
        batch_normalization_6        (None, 256)               1,024     
        dropout_3 (Dropout)          (None, 256)               0         
        dense_1 (Dense)              (None, 10)                2,570     
        =================================================================
        Total params: 586,986
        Trainable params: 585,578
        Non-trainable params: 1,408
        _________________________________________________________________
        """)

if __name__ == "__main__":
    main()