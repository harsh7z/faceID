# Face ID System using Siamese Network (PyTorch + OpenCV)

This is a Python-based Face ID system that can **register multiple users**, and later **verify a person** using facial similarity. It uses a **Siamese Neural Network** implemented in PyTorch and OpenCV for real-time webcam face capture.

---

## Features

- Register new users using webcam face capture
- Train a Siamese neural network to learn face similarity
- Verify users live via webcam
- Supports multiple users


## What is a Siamese Network?

A **Siamese Network** is a neural architecture that learns to determine **similarity** between two inputs rather than classifying them directly. It consists of **twin networks** that share weights and extract embeddings from input images.

In this project, the network compares two face images:
- If the faces belong to the **same person**, the distance between embeddings is **small**.
- If they belong to **different people**, the distance is **large**.

It is trained using **contrastive loss**, which teaches the network to minimize the distance between similar faces and push dissimilar ones apart.

## Installation

Install Python dependencies:

```bash
pip install opencv-python torch torchvision numpy pillow
```

## Usage Guide

### 1. Register a User
Run the registration script and follow the prompt:

```bash
python register.py
```
Enter a username (e.g. alice). 
Press and hold the s key to capture and save face images
300 images will be saved under data/user/alice

### 2. Train the Model
Train the Siamese network on registered face pairs:

```bash
python train.py
```
Trains a model to learn facial similarity. Model is saved to models/siamese_model.pt

### 3. Verify a Face
Launch the verification script to recognize a face:

```bash
python verify.py
```

Webcam opens, Press q to capture a face. The system compares it with stored users. Outputs the closest match or shows "Unknown"

## Technical Details

Face images are converted to grayscale and resized to 100x100
The model uses contrastive loss to learn embedding space
During verification, it uses pairwise distance between embeddings
Adjustable verification threshold (default 0.8) in verify.py: THRESHOLD = 0.8

## Requirements

Python 3.7 or later
PyTorch
torchvision
OpenCV (opencv-python)
Pillow
NumPy

$ python register.py
Enter username to register: bob
Capturing 10 face images for user: bob
Saved image 1/10
...

$ python train.py
Epoch 1/10, Loss: 0.3912
...
Model saved to models/siamese_model.pt

$ python verify.py
Best match: bob (score: 0.5473)
Verified as: bob

## To-Do (Optional Ideas)

Use MTCNN or Dlib for better face detection
Cache face embeddings for faster verification
Add feature for deleting/updating users
GUI version (Tkinter or PyQt)
