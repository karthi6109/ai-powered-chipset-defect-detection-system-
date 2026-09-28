# AI-Powered Chipset Defect Detection System

An AI-based computer vision application designed to identify defects in chipset images using a trained deep learning model. The project provides an end-to-end workflow covering dataset preparation, model training, model storage, and image-based defect prediction through a Python application.

## Overview

Defects in electronic chipsets can affect product quality and reliability. Manual inspection can be time-consuming and may lead to inconsistent results.

This project uses machine learning and image processing to automate the initial inspection process. A trained model analyzes chipset images and predicts the presence or category of a defect.

## Features

- AI-based chipset defect detection
- Image-based prediction
- Deep learning model training
- Dataset-based model development
- Python application for running predictions
- Saved trained model for inference
- Simple application structure for testing new images

## Project Structure

```text
ai-powered-chipset-defect-detection-system/
│
├── dataset/             # Dataset used for training and testing
├── scripts/             # Supporting scripts
├── static/              # Static application files
├── templates/           # Application templates
├── __pycache__/         # Python cache files
│
├── app.py               # Main application
├── model.py             # Model definition and prediction logic
├── model.pth            # Trained model
├── train.py             # Model training script
├── requirements.txt     # Required Python packages
└── README.md            # Project documentation
