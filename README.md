# Classifying Embeddings with Keras and Gemini API

This repository demonstrates how to build a text classification model using Google Gemini's embeddings and Keras. The system classifies newsgroup posts into categories by leveraging pre-trained embeddings, achieving high accuracy with relatively small training datasets.

## Overview

This project implements a classification system that:
- Generates text embeddings using Google Gemini API
- Processes the 20 Newsgroups dataset
- Builds and trains a neural network classifier using Keras
- Achieves high accuracy in categorizing scientific texts

## Features

- **Text Embedding:** Generate embeddings using Gemini API
- **Data Preprocessing:** Clean and prepare newsgroup data
- **Neural Network:** Build and train a classification model
- **Custom Predictions:** Make predictions on new text inputs

## Setup & Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/md82680/Classifying-Embeddings-Keras-Gemini.git
   cd Classifying-Embeddings-Keras-Gemini
   ```

2. **Install Dependencies:**
   ```bash
   pip install -U -q "google-genai==1.7.0"
   pip install pandas numpy keras scikit-learn tqdm
   ```

3. **API Key Setup:**
   - Store your Google API key in Kaggle secrets (GOOGLE_API_KEY)
   - Or set as environment variable

## Dataset

The project uses the 20 Newsgroups Text Dataset:
- 18,000 newsgroup posts
- 20 different topics
- Split into training and test sets
- Focuses on science categories ('sci.*')

## Model Architecture

- Input layer: Raw embedding data
- Hidden layer: ReLU activation
- Output layer: Softmax for class probabilities
- Optimization: Adam optimizer
- Loss: Sparse Categorical Crossentropy

## Usage

The notebook demonstrates:
- Dataset preprocessing and sampling
- Embedding generation with Gemini API
- Model building and training
- Custom prediction functionality

## Performance

The model achieves:
- Training accuracy: ~98%
- Validation accuracy: ~91%
- Early stopping implementation
- Efficient convergence within 12 epochs

## Task Types Supported

The text-embedding-004 model supports:
- RETRIEVAL_QUERY
- RETRIEVAL_DOCUMENT
- SEMANTIC_SIMILARITY
- CLASSIFICATION
- CLUSTERING
- FACT_VERIFICATION

## License

This project is licensed under the Apache License 2.0.

## Acknowledgements

Thanks to:
- Google GenAI for the embedding API
- Keras team for the deep learning framework
- Scikit-learn for the 20 Newsgroups dataset
