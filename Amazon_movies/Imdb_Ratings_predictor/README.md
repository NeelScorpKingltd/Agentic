# IMDb Rating Predictor: Neural Network Pipeline

This repository contains an end-to-end machine learning pipeline designed to process raw movie data (text, categories, and numerical metrics) into a highly accurate continuous prediction engine. 

The architecture utilizes a TensorFlow Artificial Neural Network (ANN) that ingests TF-IDF text vectors, encoded categorical variables, and scaled numerical data to predict IMDb ratings with a high degree of accuracy.

## Critical Engineering Decisions
1. Feature Exclusion: The Title Column
 
 The Problem: Movie titles possess extremely high cardinality (almost entirely unique).

 The Solution: The title column was intentionally dropped from the mathematical input matrix. If Target Encoded, it acts as a unique identifier, causing the network to memorize the dataset rather than learning generalizable patterns  (Data Leakage). If vectorized via TF-IDF, it injects highly sparse, mathematically useless noise into the input layer. It is retained strictly for logging during inference.

2. Regression Over Classification
   
 The Problem: IMDb ratings span from 1.0 to 10.0. A common approach is to round these to whole numbers and treat it as a 10-class classification problem.

 The Solution: This model utilizes Regression with a Linear output node and Mean Squared Error (MSE) loss. Classification models do not understand that a rating of "7" is closer to an "8" than a "2". Regression calculates the exact mathematical distance of the error, forcing the network to adjust its weights dynamically and preserving the nuance of the continuous ratings.

3. Target Encoding with Smoothing
  
 The Problem: The Directed by column contains highly valuable predictive signals, but One-Hot Encoding hundreds of directors would trigger the Curse of Dimensionality, inflating the feature matrix with sparse data and crashing the  network's memory.

 The Solution: We applied Target Encoding, replacing the director's string name with their historical average rating. To prevent overfitting on obscure directors with only one highly-rated film, we applied an $m$-estimate smoothing weight to pull low-sample outliers closer to the global population mean.

4. Text Processing: TF-IDF vs. Transformers
   
 The Problem: The plot_summary text averages 623 words, exceeding standard Transformer (BERT) token limits (512 tokens/~400 words), which causes silent data truncation and loss of critical plot resolutions.

 The Solution: We bypassed heavy contextual transformers in favor of a TfidfVectorizer capped at 2,000 features. This converts the text into a dense mathematical representation of word importance.

5. The "Funnel" ANN Architecture & Regularization
   
 The Problem: Feeding 2,000+ features into a deep neural network often leads to severe overfitting, where the network memorizes specific rare words in the training data.

 The Solution: The TensorFlow architecture utilizes a strict "funnel" design (e.g., Input(2003) -> 256 ->128 -> 64  -> Output(1)) to force an information bottleneck. Combined with  Dropout layers (0.2) and an Early Stopping callback monitoring the validation loss, the network is forced to learn broad textual themes and generalize effectively to unseen data.

  
## 📁 Project Structure

```text
├── ../Data/                 # Data directory located one folder up
│   ├── amazon_movies_cleaned.csv   # Intermediate clean data artifact
│   └── artifacts/                  # Serialized production objects
│       ├── movie_ann_model.keras
│       ├── tfidf_vectorizer.pkl
│       ├── numerical_scaler.pkl
│       └── director_encoding.pkl
├──Imdb_Ratings_predictor
        ├── eda.py                   # Data ingestion, sanitization, and anomaly filtering
        ├── train.py                 # Feature engineering and TensorFlow ANN training
        └── inference.py             # Standalone production script for live predictions
