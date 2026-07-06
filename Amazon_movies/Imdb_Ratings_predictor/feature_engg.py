import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

#Ingest the cleaned dataset
file_path = "Amazon_movies/EDA/amazon_movies_cleaned.csv"

print("--- Loading Clean Data Artifact ---")
df = pd.read_csv('Amazon_movies/Data/amazon_movies_cleaned.csv')

df['plot_summary'] = df['plot_summary'].fillna('')

#Train-test Split
X = df[['plot_summary', 'Directed by', 'running_time_clean', 'year']]
y = df['imdb_rating_clean']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Categorical Encoding for 'Directed by' column
print("--- Encoding Categorical Features (Directed by) ---")
global_mean=y_train.mean()

director_stats=y_train.groupby(X_train['Directed by']).agg(['mean','count'])

#applying m-weighted average to handle directors with few movies
smoothed_weights=5
smoothed_means=(
    (director_stats['count'] * director_stats['mean'] + smoothed_weights * global_mean) /
    (director_stats['count'] + smoothed_weights)
).to_dict()

#Mapping the smoothed means to the training and test sets, filling missing directors with the global mean
X_train_director = X_train['Directed by'].map(smoothed_means).fillna(global_mean).values.reshape(-1, 1)
X_test_director = X_test['Directed by'].map(smoothed_means).fillna(global_mean).values.reshape(-1, 1)

#TF-IDF Vectorization for 'plot_summary'
print("--- Vectorizing Text Features (Plot Summary) ---")
tfidf=TfidfVectorizer(max_features=2000, stop_words='english', min_df=2)

#fit only on training data to avoid data leakage and transform both training and test data
X_train_tfidf=tfidf.fit_transform(X_train['plot_summary']).toarray()
X_test_tfidf=tfidf.transform(X_test['plot_summary']).toarray()

#Standardizing the 'running_time_clean' feature
print("--- Standardizing Numerical Features (Running Time) ---")
scaler=StandardScaler()
num_cols=['running_time_clean','year']
X_train_num=scaler.fit_transform(X_train[num_cols])
X_test_num=scaler.transform(X_test[num_cols])

# Stack the numerical columns, the encoded director column, and the 2,000 text columns horizontally
X_train_final = np.hstack((X_train_num, X_train_director, X_train_tfidf))
X_test_final = np.hstack((X_test_num, X_test_director, X_test_tfidf))

print("\n--- Final ANN Processing Array Verification ---")
print(f"Final X_train shape for ANN input layer: {X_train_final.shape}")
print(f"Final X_test shape for ANN evaluation:  {X_test_final.shape}")


#-------------------------------------------------------------------------------------------------------------------------------
print("\n--- Constructing the Neural Network ---")
input_dim = X_train_final.shape[1]
print(f"Input dimension for the ANN: {input_dim}")

# Define the ANN architecture
model = Sequential([
    Dense(256, activation='relu', input_dim=input_dim),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])

model.compile(
    optimizer='adam', 
    loss='mean_squared_error',      # The mathematical penalty for wrong guesses
    metrics=['mean_absolute_error'] # Human-readable error (e.g., "off by 0.6 rating points")
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print("\n--- Model architecture ---")
model.summary()

#train the model
print("\n--- Training the Neural Network ---")     
history=model.fit(X_train_final, y_train,
          validation_data=(X_test_final, y_test),
          epochs=30,
          batch_size=64,
          callbacks=[early_stop],
          verbose=1)

print("\n--- Final Model Evaluation ---")
test_loss, test_mae = model.evaluate(X_test_final, y_test, verbose=0)
print(f"Test Mean Absolute Error (MAE): {test_mae:.3f} rating points")

print("\n--- Exporting Production Artifacts ---")
# 1. Save the Keras Neural Network
model.save('Amazon_movies/Data/movie_ann_model.keras')
print("Saved: Neural Network (movie_ann_model.keras)")

# 2. Save the Text Vectorizer
joblib.dump(tfidf, 'Amazon_movies/Data/tfidf_vectorizer.pkl')
print("Saved: TF-IDF Vectorizer (tfidf_vectorizer.pkl)")

# 3. Save the Numerical Scaler
joblib.dump(scaler, 'Amazon_movies/Data/numerical_scaler.pkl')
print("Saved: Standard Scaler (numerical_scaler.pkl)")

# 4. Save the Target Encoding Dictionaries
joblib.dump({'smoothed_means': smoothed_means, 'global_mean': global_mean}, 'Amazon_movies/Data/director_encoding.pkl')
print("Saved: Director Encodings (director_encoding.pkl)")
print("Global Mean IMDb Rating:")


print("All artifacts exported successfully. Training environment can safely be spun down.")