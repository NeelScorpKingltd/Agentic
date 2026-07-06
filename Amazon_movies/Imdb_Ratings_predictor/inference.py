import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

print("--- Initializing Production Inference Engine ---")

#Load Artifacts into memory
model = load_model('Amazon_movies/Data/movie_ann_model.keras')
tfidf = joblib.load('Amazon_movies/Data/tfidf_vectorizer.pkl')
scaler = joblib.load('Amazon_movies/Data/numerical_scaler.pkl')

director_data = joblib.load('Amazon_movies/Data/director_encoding.pkl')
smoothed_means = director_data['smoothed_means']
global_mean = director_data['global_mean']

print("Engine Online. Ready for inputs.\n")

def predict_movie_rating(title, plot_summary, director, runtime, year):
    """
    Ingests raw movie data, applies strictly loaded production transformations, 
    and outputs a neural network prediction.
    """
    print(f"Processing: {title} | Dir: {director}")
    
    # 1. Encode Director
    encoded_dir = smoothed_means.get(director, global_mean)
    X_new_director = np.array([[encoded_dir]]) 
    
    # 2. Scale Numbers
    raw_nums = pd.DataFrame({'running_time_clean': [runtime], 'year': [year]})
    X_new_num = scaler.transform(raw_nums)
    
    # 3. Vectorize Text
    X_new_tfidf = tfidf.transform([plot_summary]).toarray()
    
    # 4. Stack Matrix
    X_new_final = np.hstack((X_new_num, X_new_director, X_new_tfidf))
    
    # Verify exact input dimensions (Must be 2003)
    if X_new_final.shape[1] != model.input_shape[1]:
        raise ValueError(f"Matrix Dimension Mismatch. Expected {model.input_shape[1]}, got {X_new_final.shape[1]}")
    
    # 5. Predict
    predicted_rating = model.predict(X_new_final, verbose=0)[0][0]
    
    print(f"Predicted IMDb Rating: {predicted_rating:.1f} / 10.0\n")
    return predicted_rating


if __name__ == "__main__":
    
    sample_plot = """
    In the 1820s Scary American wilderness, frontiersman Hugh Glass (Leonardo DiCaprio) is brutally mauled by a bear while guiding a fur-trapping expedition. When a ruthless mercenary named Fitzgerald (Tom Hardy) kills Glass's son and leaves him to die, the grief-stricken Glass embarks on a grueling, vengeance-fueled trek to survive and exact justice.
    """

    predict_movie_rating(
        title="The Revenant", 
        plot_summary=sample_plot, 
        director="Alejandro G. Iñárritu", 
        runtime=156, 
        year=2015
    )