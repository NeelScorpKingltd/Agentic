from pathlib import Path
import matplotlib.pyplot as plt
from numpy import rint
import seaborn as sns

import pandas as pd

sns.set_theme(style="ticks", palette="pastel")
file_path="Amazon_movies/Data/Amazon_Prime_Movies.xlsx"
data_df=pd.read_excel(file_path)


print(f"DataFrame shape: {data_df.shape[0]} rows and {data_df.shape[1]} columns")

#Data Cleaning  - Running Time as running time is in string format
print(data_df['running_time'].head())
data_df['running_time_clean'] = data_df['running_time'].str.extract(r'(\d+)').astype(float)
print("\nRunning Time after cleaning:")
print(data_df['running_time_clean'].head())

#Check for any missing imdb raitings
data_df['imdb_rating_clean'] = pd.to_numeric(data_df['imdb_rating'], errors='coerce')
missing_ratings = data_df['imdb_rating_clean'].isnull().sum()
print(f"Missing or invalid ratings: {missing_ratings}")

#had to drop the rows(1276) with missing ratings as they are not useful for analysis
data_df = data_df.dropna(subset=['imdb_rating_clean'])

missing_ratings = data_df['imdb_rating_clean'].isnull().sum()
print(f"Missing or invalid ratings: {missing_ratings}")

#Univariate Analysis
fig, axes=plt.subplots(1, 2, figsize=(14, 5))

# Distribution of IMDb Ratings
sns.histplot(data_df['imdb_rating_clean'], bins=20, kde=True, ax=axes[0])
axes[0].set_title('Distribution of IMDb Ratings')
axes[0].set_xlabel('IMDb Rating')
axes[0].set_ylabel('Count')

# Distribution of Running Times
sns.histplot(data_df['running_time_clean'], bins=20, kde=True, ax=axes[1])
axes[1].set_title('Distribution of Running Times')
axes[1].set_xlabel('Running Time (minutes)')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()  

#Quick Statistical Check for extreme values in running time
print("\nStatistical Summary of Running Times:")
print(data_df['running_time_clean'].describe())

valid_runtime_mask = (data_df['running_time_clean'] >= 40) & (data_df['running_time_clean'] <= 400)
data_df = data_df[valid_runtime_mask]

print(f"New dataset shape after outlier removal: {data_df.shape[0]} rows")
print("\n--- Cleaned Running Time Stats ---")
print(data_df['running_time_clean'].describe())

#Anayzing the Plot Summary Column
data_df['plot_summary'] = data_df['plot_summary'].astype(str)

#using a lambda functionto split the string by whitespaces and count the chunks
data_df['summary_word_count'] = data_df['plot_summary'].apply(lambda x: len(x.split()))

#Visualize the text length distribution
plt.figure(figsize=(10, 5))
sns.histplot(data_df['summary_word_count'], bins=40, kde=True, color='purple')
plt.title('Distribution of Plot Summary Word Counts')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.axvline(x=400, color='red', linestyle='--', label='BERT Safe Limit (~400 words)')
plt.legend()
plt.show()

#Check hard statistics
print("\n--- Plot Summary Word Count Stats ---")
print(data_df['summary_word_count'].describe())

#Starting Bi-variate analysis for Co-orelation
correlation=data_df['imdb_rating_clean'].corr(data_df['running_time_clean'])
print(f"\nCorrelation between IMDb Ratings and Running Time: {correlation:.4f}")

# Plot a scatter plot with an automatically calculated trendline
plt.figure(figsize=(10, 6))
sns.regplot(
    data=data_df, 
    x='running_time_clean', 
    y='imdb_rating_clean', 
    scatter_kws={'alpha': 0.3, 'color': 'teal'}, 
    line_kws={'color': 'red', 'linewidth': 2}
)
plt.title(f'Bivariate Analysis: Running Time vs. IMDb Rating\nCorrelation: {correlation:.4f}')
plt.xlabel('Running Time (Minutes)')
plt.ylabel('IMDb Rating')
plt.show()

#Analyzing relation  b/w  running time over time
yearly_runtime=data_df.groupby('year')['running_time_clean'].median().reset_index()
sns.lineplot(data=yearly_runtime, x='year', y='running_time_clean', color='purple', linewidth=2.5)
plt.title('Insight A: Evolution of Median Movie Runtime Over the Years')
plt.xlabel('Release Year')
plt.ylabel('Median Running Time (Minutes)')
plt.show()

#Analyzing relation  b/w  imdb rating and Director
clean_directors = data_df[data_df['Directed by'].notnull() & (data_df['Directed by'] != 'None')]

#Directors with at least 5 movies in the dataset
director_counts = clean_directors['Directed by'].value_counts()
prolific_directors = director_counts[director_counts >= 5].index 

director_stats = clean_directors[clean_directors['Directed by'].isin(prolific_directors)]
director_ranking = director_stats.groupby('Directed by')['imdb_rating_clean'].mean().sort_values(ascending=False).head(10)

print("--- Top 10 Most Prolific Directors by Average Rating ---")
print(director_ranking)