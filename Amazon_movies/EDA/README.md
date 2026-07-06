# Exploratory Data Analysis (EDA) Report: Amazon Movies Dataset

This report outlines the ingestion, cleaning, and exploratory data analysis pipeline executed on the Amazon Movies dataset. The insights gathered here directly dictate the downstream architecture for our two core machine learning targets:
1. **Prediction Model:** Predicting IMDb ratings using a hybrid TF-IDF + ANN architecture.
2. **Search & Recommendation Engine:** Semantic search driven by Sentence-BERT (SBERT).

---

## 1. Data Ingestion & Schema Verification
* **File Format:** `.xlsx` (Excel) spreadsheet.
* **Initial Shape:** 7,258 rows across 17 distinct structural features.
* **Pipeline Fix:** Fixed a structural constraint where an intermediate `.head()` assignment truncated data to 5 rows. The full workspace matrix was successfully restored to its true length.

---

## 2. Data Cleaning & Outlier Sanitation

* <img width="1046" height="374" alt="image" src="https://github.com/user-attachments/assets/aa6e2edc-2946-420d-bfea-f93ceb951722" />

### Numerical Sanitization
* **Target Variable (`imdb_rating`):** Successfully cast to a clean numeric float type. 1,276 corrupted records missing explicit target labels were dropped to preserve model integrity (a dataset cannot train on unlabelled target outcomes).
* **Feature Column (`running_time`):** Scraped string suffixes (e.g., " minutes") were stripped using Regular Expressions (`r'(\d+)'`) and cast to floats.

### Extreme Outlier Treatment
* **The Anomaly:** The raw data contained severe systemic noise, showing a minimum runtime of 1 minute (trailers/shorts) and a maximum runtime of 5,517 minutes (~92 hours, likely a box-set entry error).
* **The Fix:** Applied a domain-specific filter keeping only valid feature films between **40 and 300 minutes**. This preserved the mathematical consistency of the vector space.
* **Final Post-Cleaning Matrix Shape:** 5,959 clean rows.

### Feature Pruning
* **`maturity_rating`:** This column was audited and found to be entirely vacant of data. It was permanently dropped from the dataframe to optimize processing matrix dimensions.

---

## 3. Core Architectural Insights & AI Strategies

### A. The SBERT Buffer Limit (Text Data Constraint)
<img width="735" height="379" alt="image" src="https://github.com/user-attachments/assets/be625d16-a2a5-4287-b31f-f408f9eeb7eb" />

* **Discovery:** A distribution audit of the `plot_summary` word counts revealed a **median length of 623 words**, with over 75% of the dataset exceeding 508 words.
* **The Architecture Impact:** Standard SBERT architectures feature a hard input cap of 512 tokens (~400 words) and fail silently by truncating text. 
* **The Strategy:** To ensure the semantic search engine doesn't lose crucial plot points at the end of summaries, we will implement either **Head+Tail Truncation** or **Windowed Chunking with Mean Pooling** in the feature engineering pipeline.

### B. Linear Feature Correlation
* **Discovery:** A bivariate analysis between `running_time_clean` and `imdb_rating_clean` revealed a Pearson correlation coefficient of **$r = 0.2908$**. 
* **The Strategy:** While the positive linear slope is modest, it represents a valid mathematical signal. This runtime feature will be explicitly concatenated with text vectors to train the Artificial Neural Network (ANN).

<img width="739" height="485" alt="image" src="https://github.com/user-attachments/assets/b17a3c9f-8b73-4ba8-849a-15a91266a5d8" />



### C. Temporal Volatility (Time Evolution)
<img width="466" height="355" alt="image" src="https://github.com/user-attachments/assets/d84bd748-37cb-4f99-98fc-14bec2644505" />

* **Discovery:** An analysis of median running time by release year showed high volatility in early cinema (1920s–1940s) before stabilizing into a consistent 100-110 minute corridor from the 1980s onward, with a slight upward trend in the 2020s.
* **The Strategy:** Because the relationship between year and runtime is non-linear, the `year` feature may require "binning" (e.g., grouping into decades) during feature engineering to help the ANN interpret the eras effectively.

### D. The Heavyweight Signal (Prolific Directors)
* **Discovery:** Certain highly prolific directors act as massive statistical outliers for positive ratings (e.g., Billy Wilder and Denis Villeneuve both averaging above 8.0, far above the global median).
* **The Strategy:** The text column `Directed by` will undergo **Target Encoding**. Director names will be replaced by their historical average rating, translating a categorical text string into a highly correlated numerical feature for the ANN.
--- Top 10 Most Prolific Directors by Average Rating ---
Directed by
Billy Wilder         8.077778
Denis Villeneuve     8.033333
Elia Kazan           7.860000
Quentin Tarantino    7.860000
Pete Docter          7.820000
Rajkumar Hirani      7.800000
Frank Capra          7.766667
James Cameron        7.680000
David Fincher        7.680000
S. S. Rajamouli      7.666667
---

## 4. Execution Workflow
To reproduce this analysis locally, run the standalone tracking script:
```bash
python eda.py
