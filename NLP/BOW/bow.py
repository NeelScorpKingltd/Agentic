import pandas as pd
import math
from collections import Counter

def tokenizer(text):
    """A simple tokenizer that splits text into words based on whitespace.converts it into lowercase and removes punctuation. Remove stop word from it and return the list of tokens."""
    # List of common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'i' ,'is' ,'you','are','was','were','be','been','being','have','has','had','do','does','did','will','would','shall','should','can','could','may','might','must'  }
    
    text = text.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
    tokens = text.split()
    return [token for token in tokens if token not in stop_words]


#simple corpus BOW function that creates a BOW representation of the given corpus.
def bow(corpus):    
    """Creates a Bag of Words (BOW) representation of the given corpus."""
    # Step 1: Tokenize the corpus and build the vocabulary
    tokenized_corpus = [tokenizer(doc) for doc in corpus]
    vocabulary = sorted({word for doc in tokenized_corpus for word in doc})
    
    # Step 2: Create a BOW representation
    bow_representation = [] 
    for doc in tokenized_corpus:
        word_count = Counter(doc)
        bow_vector = [word_count.get(word, 0) for word in vocabulary]
        bow_representation.append(bow_vector)
    
    return pd.DataFrame(bow_representation, columns=vocabulary)

# Example usage
corpus = ["YouTube Music is a music streaming service developed by the American video platform YouTube, a subsidiary of Google. The service is designed with an interface that allows users to simultaneously explore music audios and music videos from YouTube-based genres, playlists and recommendations. On December 1, 2020, YouTube Music replaced Google Play Music as Google's primary brand for music streaming. In April 2023, the service expanded its offerings to include support for podcasts[2] shortly before Google Podcasts was shut down.", 
          "YouTube Music also features a premium tier that provides several benefits to paying subscribers. These include ad-free playback, the ability to play audio in the background, and the option to download songs for offline listening. These benefits are also bundled with and available to subscribers of YouTube Premium."]
bow_df = bow(corpus)
print(bow_df)   
