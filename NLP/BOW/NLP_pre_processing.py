import pandas as pd
#NLTK is a library for natural language processing in Python. It provides tools for tokenization, stemming, lemmatization, and more.
import nltk 
nltk.download('stopwords')    # Downloading the stopwords resource for text preprocessing.
from nltk.corpus import stopwords   # Importing the stopwords corpus and word_tokenize function from NLTK
from nltk.tokenize import word_tokenize

##TEXT NORMALIZATION (pre-processing before tokenization)##

Text="This is a sample text for Text Normalization. It includes various forms of words, such as running, ran, and runs! R.U.N .....  "    # Sample text for normalization.

# Converting the text to lowercase.
normalized_text = Text.lower()  
print("Lowercase Text: ", normalized_text, "\n")    # Printing the lowercase version of the text.
print("="*100)

#removing punctuation from the text using regex
import re    # Importing the regular expression module. 
punctuation_removed_text = re.sub(r'[^\w\s]', '', normalized_text)    # Removing punctuation from the text using a regular expression.
print("Punctuation Removed Text: ", punctuation_removed_text, "\n")    # Printing the text with punctuation removed.
print("="*100)

#removing stop words from the text
from nltk.corpus import stopwords    # Importing the stopwords from NLTK.
stop_words = set(stopwords.words('english'))    # Getting the set of English stop words.
words = word_tokenize(punctuation_removed_text)    # Tokenizing the text into words.
filtered_text = [word for word in words if word not in stop_words]    # Removing stop words from the list of words.
print("Filtered Text: ", filtered_text, "\n")    # Printing the text with stop words removed.
print("="*100)

#stemming the words in the text 
#Stemming is the process of reducing words to their base or root form.extracts the root of a word.
from nltk.stem import PorterStemmer , WordNetLemmatizer   
stemmer = PorterStemmer()    # Creating an instance of the PorterStemmer.
stemmed_words = [stemmer.stem(word) for word in filtered_text]    # Stemming each word in the filtered text.
print("Stemmed Text: ", stemmed_words, "\n")    # Printing the text with stemmed words.
print("="*100)


nltk.download('wordnet')
nltk.download('omw-1.4')
stemmer1=WordNetLemmatizer()    # Creating an instance of the WordNetLemmatizer.
wordnet_lemmatized_words = [stemmer1.lemmatize(word) for word in filtered_text]    # Lemmatizing each word in the filtered text.
print("WordNet Lemmatized Text: ", wordnet_lemmatized_words, "\n")    # Printing the text with lemmatized words.
print("="*100)  