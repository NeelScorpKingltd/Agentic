import pandas as pd
#NLTK is a library for natural language processing in Python. It provides tools for tokenization, stemming, lemmatization, and more.
import nltk 
nltk.download(['punkt','punkt_tab', 'averaged_perceptron_tagger_eng', 'maxent_ne_chunker_tab', 'stopwords'])    # Downloading the necessary resources for tokenization, part-of-speech tagging, and named entity recognition.
from nltk.tokenize import word_tokenize, sent_tokenize    # Importing functions for tokenizing text into words and sentences.

text=input("Enter the text: ")    # Taking input text from the user.""

# Tokenizing the input text into sentences and words.
sentences=sent_tokenize(text)    # Tokenizing the text into sentences.
words=word_tokenize(text)    # Tokenizing the text into words.  

print("Sentences: ", sentences, "\n") 
print("="*100)    
print("Words: ", words, "\n")    # Printing the list of words.
print("="*100) 

#exploring different tokenization techniques provided by NLTK
from nltk.tokenize import TreebankWordTokenizer, WordPunctTokenizer, TweetTokenizer   # Importing different tokenizers from NLTK.

treebank_tokenizer = TreebankWordTokenizer()    
treebank_tokens = treebank_tokenizer.tokenize(text)   
print("Treebank Tokens: ", treebank_tokens, "\n")    # Printing the tokens obtained from TreebankWordTokenizer.
print("="*100)


word_punct_tokenizer = WordPunctTokenizer()    
word_punct_tokens = word_punct_tokenizer.tokenize(text)   
print("WordPunct Tokens: ", word_punct_tokens, "\n")    # splits text into tokens by separating alphabetic characters, numeric characters, and non-alphabetic/non-numeric characters (punctuation) into distinct units
print("="*100)          


tweet_tokenizer = TweetTokenizer()       
tweet_tokens = tweet_tokenizer.tokenize(text)    
print("Tweet Tokens: ", tweet_tokens, "\n")    # Printing the tokens obtained from TweetTokenizer.
print("="*100)

#custom regex tokenizer to extract numbers and hashtags from the text
from nltk.tokenize import RegexpTokenizer    # Importing the RegexpTokenizer from NLTK.
regex_tokenizer = RegexpTokenizer(r'\d+|#\w+')    # Creating a regular expression tokenizer that extracts numbers and hashtags.
regex_tokens = regex_tokenizer.tokenize(text)                   
print("Regex Tokens: ", regex_tokens, "\n")    # Printing the tokens obtained from the custom regex tokenizer.
print("="*100)



