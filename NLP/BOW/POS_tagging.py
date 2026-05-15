##POS (Part-of-Speech) Tagging is a process in natural language processing (NLP) that involves assigning a part of speech to each word in a given text. This can include categories such as nouns, verbs, adjectives, adverbs, etc. POS tagging helps in understanding the grammatical structure of sentences and is often used in various NLP applications such as parsing, information extraction, and machine translation.
from nltk import pos_tag   
import nltk
from nltk.tokenize import word_tokenize  

nltk.download('punkt')    # Downloading the punkt tokenizer resource.
sentence = "The quick brown fox jumps over the lazy dog."    # Sample sentence for POS tagging.
tokens = word_tokenize(sentence)           
pos_tags = pos_tag(tokens)    
print("POS Tags: ", pos_tags, "\n")    
print("="*100)

#universal POS tags provided by NLTK
nltk.download('universal_tagset')    # Downloading the universal tagset resource for POS tagging which provides a simplified set of POS tags like noun, verb, adjective, etc.
universal_pos_tags = pos_tag(tokens, tagset='universal')
print("Universal POS Tags: ", universal_pos_tags, "\n")
print("="*100)

#count Nouns and verbs in the sentence based on universal POS tags          
noun_count = sum(1 for word, tag in universal_pos_tags if tag == 'NOUN')
adj_count = sum(1 for word, tag in universal_pos_tags if tag == 'ADJ')
print("Number of Nouns: ", noun_count)                  
print("Number of Adjectives: ", adj_count)
print("="*100)
Proper_nouns = [word for word, tag in pos_tags if tag == 'NNP']    # Extracting proper nouns from the original POS tags.
print("Proper Nouns: ", Proper_nouns)

#comparison of POS tags with universal POS tags
for (word, original_tag), (_, universal_tag) in zip(pos_tags, universal_pos_tags    ):
    print(f"Word: {word}, Original POS Tag: {original_tag}, Universal POS Tag: {universal_tag}")   

#function to get the frequency of each POS tag in the sentence
from collections import Counter    # Importing the Counter class from the collections module to count the frequency of POS tags.
def pos_tag_frequency(pos_tags):
    tag_counts = Counter(tag for word, tag in pos_tags)    # Counting the frequency of each POS tag in the list of POS tags.
    return tag_counts
frequency = pos_tag_frequency(universal_pos_tags)    # Getting the frequency of POS tags in the universal POS tags.
print("POS Tag Frequency: ", frequency)      

#visualization of POS tag distribution using a bar chart
import matplotlib.pyplot as plt     
tags, counts = zip(*frequency.items())    # Unzipping the tags and their corresponding counts from the frequency dictionary.
plt.bar(tags, counts)    # Creating a bar chart with tags on the x-axis and
# counts on the y-axis.
plt.xlabel('POS Tags')    
plt.ylabel('Frequency')    
plt.title('POS Tag Distribution')   
plt.xticks(rotation=45)   
plt.show()    # Displaying the plot.
    

    
