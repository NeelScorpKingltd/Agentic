import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, Flatten, Input
from tensorflow.keras.preprocessing.text import one_hot  # type: ignore
from tensorflow.keras.utils import pad_sequences  # type: ignore
import numpy as np

# Global constants needed for both training and prediction
vocab_size = 8000
max_length = 2312
model_file = 'fiction_classifier.keras'

if os.path.exists(model_file):
    print("Loading saved model from disk...")
    model = load_model(model_file)
else:
    print("No saved model found. Training new model...")
    import pandas as pd
    from tqdm.auto import tqdm
    import nltk
    from nltk.corpus import brown
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    # download the brown corpus
    nltk.download('brown')

    # get the sentences and categories
    sentences = brown.sents()
    categories = brown.categories()

    browntable = pd.DataFrame()
    category_list = []
    filename_list = []
    text_list = []

    for i in tqdm(brown.categories()):
        for j in brown.fileids(categories=i):
            text = ""
            for sent in brown.sents(j):
                text += " ".join(sent)
            category_list.append(i)
            filename_list.append(j)
            text_list.append(text)

    browntable["category"] = category_list
    browntable["filename"] = filename_list
    browntable["text"] = text_list

    # categorise into two main categories: Fiction and non-Fiction  
    for i in ['news','editorial','reviews','government','learned','hobbies','religion','lore']:
        browntable = browntable.replace(to_replace=i, value='nonfiction')

    for i in ['fiction','mystery','science_fiction','adventure','romance']:
        browntable = browntable.replace(to_replace=i, value='fiction')

    index_names = browntable[(browntable['category'] != 'fiction') & (browntable['category'] != 'nonfiction')].index
    browntable.drop(index_names, inplace=True)

    browntable = browntable.replace(to_replace='nonfiction', value='0')
    browntable = browntable.replace(to_replace='fiction', value='1')

    text_list = browntable["text"].to_list()
    category_list = browntable["category"].astype(int).to_list()

    encoded_texts = [one_hot(text, vocab_size) for text in text_list]
    
    padded_texts = pad_sequences(encoded_texts, maxlen=max_length, padding='post')

    model = Sequential()
    model.add(Input(shape=(max_length ,)))
    model.add(Embedding(vocab_size, 8))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # generate training and test datasets
    X_train , X_test , y_train , y_test = train_test_split(padded_texts , category_list , test_size=0.2 , random_state=23 ) 
    
    model.fit(X_train , np.array(y_train) , epochs=5, verbose=1 )   

    # calculate loss and accuracy
    loss, accuracy = model.evaluate(X_test , np.array(y_test))
    print(f"Loss: {loss:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

    # predict on test data
    y_pred = model.predict(X_test)
    y_pred_labels = np.where(y_pred > 0.5, 1, 0)

    # save the model to disk
    model.save(model_file)
    print("Model saved to disk as", model_file)


def predict_category(text):
    encoded = [one_hot(text, vocab_size)]
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    pred = model.predict(padded, verbose=0)[0][0]
    return "Fiction" if pred > 0.5 else "Non-fiction"
