import os
os.environ.setdefault("TRANSFORMERS_NO_RICH", "1")
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

BERT_MODEL_NAME = 'bert-base-uncased'


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BERT_MODEL_NAME)
    return tokenizer, model

def get_sent_Token(tokenizer, sentence):
    inputs = tokenizer(sentence, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    return inputs

if __name__ == "__main__":
    tokenizer, model = load_model()
    from sklearn.metrics.pairwise import cosine_similarity

    Sentence=input("Enter the sentence: ")
    sentence_token = get_sent_Token(tokenizer, Sentence)
    

"""   
Showing the tokenized sentence

inputs = sentence_token['input_ids'][0]
    for token_id in inputs:
        if token_id != 0:
            print(tokenizer.decode(token_id))"""

#gettting word embedding from bert
    
outputs = model(**sentence_token)
print("Output keys:", outputs.keys())
last_hidden_state = outputs.last_hidden_state  #last hindden state countains the contextualized word embeddings for each token in the input sentence.
print("Last hidden state shape:", last_hidden_state.shape)  #shape is (batch_size, sequence_length, hidden_size)

#contextual similarity of words in list of different sentences
sentences = ["I love to watch TV",
        "I am wearing a wrist watch",
        "My brother goes to the ground every Sunday to watch Football",
        "My wife gifted me a beautiful watch on my birthday",
        "My wife gifted me an awesome watch"]

# get the embeddings of watch from each sentence
watch_embeddings = []
for sentence in sentences:
    sentence_token = get_sent_Token(tokenizer, sentence)
    outputs = model(**sentence_token)
    last_hidden_state = outputs.last_hidden_state

    watch_index = (sentence_token['input_ids'][0] == tokenizer.convert_tokens_to_ids('watch')).nonzero(as_tuple=True)[0]
    if len(watch_index) > 0:
        watch_embedding = last_hidden_state[0, watch_index[0], :].detach().numpy()
        watch_embeddings.append(watch_embedding)

if len(watch_embeddings) >= 2:
    watch_embeddings = np.vstack(watch_embeddings)

    # Calculate cosine similarity between the watch embeddings
    similarity_matrix = cosine_similarity(watch_embeddings)
    print("Cosine similarity between 'watch' embeddings:")
    print(similarity_matrix)

    print("Cosine similarity between the first and second 'watch' embeddings:")
    print(cosine_similarity(watch_embeddings[0:1], watch_embeddings[1:2]))
else:
    print("Not enough 'watch' embeddings found to compute similarity.")


