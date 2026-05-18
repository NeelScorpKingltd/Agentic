import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64


def ensure_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words')
    try:
        nltk.data.find('chunkers/maxent_ne_chunker')
    except LookupError:
        nltk.download('maxent_ne_chunker')


ensure_nltk_resources()


STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def text_preprocess(text: str) -> str:
    text = text.strip()
    text = re.sub(r'[^A-Za-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word.lower() not in STOPWORDS]
    return tokens


def lemmatize_tokens(tokens):
    return [LEMMATIZER.lemmatize(t.lower()) for t in tokens]


def pos_tag(tokens):
    return nltk.pos_tag(tokens, tagset='universal')


def named_entities(tokens):
    pos = nltk.pos_tag(tokens)
    tree = nltk.ne_chunk(pos, binary=False)
    entities = []
    seen = set()
    for subtree in tree:
        if isinstance(subtree, nltk.Tree):
            entity_text = ' '.join(token for token, _ in subtree.leaves())
            entity_label = subtree.label()
            pair = (entity_text, entity_label)
            if pair not in seen:
                entities.append(pair)
                seen.add(pair)
    return entities


def top_n_words(tokens, n):
    filtered = [t.lower() for t in tokens if t.isalpha() and t.lower() not in STOPWORDS]
    ctr = Counter(filtered)
    return ctr.most_common(n)


def top_nouns(pos_tags, n):
    nouns = [word for word, pos in pos_tags if pos.startswith('NOUN')]
    return top_n_words(nouns, n)

def get_locations(named_entities_list):
    return [entity for entity in named_entities_list if entity[1] == 'GPE']

# function that uses POS tagging and frequency distribution to get the most frequent verbs
def top_verbs(pos_tags, n=5):
    verbs = [word.lower() for word, pos in pos_tags if pos.startswith('VERB')]
    if not verbs:
        return []
    fd = FreqDist(verbs)
    return fd.most_common(n)


def plot_top_verbs(top_verbs_list):
    if not top_verbs_list:
        return None
    labels = [w for w, _ in top_verbs_list]
    counts = [c for _, c in top_verbs_list]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, counts, color='#2563eb')
    ax.set_title('Top verbs')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Verb')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('ascii')
    return f"data:image/png;base64,{img_b64}"


def analyze_article(text: str) -> dict:
    # We must use raw tokens (which keep punctuation like commas) for NER to work correctly!
    raw_tokens = nltk.word_tokenize(text)
    tokens = text_preprocess(text)

    pos_tags = pos_tag(tokens)
    named_entities_list = named_entities(raw_tokens)
    top_words = top_n_words(tokens, n=5)
    top_nouns_list = top_nouns(pos_tags, n=5)
    locations = get_locations(named_entities_list)
    top_verbs_list = top_verbs(pos_tags, n=5)
    verbs_plot = plot_top_verbs(top_verbs_list)

    return {
        
        'tokens': tokens,
        'pos_tags': pos_tags,
        'top_nouns': top_nouns_list,
        'named_entities': named_entities_list,
        'top_words': top_words,
        'locations': locations,
        'top_verbs': top_verbs_list,
        'verbs_plot': verbs_plot,
    }
