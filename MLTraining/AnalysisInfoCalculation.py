import numpy as np
import pandas as pd
from pandas import DataFrame
import re
import nltk as nlp
import numpy as np
import pickle

data = pd.read_csv("Data.csv")
data = data.replace(np.nan, '', regex=True)
df = DataFrame(columns=["Words", "Vocabs"])
count_vectorizer = pickle.load(open("vector.pickle", "rb"))


def process_text_to_root(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = nlp.word_tokenize(text)
    lemma = nlp.WordNetLemmatizer()
    text = [lemma.lemmatize(word) for word in text]
    text = " ".join(text)
    return text


for text in data.iloc[:, 2].values:
    text = process_text_to_root(text)
    count_words = text.count(" ") + 1
    trans_text = count_vectorizer.transform([text]).toarray()[0]
    count_vocab = 0
    for entry in trans_text:
        if entry > 0:
            count_vocab = count_vocab + 1
    data = {
        "Words":count_words,
        "Vocabs":count_vocab
    }
    df = df.append(data, ignore_index=True)


export_csv = df.to_csv(r'CountData.csv', index=None, header=True)
