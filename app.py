from flask import Flask, request, abort
import pickle
import re
import nltk as nlp
import ast
import pandas as pd
from scipy import stats

app = Flask(__name__)

count_vectorizer = pickle.load(open("MLTraining/vector.pickle", "rb"))
model = pickle.load(open("MLTraining/LRModel.pickle", "rb"))
analysis_data = pd.read_csv("MLTraining/CountData.csv")

def process_text_to_root(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = nlp.word_tokenize(text)
    lemma = nlp.WordNetLemmatizer()
    text = [lemma.lemmatize(word) for word in text]
    text = " ".join(text)
    return text


@app.route('/predict', methods=["POST"])
def predict():
    body = request.get_data()
    if not body:
        abort(400)
    body = ast.literal_eval(body.decode('UTF-8'))
    text = body.get('text', "")
    if text == "":
        return "0"
    else:
        text = process_text_to_root(text)
        word_count = text.count(" ") + 1

        test_vector = count_vectorizer.transform([text]).toarray()
        val = model.predict_proba(test_vector)[0][1]

        count_vocab = 0
        for entry in test_vector[0]:
            if entry > 0:
                count_vocab = count_vocab + 1

        word_percentile = stats.percentileofscore(analysis_data.iloc[:, 0],
                                                  word_count)
        vocab_percentile = stats.percentileofscore(analysis_data.iloc[:, 1],
                                                  count_vocab)

        # Return Final Weighted Score ---
        # 35% descriptiveness, 35% language diversity, 30% project scoring
        # Make some projections to make 75 percentile optimal point

        if word_percentile > 0.75:
            word_percentile = 0.75 - (word_percentile - 0.75)
        if vocab_percentile > 0.75:
            vocab_percentile = 0.75 - (vocab_percentile - 0.75)

        final_score = (35 * word_percentile/100) + \
                      (40 * vocab_percentile/100) + (25 * val)
        if final_score < 75:
            final_score = int(final_score + 10)

        print("descriptive: ", word_percentile)
        print("diversity: ", vocab_percentile)
        print("project rating: ", val)
        print("final verdict: ", final_score)

        return str(final_score)


if __name__ == '__main__':
    app.run()
