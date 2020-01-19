from flask import Flask, request, abort
import pickle
import re
import nltk as nlp

app = Flask(__name__)

count_vectorizer = pickle.load(open("MLTraining/vector.pickle", "rb"))
model = pickle.load(open("MLTraining/LRModel.pickle", "rb"))


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
    body = request.get_json()
    if not body:
        abort(400)
    text = body.get('text', "")
    if text == "":
        return "0"
    else:
        test_array = [process_text_to_root(text)]
        test_vector = count_vectorizer.transform(test_array).toarray()
        val = model.predict_proba(test_vector)[0][1]
        val = int(val * 100)
        if 50 < val < 65:
            val = val + 10
        if val > 90:
            val = val - 10
        if val < 10:
            val = val + 10
        return str(val)


if __name__ == '__main__':
    app.run()
