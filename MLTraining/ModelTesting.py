import pickle
import re
import nltk as nlp

def process_text_to_root(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = nlp.word_tokenize(text)
    lemma = nlp.WordNetLemmatizer()
    text = [lemma.lemmatize(word) for word in text]
    text = " ".join(text)
    return text

count_vectorizer = pickle.load(open("vector.pickle", "rb"))
model = pickle.load(open("LRModel.pickle", "rb"))


test = "Inspiration**************************************we wanted to find a solution to make it easier for students to be able to study for classes as well as finals without having to deal with the awkward dm's and continuous flakes.What it does************************************DropStudy website will take inputs of your classes as well as your active location. With this information and using radar.io API DropStudy will create a perimeter within your area showing you other students as well as allowing other students that are studying for the same classes as you, all in real-time.How I built it************************************Challenges I ran into**************************Accomplishments that I'm proud of**********What I learned********************************What's next for DropStudy*******************"
test_array = [process_text_to_root(test)]
test_vector = count_vectorizer.transform(test_array).toarray()

print("Normal: ")
print(model.predict(test_vector))
print(model.predict_proba(test_vector)[0][1])
