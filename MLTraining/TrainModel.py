import pandas as pd
import re
import nltk as nlp
import numpy as np
import pickle

# First lemmatize the words
data = pd.read_csv("Data.csv")
data = data.replace(np.nan, '', regex=True)


def process_text_to_root(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = nlp.word_tokenize(text)
    lemma = nlp.WordNetLemmatizer()
    text = [lemma.lemmatize(word) for word in text]
    text = " ".join(text)
    return text

text_list = []
for text in data["Text"]:
    text_list.append(process_text_to_root(text))

#count_vectorizer = pickle.load(open("vector.pickle", "rb"))

from sklearn.feature_extraction.text import CountVectorizer
max_features = 1200  # We use the most common word
count_vectorizer = CountVectorizer(max_features=max_features,
                                   stop_words="english")
sparce_matrix = count_vectorizer.fit_transform(text_list).toarray()
pickle.dump(count_vectorizer, open("vector.pickle", "wb"))

y = data.iloc[:, 0].values  # male or female classes
x = sparce_matrix
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,
                                                    random_state=5)
#
# from imblearn.over_sampling import RandomOverSampler as OverSampler
#
# over_sampler = OverSampler(random_state=20)
#
# x_sample, y_sample = over_sampler.fit_sample(x_train, y_train)
#
#
# from imblearn.under_sampling import RandomUnderSampler as UnderSampler
#
# under_sampler = UnderSampler(random_state=23)
#
# x_sample_u, y_sample_u = under_sampler.fit_sample(x_train, y_train)


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=200)
lr.fit(x_train, y_train)
print("our accuracy is: {}".format(lr.score(x_test, y_test)))
print("our accuracy is: {}".format(lr.score(x_train, y_train)))
pickle.dump(lr, open("LRModel.pickle", "wb"))
#
# ls = LogisticRegression(max_iter=225)
# ls.fit(x_sample, y_sample)
# print("our accuracy is: {}".format(ls.score(x_test, y_test)))
# print("our accuracy is: {}".format(lr.score(x_sample, y_sample)))
# #pickle.dump(ls, open("LROverSample.pickle", "wb"))
#
# lsu = LogisticRegression(max_iter=225)
# lsu.fit(x_sample_u, y_sample_u)
# print("our accuracy is: {}".format(ls.score(x_test, y_test)))
# print("our accuracy is: {}".format(lr.score(x_sample_u, y_sample_u)))
# #pickle.dump(lsu, open("LRUnderSample.pickle", "wb"))

test = "We think improving cybersecurity does not always entail passively anticipating possible attacks. It is an equally valid strategy to go on the offensive against the transgressors. Hence, we employed the strategy of the aggressors against themselves --- by making what's basically a phishing bank app that allows us to gather information about potentially stolen phones.Our main app, Bait Master, is a cloud application linked to Firebase. Once the user finishes the initial setup, the app will disguise itself as a banking application with fairly convincing UI/UX with fake bank account information. Should the phone be ever stolen or password-cracked, the aggressor will likely be tempted to take a look at the obvious bank information. When they open the app, they fall for the phishing bait. The app will discreetly take several pictures of the aggressor's face from the front camera, as well as uploading location/time information periodically in the background to Firebase. The user can then check these information by logging in to our companion app --- Trap Master Tracker --- using any other mobile device with the credentials they used to set up the main phishing app, where we use Google Cloud services such as Map API to display the said information.Both the main app and the companion app are developed in Java Android using Android Studio. We used Google's Firebase as a cloud platform to store user information such as credentials, pictures taken, and location data. Our companion app is also developed in Android and uses Firebase, and it uses Google Cloud APIs such as Map API to display information.1) The camera2 library of Android is very difficult to use. Taking a picture is one thing --- but taking a photo secretly without using the native camera intent and to save it took us a long time to figure out. Even now, the front camera configuration sometimes fails in older phones --- we are still trying to figure that out. 2) The original idea was to use Twilio to send SMS messages to the back-up phone number of the owner of the stolen phone. However, we could not find an easy way to implement Twilio in Android Studio without hosting another server, which we think will hinder maintainability. We eventually decided to opt out of this idea as we ran out of time.I think we really pushed the boundary of our Android dev abilities by using features of Android that we did not even know existed. For instance, the main Bait Master app is capable of morphing its own launcher to acquire a new icon as well as a new app name to disguise itself as a banking app. Furthermore, discreetly taking pictures without any form of notification and uploading them is technically challenging, but we pulled it off nonetheless. We are really proud of the product that we built at the end of this weekend.Appearances can be misleading. Don't trust everything that you see. Be careful when apps ask for access permission that it shouldn't use (such as camera and location).We want to add more system-level mobile device management feature such as remote password reset, wiping sensitive data, etc. We also want to make the app more accessible by adding more disguise appearance options, as well as improving our client support by making the app more easy to understand."
test_array = [process_text_to_root(test)]
test_vector = count_vectorizer.transform(test_array).toarray()

print(lr.predict(test_vector))
print(lr.predict_proba(test_vector))
