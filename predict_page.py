import streamlit as st
import pickle

import json
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



class Sentiment:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"

class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else:
            return Sentiment.POSITIVE

class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews

    def get_text(self):
        return [x.text for x in self.reviews]


    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]

    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))

        positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))

        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)






file_name = "./data/Books_small_10000.json"


reviews_so_far = []
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        list.append(reviews_so_far, Review(review["reviewText"], review["overall"]))


# Prep data
training, test = train_test_split(reviews_so_far, test_size = 0.33, random_state= 42)

train_container = ReviewContainer(training)

test_container = ReviewContainer(test)


train_container.evenly_distribute()

train_x = train_container.get_text()
train_y = train_container.get_sentiment()

test_container.evenly_distribute()

test_x = test_container.get_text()
test_y = test_container.get_sentiment()

vectorizer = TfidfVectorizer()

train_x_vectors = vectorizer.fit_transform(train_x)


def load_model():
    with open('sentiment_classifier.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()


def show_predict_page():
    st.title("Text Sentiment Prediction")

    st.write("""### We need the text to predict its sentiment""")

    text_predict = st.text_input("Input Text Here")

    ok = st.button("Predict Sentiment")

    if ok:
        vect = vectorizer.transform([text_predict])
        result = data.predict(vect)
        st.subheader(f"The predicted sentiment of the text is {result[0]}")









