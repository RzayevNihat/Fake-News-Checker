import streamlit as st
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

model_path = 'fakenewschecker.keras'
model = load_model(model_path)

voc_size = 10000
sent_length = 40
tokenizer = Tokenizer(num_words=voc_size)
lemmatizer = WordNetLemmatizer()

st.title('Fake News Classifier')
st.write('Enter a news headline below to check if it is fake or real.')

headline = st.text_input('News Headline')

if headline:

    review = re.sub('[^a-zA-Z]', ' ', headline)
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)

    sequence = tokenizer.texts_to_sequences([review])
    X_new = pad_sequences(sequence, maxlen=sent_length)  # Use the same `maxlen` as in training

    prediction = model.predict(X_new)
    predicted_label = np.round(prediction, 0).astype("int32")

    if predicted_label[0][0] == 1:
        st.write('Prediction: Fake News')
    else:
        st.write('Prediction: Real News')
