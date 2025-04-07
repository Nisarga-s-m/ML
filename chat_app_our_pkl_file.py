import streamlit as st
import pickle

# Load models
@st.cache_resource
def load_models():
    with open("logistic_model.pkl", "rb") as f:
        logistic_model = pickle.load(f)
    with open("naivebayes_model.pkl", "rb") as f:
        nb_model = pickle.load(f)
    return logistic_model, nb_model

logistic_model, nb_model = load_models()

# Preprocessing function (same as your training code)
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english')) - {'not', 'no', 'never'}
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'<.*?>|[^a-zA-Z\s]', '', str(text).lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("🗣️ WhatsApp-Style Sentiment Analysis")

model_choice = st.selectbox("Choose a model", ["Logistic Regression", "Naive Bayes"])
user_input = st.chat_input("Type your message here...")

if user_input:
    st.markdown(f'<div class="user-msg">{user_input}</div><div class="clear"></div>', unsafe_allow_html=True)
    
    cleaned_input = preprocess_text(user_input)
    
    if model_choice == "Logistic Regression":
        prediction = logistic_model.predict([cleaned_input])[0]
    else:
        prediction = nb_model.predict([cleaned_input])[0]

    sentiment_emojis = {"positive": "🟢😊", "neutral": "🟡😐", "negative": "🔴😠"}
    sentiment_classes = {"positive": "positive", "neutral": "neutral", "negative": "negative"}

    emoji = sentiment_emojis.get(prediction, "⚪🤖")
    style_class = sentiment_classes.get(prediction, "neutral")

    st.markdown(f"""
        <div class="sentiment-box {style_class}">
            {emoji} {prediction.capitalize()}
        </div>
        """, unsafe_allow_html=True)

    # Bot reply
    bot_reply = "Thanks! Let me know if you need anything else."
    st.markdown(f'<div class="bot-msg">{bot_reply}</div><div class="clear"></div>', unsafe_allow_html=True)
