import streamlit as st
import pickle

# Load both models (Logistic and Naive Bayes)
@st.cache_resource
def load_models():
    with open("logistic_model.pkl", "rb") as f:
        logistic_model = pickle.load(f)
    with open("naivebayes.pkl", "rb") as f:
        nb_model = pickle.load(f)
    return logistic_model, nb_model

logistic_model, nb_model = load_models()

# CSS for chat UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');
    body { font-family: 'Roboto', sans-serif; }
    .chat-container { max-width: 600px; margin: auto; }
    .user-msg { background-color: #DCF8C6; text-align: right; padding: 12px; border-radius: 12px; margin: 5px 0; float: right; clear: both; max-width: 70%; }
    .bot-msg { background-color: #E5E5EA; text-align: left; padding: 12px; border-radius: 12px; margin: 5px 0; float: left; clear: both; max-width: 70%; }
    .sentiment-box { display: block; padding: 6px; border-radius: 6px; font-weight: bold; margin: 10px auto; max-width: 200px; text-align: center; }
    .positive { background-color: #D4EDDA; color: #155724; }
    .neutral { background-color: #FFF3CD; color: #856404; }
    .negative { background-color: #F8D7DA; color: #721C24; }
    .clear { clear: both; }
    </style>
""", unsafe_allow_html=True)

st.title("💬 WhatsApp-Style Sentiment Chatbot")

# Session to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display previous messages
for msg in st.session_state.messages:
    css_class = "user-msg" if msg["role"] == "user" else "bot-msg"
    st.markdown(f'<div class="{css_class}">{msg["content"]}</div><div class="clear"></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input from user
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f'<div class="user-msg">{user_input}</div><div class="clear"></div>', unsafe_allow_html=True)

    # --- Preprocessing (must match training pipeline) ---
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    def preprocess(text):
        stop_words = set(stopwords.words('english')) - {'not', 'no', 'never'}
        lemmatizer = WordNetLemmatizer()
        text = re.sub(r'<.*?>|[^a-zA-Z\s]', '', str(text).lower())
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        return ' '.join(tokens)

    cleaned_input = preprocess(user_input)

    # Vectorizer (refit this the same way you did while training)
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    # Load same vectorizer settings
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
    df = pd.read_csv("your_dataset.csv")  # The dataset used to train original models
    df['cleaned_text'] = df['reviewText'].apply(preprocess)
    vectorizer.fit(df['cleaned_text'])  # Refitting vectorizer here
    input_vector = vectorizer.transform([cleaned_input])

    # Predict from both models
    lr_pred = logistic_model.predict(input_vector)[0]
    nb_pred = nb_model.predict(input_vector)[0]

    # Use majority vote or just show both predictions
    sentiment_map = {"positive": "🟢😊", "neutral": "🟡😐", "negative": "🔴😠"}

    st.markdown(f'<div class="sentiment-box {lr_pred}">{sentiment_map.get(lr_pred, "⚪🤖")} Logistic: {lr_pred.title()}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sentiment-box {nb_pred}">{sentiment_map.get(nb_pred, "⚪🤖")} Naive Bayes: {nb_pred.title()}</div>', unsafe_allow_html=True)

    bot_reply = "Thanks for sharing! Let me know if you want to analyze more text."
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.markdown(f'<div class="bot-msg">{bot_reply}</div><div class="clear"></div>', unsafe_allow_html=True)
