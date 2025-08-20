import streamlit as st
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# -------------------------------
# Load and prepare the data
@st.cache_data
def load_model_and_data():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
    fake["class"] = 0
    true["class"] = 1

    data = pd.concat([fake, true], axis=0)
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.drop(['title', 'subject', 'date'], axis=1)

    data["text"] = data["text"].apply(clean_text)

    X = data["text"]
    y = data["class"]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec, y)

    return model, vectorizer, fake, true

# -------------------------------
# Load Model & Data
model, vectorizer, fake_df, true_df = load_model_and_data()

# -------------------------------
# Streamlit UI
st.set_page_config(page_title="📰 Fake News Detector")
st.title("🧠 Fake News Detector")
st.markdown("Enter any  **News Article** below to check whether it's **REAL** or **FAKE**.")
# -------------------------------
# Show Example Data for Users
st.subheader("📂 Example News Articles from Dataset")

example_news = None

with st.expander("📰 View Sample Fake News"):
    for i, news in enumerate(fake_df["text"].head(3)):  # show 3 for demo
        if st.button(f"👉 Use Fake News {i+1}"):
            example_news = news

with st.expander("✅ View Sample Real News"):
    for i, news in enumerate(true_df["text"].head(3)):
        if st.button(f"👉 Use Real News {i+1}"):
            example_news = news

# If user clicks a button, fill text area
if example_news:
    st.session_state["news_input"] = example_news

# User Input Section
user_input = st.text_area("📝 Paste News Text Here:", height=250, 
    key="news_input")

if st.button("🔍 Check"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        processed = clean_text(user_input)
        vector_input = vectorizer.transform([processed])
        prediction = model.predict(vector_input)[0]

        if prediction == 1:
            st.success("✅ This news is **REAL**.")
            st.balloons()
        else:  
            st.error("🚫 This news is **FAKE**.")
            st.snow()

