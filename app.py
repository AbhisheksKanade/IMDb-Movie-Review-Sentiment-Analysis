from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import streamlit as st
import pickle
import pandas as pd

# ---------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------

st.set_page_config(
    page_title="IMDb Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.stButton>button{
    background-color:#E50914;
    color:white;
    border-radius:10px;
    height:50px;
    width:250px;
    font-size:18px;
}

h1{
    color:#E50914;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load IMDb dataset
df = pd.read_csv("IMDB Dataset.csv")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("📌 Project Information")

st.sidebar.info(
"""
### IMDb Movie Review Sentiment Analysis

Model Used
- Logistic Regression

Dataset
- IMDb 50,000 Reviews

Accuracy
- 89%

Tech Stack
- Python
- Pandas
- NLTK
- Scikit-Learn
- Streamlit
"""
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🎬 IMDb Movie Review Sentiment Analysis")

st.markdown("---")

# ---------------------------------------------------
# POSITIVE WORD CLOUD
# ---------------------------------------------------

st.subheader("☁️ Positive Reviews Word Cloud")

positive_reviews = df[df["sentiment"] == "positive"]

positive_text = " ".join(positive_reviews["review"].astype(str))

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(positive_text)

fig, ax = plt.subplots(figsize=(10,5))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")

st.pyplot(fig)

# ---------------------------------------------------
# NEGATIVE WORD CLOUD
# ---------------------------------------------------

st.subheader("☁️ Negative Reviews Word Cloud")

negative_reviews = df[df["sentiment"] == "negative"]

negative_text = " ".join(negative_reviews["review"].astype(str))

negative_wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(negative_text)

fig, ax = plt.subplots(figsize=(10,5))
ax.imshow(negative_wordcloud, interpolation="bilinear")
ax.axis("off")

st.pyplot(fig)

# ---------------------------------------------------
# DATASET DISTRIBUTION
# ---------------------------------------------------

st.subheader("📊 Dataset Distribution")

labels = ["Positive", "Negative"]
sizes = df["sentiment"].value_counts()

fig, ax = plt.subplots(figsize=(5,5))

ax.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90
)

ax.axis("equal")

st.pyplot(fig)

# ---------------------------------------------------
# TOP 20 WORDS
# ---------------------------------------------------

st.subheader("📈 Top 20 Frequent Words")

all_words = positive_text.lower().split()

common_words = Counter(all_words).most_common(20)

words_df = pd.DataFrame(common_words, columns=["Word", "Count"])

st.bar_chart(words_df.set_index("Word"))

# ---------------------------------------------------
# PREDICTION HISTORY
# ---------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------

st.subheader("📝 Enter Movie Review")

review = st.text_area(
    "Type your review below:",
    height=150
)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if st.button("Analyze Sentiment"):

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)

    probabilities = model.predict_proba(review_vector)

    confidence = probabilities.max() * 100

    if prediction[0] == "positive":
        st.success("😊 Positive Review")
        st.balloons()
    else:
        st.error("😞 Negative Review")

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    st.progress(float(probabilities.max()))

    probability_df = pd.DataFrame({
        "Sentiment": ["Negative", "Positive"],
        "Probability (%)": [
            round(probabilities[0][0]*100,2),
            round(probabilities[0][1]*100,2)
        ]
    })

    st.subheader("📊 Prediction Probabilities")

    st.dataframe(probability_df)

    st.bar_chart(
        probability_df.set_index("Sentiment")
    )

    # Save history

    st.session_state.history.append({
        "Review": review,
        "Prediction": prediction[0],
        "Confidence": round(confidence,2)
    })

# ---------------------------------------------------
# HISTORY
# ---------------------------------------------------

st.subheader("📜 Prediction History")

history_df = pd.DataFrame(st.session_state.history)

if len(history_df) > 0:

    st.dataframe(history_df)

    csv = history_df.to_csv(index=False)

    st.download_button(
        label="⬇ Download Prediction History",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown("""
### 🚀 Built Using

- Python
- Pandas
- NLTK
- TF-IDF Vectorization
- Logistic Regression
- Streamlit

**Accuracy: 89%**
""")