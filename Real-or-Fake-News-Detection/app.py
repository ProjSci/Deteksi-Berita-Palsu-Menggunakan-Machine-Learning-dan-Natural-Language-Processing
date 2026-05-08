import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background-color: #0b1120;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: white;
    border-right: 1px solid #e5e7eb;
}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] * {
    color: black !important;
}

/* SIDEBAR TITLE */
.sidebar-title {
    color: black;
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    margin-top: 10px;
}

/* SIDEBAR SUBTITLE */
.sidebar-subtitle {
    color: #444444;
    text-align: center;
    font-size: 14px;
    margin-bottom: 20px;
}

/* FOOTER */
.footer-text {
    color: black;
    font-size: 14px;
}

/* MAIN TITLE */
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    margin-bottom: 15px;
    color: white;
}

/* RESULT BOX */
.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}

/* FAKE */
.fake {
    background-color: #ffdddd;
    color: #b30000;
}

/* REAL */
.real {
    background-color: #ddffdd;
    color: #006600;
}

/* REMOVE EXTRA TOP SPACE */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* SMALLER IMAGE */
.small-image img {
    border-radius: 15px;
    max-height: 100px;
    max-width: 85%;
    margin: auto;
    display: block;
    object-fit: cover;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    '<div class="sidebar-title">FakeNews AI</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="sidebar-subtitle">Fake News Detection System</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "<hr style='border:1px solid #d1d5db;'>",
    unsafe_allow_html=True
)

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Step 1: EDA",
        "🧹 Step 2: Preprocessing",
        "🤖 Step 3: Model Training",
        "📰 Step 4: Detection Demo",
        "📈 Step 5: Results"
    ]
)

st.sidebar.markdown(
    "<hr style='border:1px solid #d1d5db;'>",
    unsafe_allow_html=True
)

st.sidebar.markdown("""
<div class="footer-text">
<b>Kelompok 9 - LC01</b><br><br>

1. Samuel Christoff<br>
2. Jovin Prasetia Willim
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    "<hr style='border:1px solid #d1d5db;'>",
    unsafe_allow_html=True
)

st.sidebar.markdown("""
<div class="footer-text" style="text-align:center;">
Machine Learning Project<br>
Binus University
</div>
""", unsafe_allow_html=True)

# =========================================================
# TEXT CLEANING FUNCTION
# =========================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# =========================================================
# HOME
# =========================================================

if menu == "🏠 Home":

    st.markdown(
        '<div class="main-title">📰 Fake News Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div style="text-align: center" class="small-image">', unsafe_allow_html=True)

    left, center, right = st.columns([1,2,1])

    with center:
        st.image(
            "https://images.unsplash.com/photo-1504711434969-e33886168f5c",
            width=800
        )

    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div style='
        background-color:#111827;
        padding:20px;
        border-radius:15px;
        color:white;
    '>

    <h2>📌 About This Project</h2>

    This project uses <b>Machine Learning</b> and 
    <b>Natural Language Processing (NLP)</b>
    to classify news articles as:

    <ul>
    <li>✅ Real News</li>
    <li>🚨 Fake News</li>
    </ul>

    <h3>Technologies Used</h3>

    <ul>
    <li>TF-IDF</li>
    <li>Naive Bayes</li>
    <li>LSTM</li>
    <li>Streamlit</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# EDA
# =========================================================

elif menu == "📊 Step 1: EDA":

    st.title("📊 Exploratory Data Analysis")

    st.write("""
    Exploratory Data Analysis (EDA) is used to understand the dataset
    before training Machine Learning models.
    """)

    labels = ["Fake News", "Real News"]
    values = [23481, 21417]

    fig, ax = plt.subplots()

    ax.bar(labels, values)

    ax.set_ylabel("Number of Articles")
    ax.set_title("Dataset Distribution")

    st.pyplot(fig, width=800)

# =========================================================
# PREPROCESSING
# =========================================================

elif menu == "🧹 Step 2: Preprocessing":

    st.title("🧹 Data Preprocessing")

    st.write("""
    The preprocessing steps include:
    - Lowercasing
    - Removing punctuation
    - Removing URLs
    - Removing special characters
    - Text cleaning
    """)

    sample_text = """
    Breaking News!!! Visit https://example.com NOW!!!
    """

    st.subheader("Before Cleaning")
    st.code(sample_text)

    st.subheader("After Cleaning")
    st.code(clean_text(sample_text))

# =========================================================
# MODEL TRAINING
# =========================================================

elif menu == "🤖 Step 3: Model Training":

    st.title("🤖 Model Training")

    st.write("""
    Two approaches were compared:

    ### 1. Naive Bayes + TF-IDF
    - Fast
    - Efficient
    - Good baseline model

    ### 2. LSTM + Word Embedding
    - Understands context
    - Better semantic understanding
    - Requires more computation
    """)

    comparison = pd.DataFrame({
        "Model": ["Naive Bayes", "LSTM"],
        "Representation": ["TF-IDF", "Word Embedding"],
        "Strength": [
            "Fast and lightweight",
            "Captures context"
        ]
    })

    st.dataframe(comparison)

# =========================================================
# DETECTION DEMO
# =========================================================

elif menu == "📰 Step 4: Detection Demo":

    st.title("📰 Fake News Detection Demo")

    st.write("""
    Paste a news article below to analyze whether it is REAL or FAKE.
    """)

    news_input = st.text_area(
        "Enter News Text",
        height=250,
        placeholder="Paste article here..."
    )

    if st.button("Analyze News"):

        if news_input.strip() == "":
            st.warning("Please enter some text.")

        else:

            cleaned = clean_text(news_input)

            vectorized = vectorizer.transform([cleaned])

            prediction = model.predict(vectorized)[0]

            probability = model.predict_proba(vectorized)[0]

            confidence = max(probability) * 100

            st.write("")
            st.subheader("Prediction Result")

            if prediction == 0:

                st.markdown(
                    f"""
                    <div class="result-box fake">
                    🚨 FAKE NEWS<br>
                    Confidence: {confidence:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="result-box real">
                    ✅ REAL NEWS<br>
                    Confidence: {confidence:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")

            st.subheader("Confidence Scores")

            fake_score = probability[0] * 100
            real_score = probability[1] * 100

            st.progress(int(fake_score))
            st.write(f"Fake News Probability: {fake_score:.2f}%")

            st.progress(int(real_score))
            st.write(f"Real News Probability: {real_score:.2f}%")

# =========================================================
# RESULTS
# =========================================================

elif menu == "📈 Step 5: Results":

    st.title("📈 Results & Discussion")

    st.write("""
    ### Experimental Results

    - Naive Bayes achieved strong performance using TF-IDF.
    - LSTM demonstrated better contextual understanding.
    - English news performed better than Indonesian news.
    """)

    models = ["Naive Bayes", "LSTM"]
    accuracy = [0.94, 0.91]

    fig, ax = plt.subplots()

    ax.bar(models, accuracy)

    ax.set_ylim([0, 1])

    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")

    st.pyplot(fig, width=800)

    st.write("""
    ### Conclusion

    Machine Learning and NLP are effective approaches
    for detecting fake news automatically.
    """)