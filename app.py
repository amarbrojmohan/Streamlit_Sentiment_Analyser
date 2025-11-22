import streamlit as st
from transformers import pipeline

# -------------------------
# Load the sentiment model
# -------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Email Sentiment Analyzer", layout="centered")

st.title("📧 Email Sentiment Analyzer")
st.write("Paste an email below and click **Analyze** to detect sentiment.")

email_text = st.text_area(
    "Email Content:",
    placeholder="Paste your email text here...",
    height=200
)

if st.button("Analyze Sentiment"):
    if email_text.strip() == "":
        st.warning("Please paste an email before analyzing.")
    else:
        with st.spinner("Analyzing sentiment..."):
            result = classifier(email_text)[0]

        label = result["label"]
        score = round(float(result["score"]), 4)

        # Color-coded output
        if label == "POSITIVE":
            st.success(f"😊 POSITIVE — Confidence: {score}")
        elif label == "NEGATIVE":
            st.error(f"😞 NEGATIVE — Confidence: {score}")
        else:
            st.info(f"😐 NEUTRAL — Confidence: {score}")

st.markdown("---")
st.caption("Powered by DistilBERT • Streamlit • Hugging Face Transformers")
