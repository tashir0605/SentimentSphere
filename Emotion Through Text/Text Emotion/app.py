import streamlit as st
import os
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Set page config
st.set_page_config(
    page_title="Text Emotion Detector",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f4f4f4;
    }
    textarea {
        font-size: 16px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "text_emotion.pkl")
pipe_lr = joblib.load(open(model_path, "rb"))

# Emoji mapping
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}

# Prediction functions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Main App
def main():
    st.title("🧠 Text Emotion Detector")
    st.subheader("🎯 Understand emotions hidden in your text!")

    # Sidebar
    with st.sidebar:
        st.image("https://em-content.zobj.net/source/microsoft-teams/363/thought-balloon_1f4ad.png", width=100)
        st.header("About")
        st.write("This app detects emotions from any given text using a trained ML model.")
        st.write("Made with ❤️ using Streamlit")

    # Input form
    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here", placeholder="e.g. I'm feeling amazing today!")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict.get(prediction, "")
            st.markdown(f"### 🎭 **Emotion:** `{prediction}` {emoji_icon}")
            st.metric(label="Confidence", value=f"{np.max(probability)*100:.2f}%")

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x=alt.X('emotions', sort='-y'),
                y='probability',
                color=alt.Color('emotions', legend=None),
                tooltip=['emotions', 'probability']
            ).properties(
                width=300,
                height=300
            )

            st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()

