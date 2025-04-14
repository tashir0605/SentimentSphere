import streamlit as st
import os
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Page configuration
st.set_page_config(
    page_title="Text Emotion Detector",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: white;
    }
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > textarea {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: 1px solid #333333;
    }
    .stButton button {
        background-color: #04AA6D;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: #028a57;
    }
    .css-1d391kg {
        color: white !important;
    }
    .stMarkdown {
        color: white !important;
    }
    .st-bc {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "text_emotion.pkl")
pipe_lr = joblib.load(open(model_path, "rb"))

# Emoji dictionary
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}

# Emotion prediction
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

# Probability prediction
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# App layout
def main():
    st.title("🧠 Text Emotion Detector")
    st.subheader("🎯 Understand emotions hidden in your text")

    with st.form(key='emotionForm'):
        raw_text = st.text_area("Type something emotional...", placeholder="E.g. I'm so excited for my new journey!")
        submit_button = st.form_submit_button(label="Analyze")

    if submit_button:
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ✍️ Original Text")
            st.info(raw_text)

            st.markdown("#### 🎭 Predicted Emotion")
            emoji_icon = emotions_emoji_dict.get(prediction, "")
            st.success(f"**{prediction.upper()}** {emoji_icon}")

            st.metric(label="Confidence", value=f"{np.max(probability)*100:.2f}%")

        # The `with col2:` block should be inside `main()`
        with col2:
            st.markdown("#### 📊 Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            chart = alt.Chart(proba_df_clean).mark_bar().encode(
                x=alt.X('emotions', sort='-y', axis=alt.Axis(labelColor='white', titleColor='white')),
                y=alt.Y('probability', axis=alt.Axis(labelColor='white', titleColor='white')),
                color=alt.Color('emotions', legend=None)
            ).properties(
                width=350,
                height=300,
                background='#1e1e1e'  # Lighter dark background for the chart
            ).configure_view(
                strokeWidth=0
            ).configure_axis(
                grid=False
            ).configure_title(
                color='white'
            )

            st.altair_chart(chart, use_container_width=True)

if __name__ == '__main__':
    main()

