# SentimentSphere

*SentimentSphere* is a multimodal real-time emotion recognition system that integrates three powerful models to understand and classify human emotions based on *visual*, *textual*, and *speech* inputs.

---

## Overview

This project is developed as part of a DC credit initiative and includes:

### 1. *Visual Emotion Recognition (CNN)*
- Captures live webcam feed.
- Uses a Convolutional Neural Network to classify facial emotions in real time.
- Recognized emotions include: Happy, Sad, Angry, Neutral, Surprised, and more.

### 2. *Text-Based Emotion Recognition (NLP + LSTM)*
- Accepts a sentence as input.
- Predicts the tone and emotion conveyed using NLP techniques and an LSTM model.
- Example input: "I'm feeling great today!" → Emotion: Happy

### 3. *Speech-Based Emotion Recognition (LSTM)*
- Records live audio through microphone.
- Extracts features like MFCCs and feeds them to an LSTM model to classify the emotion in real time.
- Real-time feedback during conversation or speech.

---

## Tech Stack

- *Frontend/Hosting*: Streamlit
- *Backend Models*: Python (TensorFlow, Keras, OpenCV, NLTK, Librosa)
- *Other Tools*: NumPy, Pandas, Matplotlib

---

## How It Works

Each model runs independently but is hosted together on a unified *Streamlit* interface. Due to deployment constraints, the models are currently hosted locally.

---


## Try It Locally

Clone the repo:
```bash
git clone [GitHub Repo Link]
cd SentimentSphere
