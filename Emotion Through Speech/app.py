import streamlit as st
import numpy as np
import librosa
import os
import pyaudio
import wave
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation, Dropout, Input
from keras.layers import Conv1D, MaxPooling1D, Flatten
import tensorflow as tf
# Add these at the top of your file
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow warnings

# Add this before running your app
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Set page configuration
st.set_page_config(
    page_title="Speech Emotion Analyzer",
    page_icon="🎤",
    layout="wide"
)

# Function to create and load model
@st.cache_resource
def load_model():
    # Create a new model with the same architecture as in the saved weights
    model = Sequential()
    
    # Change this from 256 to 128 to match the saved weights
    model.add(Conv1D(128, 5, padding='same', input_shape=(216, 1)))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=8))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    # Compile model - use standard optimizer (not legacy)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00001),
        metrics=['accuracy']
    )
    
    # Load pre-trained weights
    if os.path.exists("saved_models/Emotion_Voice_Detection_Model.h5"):
        try:
            model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
            print("Model weights loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
            print(f"Error loading model weights: {e}")
    else:
        st.error("Model weights file not found. Please check the file path.")
        print("Model weights file not found!")
    
    return model

# Function to extract features from audio
def extract_feature(audio_path, duration=2.5, offset=0.5):
    try:
        # Load audio file
        X, sample_rate = librosa.load(
            audio_path, 
            res_type='kaiser_fast',
            duration=duration,
            sr=22050*2,
            offset=offset
        )
        
        # Make sure X has consistent length by padding or truncating
        target_length = int(duration * 22050 * 2)
        if len(X) < target_length:
            # Pad with zeros if too short
            X = np.pad(X, (0, target_length - len(X)))
        elif len(X) > target_length:
            # Truncate if too long
            X = X[:target_length]
        
        # Extract MFCC features
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
        
        # Ensure consistent feature length - this is critical!
        if len(mfccs) < 216:
            mfccs = np.pad(mfccs, (0, 216 - len(mfccs)))
        elif len(mfccs) > 216:
            mfccs = mfccs[:216]
            
        return mfccs
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# Function to predict emotion
def predict_emotion(audio_path, model):
    # Extract features
    features = extract_feature(audio_path)
    
    if features is None:
        return None
    
    # Verify feature length
    if len(features) != 216:
        st.warning(f"Feature length mismatch. Expected 216, got {len(features)}. Fixing...")
        if len(features) < 216:
            features = np.pad(features, (0, 216 - len(features)))
        else:
            features = features[:216]
    
    # Convert features to DataFrame
    features_df = pd.DataFrame([features])
    
    # Reshape for CNN model
    features_expanded = np.expand_dims(features_df.values, axis=2)
    
    # Make prediction
    prediction = model.predict(features_expanded, verbose=0)
    
    return prediction

# Function to record audio
def record_audio(duration=4, filename="temp_recording.wav"):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    
    p = pyaudio.PyAudio()
    
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    st.info("Recording...")
    
    frames = []
    
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    st.success("Recording complete!")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return filename

# Function to display waveform
def display_waveform(audio_path):
    y, sr = librosa.load(audio_path)
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title('Audio Waveform')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    st.pyplot(fig)

# Main function
def main():
    # Load the model
    model = load_model()
    
    st.title("🎤 Speech Emotion Analyzer")
    st.markdown("Upload an audio file or record your voice to detect emotion.")
    
    tab1, tab2 = st.tabs(["File Upload", "Record Audio"])
    
    with tab1:
        st.header("Analyze Audio File")
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])
        
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            file_path = f"temp_upload_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success("File uploaded successfully!")
            
            # Display waveform
            st.subheader("Audio Waveform")
            display_waveform(file_path)
            
            # Add audio player
            st.subheader("Listen to Audio")
            st.audio(file_path)
            
            # Make prediction
            if st.button("Analyze Emotion", key="analyze_upload"):
                with st.spinner("Analyzing..."):
                    prediction = predict_emotion(file_path, model)
                
                # Map prediction indices to emotion labels
                emotions = ['female_angry', 'female_calm', 'female_fearful', 'female_happy', 'female_sad', 
                           'male_angry', 'male_calm', 'male_fearful', 'male_happy', 'male_sad']
                
                # Get the predicted emotion
                predicted_emotion = emotions[np.argmax(prediction)]
                emotion_only = predicted_emotion.split('_')[1]
                gender = predicted_emotion.split('_')[0]
                
                # Display results
                st.subheader("Prediction Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Detected Emotion", emotion_only.capitalize())
                with col2:
                    st.metric("Detected Gender", gender.capitalize())
                
                # Display probabilities
                st.subheader("Emotion Probabilities")
                probs_df = pd.DataFrame({
                    'Emotion': emotions,
                    'Probability': prediction[0] * 100
                })
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(probs_df['Emotion'], probs_df['Probability'])
                ax.set_ylabel('Probability (%)')
                ax.set_title('Emotion Prediction Probabilities')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Clean up
                if os.path.exists(file_path):
                    os.remove(file_path)
    
    with tab2:
        st.header("Record and Analyze")
        st.write("Click the button below to record your voice for analysis.")
        
        if st.button("Start Recording", key="start_record"):
            # Record audio
            audio_path = record_audio(duration=4, filename=f"recording_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav")
            
            # Display waveform
            st.subheader("Audio Waveform")
            display_waveform(audio_path)
            
            # Add audio player
            st.subheader("Listen to Audio")
            st.audio(audio_path)
            
            # Make prediction
            with st.spinner("Analyzing..."):
                prediction = predict_emotion(audio_path, model)
            
            # Map prediction indices to emotion labels
            emotions = ['female_angry', 'female_calm', 'female_fearful', 'female_happy', 'female_sad', 
                       'male_angry', 'male_calm', 'male_fearful', 'male_happy', 'male_sad']
            
            # Get the predicted emotion
            predicted_emotion = emotions[np.argmax(prediction)]
            emotion_only = predicted_emotion.split('_')[1]
            gender = predicted_emotion.split('_')[0]
            
            # Display results
            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Detected Emotion", emotion_only.capitalize())
            with col2:
                st.metric("Detected Gender", gender.capitalize())
            
            # Display probabilities
            st.subheader("Emotion Probabilities")
            probs_df = pd.DataFrame({
                'Emotion': emotions,
                'Probability': prediction[0] * 100
            })
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(probs_df['Emotion'], probs_df['Probability'])
            ax.set_ylabel('Probability (%)')
            ax.set_title('Emotion Prediction Probabilities')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Clean up
            if os.path.exists(audio_path):
                os.remove(audio_path)

if __name__ == "__main__":
    main()