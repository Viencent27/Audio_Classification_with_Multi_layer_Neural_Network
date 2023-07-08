import matplotlib.pyplot as plt
import librosa.display
import librosa
import soundfile as sf
import numpy as np
import streamlit as st
from Machine_Learning import evaluate_model
from Audio_Preprocessing import preprocess_audio

def deploy_web_app(best_model, X_test, y_test, target_sample_rate):
  """
  Mendeploy model terbaik ke dalam aplikasi web menggunakan Streamlit.
  """
  st.title("Identifikasi Sentimen dari Suara")
  st.write("Aplikasi ini menggunakan model neural network untuk mengidentifikasi sentimen atau emosi dari suara.")

  audio_files = st.file_uploader("Upload satu atau beberapa file audio", type=".wav", accept_multiple_files=True)

  if audio_files is not None:
    for audio_file in audio_files:
      mfccs_mean, mfccs = preprocess_audio(audio_file, target_sample_rate)
      input_data = np.array([mfccs_mean])
      sentiment = "Positive" if np.argmax(best_model.predict(input_data)) == 0 else "Negative"

      accuracy, precision, recall, f1 = evaluate_model(best_model, X_test, y_test)
      st.write("Hasil identifikasi sentimen:")
      st.write(f"File: {audio_file.name}")
      st.write(f"Sentimen: {sentiment}")
      st.write(f"Akurasi: {accuracy}")
      st.write(f"Presisi: {precision}")
      st.write(f"Recall: {recall}")
      st.write(f"F1-Score: {f1}")
      st.write("=========================================")

      st.write("Cepstral Coefficients:")
      for i, coef in enumerate(mfccs_mean):
          st.write(f"MFCC {i+1}: {coef}")

      st.write("MFCC Chart Analysis:")
      plt.figure(figsize=(10, 4))
      librosa.display.specshow(mfccs, x_axis='time')
      plt.colorbar(format='%+2.0f dB')
      plt.title('MFCC')
      plt.xlabel('Time')
      plt.ylabel('MFCC Coefficients')
      st.pyplot(plt)

      st.write("======================================================================================")